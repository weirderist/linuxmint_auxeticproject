#!/usr/bin/env python3
"""relax_job.py - run the baseline -> relaxation transition for one result folder. This is what the UI's Simulate button launches (as a subprocess).
Usage: python3 relax_job.py --folder webui/results/<id> --E <modulus> --mu <friction> --k-contact <kc> [--perturb 0.15] [--outer 40] [--maxiter 300] [--shell [--shell-w 1.0]]
Units follow the folder: mm if built with --ne --cpi --wpi (E in N/mm^2 = MPa, kc in N/mm), otherwise R-units (dimensionless).
Shell option: first the thin-shell model (shell_sim.py, Niu et al. 2025) is run on the stitch matrix to get the folded shape. The yarn is then moved onto that
shape (each vertex keeps its offset from the sheet along the sheet normal) and the loops are relaxed with the mean position of every stitch-sized cell held
toward the shell shape, with stiffness shell-w x EA/R per vertex (0 = start from the fold but do not hold it).
Writes into the folder: state.json (progress, atomic, every outer iteration), relaxed_axis.csv (course,x,y,z), relaxed_tube.obj, shell.npz (with --shell). Never touches meta.json.
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relax_state as rs, relax_solver as sv
from build_knit import write_obj

SHELL_KAPPA_C, SHELL_H_C = 1.2, 0.5/1.2     # Niu et al.: kappa = 1 and h = 0.5 for a stitch 1.4 x 1.2. Written per course spacing C so they scale with the fabric.

def run_shell(g, meta):
    """Fold the stitch matrix with shell_sim. Returns the shell vertex grid (gy, gx, 3) and its layout; also saved as shell.npz in the result folder."""
    import shell_sim as ss
    M, tiles, W, C = meta['matrix'], meta['tiles'], meta['W'], meta['C']; kappa, h = SHELL_KAPPA_C/C, SHELL_H_C*C
    print('shell stage: %d x %d stitches, sub %d, %d growth steps, kappa %.4g, h %.4g' % (len(M[0])*tiles, len(M)*tiles, g.shell_sub, g.shell_steps, kappa, h), flush=True)
    t = time.time()
    X0, X, tri, (nx, ny), log = ss.fold(M, W, C, kappa=kappa, h=h, sub=g.shell_sub, ntile=(tiles, tiles), steps=g.shell_steps, maxiter=g.shell_iter, verbose=True)
    Xg = X.reshape(ny*g.shell_sub + 1, nx*g.shell_sub + 1, 3)
    print('shell stage done in %.0f s: z-range %.4g, last step used %d iterations' % (time.time() - t, np.ptp(Xg[..., 2]), log[-1][2]), flush=True)
    np.savez(os.path.join(g.folder, 'shell.npz'), X=Xg, W=W, C=C, sub=g.shell_sub, nx=nx, ny=ny, kappa=kappa, h=h)
    return dict(X=Xg, W=W, C=C, sub=g.shell_sub, nx=nx, ny=ny)

def shell_field(sh):
    """Displacement U (shell position minus flat position) and unit normal N at every shell grid vertex; +z side = knit face."""
    Xg, W, C, sub = sh['X'], sh['W'], sh['C'], sh['sub']; gy, gx, _ = Xg.shape
    ii, jj = np.meshgrid(np.arange(gx), np.arange(gy)); flat = np.stack([ii*W/sub, jj*C/sub, np.zeros((gy, gx))], -1)
    N = np.cross(np.gradient(Xg, axis=1), np.gradient(Xg, axis=0)); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-30
    N *= np.where(N[..., 2:3] < 0, -1.0, 1.0)                                  # keep the normal on the +z side (valid until a fold turns over)
    return Xg - flat, N

def shell_targets(sh, U, N, P):
    """Where each baseline vertex P goes on the folded sheet: its flat (x, y) plus the sheet displacement there, plus its out-of-plane offset z along the sheet normal.
    Outside the shell grid the nearest grid value is used."""
    W, C, sub = sh['W'], sh['C'], sh['sub']; gy, gx, _ = U.shape
    u = np.clip(P[:, 0]/(W/sub), 0, gx - 1); v = np.clip(P[:, 1]/(C/sub), 0, gy - 1)
    i0 = np.minimum(u.astype(int), gx - 2); j0 = np.minimum(v.astype(int), gy - 2); a = (u - i0)[:, None]; b = (v - j0)[:, None]
    bil = lambda F: F[j0, i0]*(1-a)*(1-b) + F[j0, i0+1]*a*(1-b) + F[j0+1, i0]*(1-a)*b + F[j0+1, i0+1]*a*b
    n = bil(N); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    return np.c_[P[:, :2], np.zeros(len(P))] + bil(U) + P[:, 2:3]*n

def shell_cells(P, sh):
    """Stitch-sized cell (W x C, from the baseline x, y) of every vertex: (cell index per vertex, number of cells)."""
    ci = np.clip((P[:, 0]//sh['W']).astype(int), 0, sh['nx'] - 1); cj = np.clip((P[:, 1]//sh['C']).astype(int), 0, sh['ny'] - 1)
    uniq, cell = np.unique(cj*sh['nx'] + ci, return_inverse=True)
    return cell, len(uniq)

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--folder', required=True); a.add_argument('--E', type=float, required=True, help='effective yarn modulus')
    a.add_argument('--mu', type=float, required=True, help='yarn-yarn friction coefficient'); a.add_argument('--k-contact', type=float, required=True)
    a.add_argument('--perturb', type=float, default=0.15, help='out-of-plane symmetry-breaking amplitude, fraction of R (numerical device)')
    a.add_argument('--outer', type=int, default=40); a.add_argument('--maxiter', type=int, default=300); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--sides', type=int, default=8)
    a.add_argument('--curl', type=float, default=0.0, help='stored curl, turning angle (rad) per vertex, front loops +, back loops -')
    a.add_argument('--tension', type=float, default=0.0, help='feeder tension in cN (mm results only); the yarn is knitted stretched by this force')
    a.add_argument('--creep-tau', type=float, default=0.0, help='creep relaxation time in hours (0 = no creep)'); a.add_argument('--hours', type=float, default=0.0, help='simulated duration in hours (with creep)')
    a.add_argument('--shell', action='store_true', help='run the shell model first and relax the loops toward its folded shape')
    a.add_argument('--shell-w', type=float, default=1.0, help='hold strength on each stitch cell, x EA/R per vertex (0 = start from the fold only)')
    a.add_argument('--shell-sub', type=int, default=2, help='shell mesh cells per stitch side'); a.add_argument('--shell-steps', type=int, default=15); a.add_argument('--shell-iter', type=int, default=400)
    g = a.parse_args()
    meta = json.load(open(os.path.join(g.folder, 'meta.json')))
    EA, EI = sv.section(g.E, meta['R'])
    if g.tension and meta.get('structure', {}).get('units') != 'mm': sys.exit('feeder tension needs a result built in mm (--ne --cpi --wpi)')
    if g.curl and 'matrix' not in meta: sys.exit('curl needs the stitch matrix in meta.json (result built by an older pipeline)')
    if g.shell and 'matrix' not in meta: sys.exit('the shell option needs the stitch matrix in meta.json (result built by an older pipeline)')
    if g.creep_tau and g.hours <= 0: sys.exit('creep needs --hours > 0')
    shell = run_shell(g, meta) if g.shell else None
    ratio = 1.0/(1.0 + 0.01*g.tension/EA)                          # rest length = baseline length / (1 + T/EA), T in N (cN x 0.01)
    p = rs.Params(mu=g.mu, EA=EA, EI=EI, GJ=0.0, k_contact=g.k_contact, outer_max=g.outer, rest_length_ratio=ratio, curl=g.curl, creep_tau=g.creep_tau, hours=g.hours)       # GJ unused: twist not in the energy (see relax_solver)
    a_creep = 1.0 - math.exp(-(g.hours/g.outer)/g.creep_tau) if g.creep_tau else 0.0
    sim = rs.load(g.folder, p); rs.release(sim)
    e0 = sv.energy_parts(sim, g.k_contact, g.E); d0 = sv.min_axis_distance(sim)           # also fixes the reference curvature at the baseline
    if shell:
        U, N = shell_field(shell); P0 = np.vstack([y.X for y in sim.yarns]); T = shell_targets(shell, U, N, P0)
        cell, nc = shell_cells(P0, shell); cnt = np.bincount(cell, minlength=nc).astype(float)
        tgt = np.c_[[np.bincount(cell, weights=T[:, k], minlength=nc) for k in range(3)]].T/cnt[:, None]
        sim.shell = dict(cell=cell, tgt=tgt, cnt=cnt, ks=g.shell_w*EA/sim.R)
        o = 0
        for y in sim.yarns: y.X = T[o:o+len(y.X)].copy(); o += len(y.X)                   # start the loops on the folded shape
        es = sv.energy_parts(sim, g.k_contact, g.E)
        print('shell seed: %d stitch cells, hold %g x EA/R; energy on the folded shape %.6g (stretch %.4g bend %.4g contact %.4g shell %.4g)' % (nc, g.shell_w, es['total'], es['stretch'], es['bend'], es['contact'], es['shell']), flush=True)
    rng = np.random.default_rng(g.seed)
    for y in sim.yarns: y.X[:, 2] += g.perturb*sim.R*rng.standard_normal(len(y.X))
    print('curl %g, feeder tension %g cN (strain %.3g), creep tau %g h over %g h' % (g.curl, g.tension, 0.01*g.tension/EA, g.creep_tau, g.hours), flush=True)
    print('start: E %.6g (stretch %.4g bend %.4g contact %.4g), min axis distance %.4f = %.1f%% of 2R' % (e0['total'], e0['stretch'], e0['bend'], e0['contact'], d0, 100*d0/(2*sim.R)), flush=True)
    t0 = time.time()
    def relax(s):
        sv.relax(s, g.E, g.k_contact, maxiter=g.maxiter, verbose=True)
        if a_creep: sv.creep(s, a_creep)
    energy = lambda s: sv.energy_parts(s, g.k_contact, g.E)['total']
    rs.run(sim, relax, energy)
    e1 = sv.energy_parts(sim, g.k_contact, g.E); d1 = sv.min_axis_distance(sim)
    tot_l0 = sum(float(y.l0.sum()) for y in sim.yarns)
    print('end (%s, %.0f s): E %.6g (stretch %.4g bend %.4g contact %.4g shell %.4g), min axis distance %.4f = %.1f%% of 2R, total rest length %.6g' % (sim.state.value, time.time()-t0, e1['total'], e1['stretch'], e1['bend'], e1['contact'], e1['shell'], d1, 100*d1/(2*sim.R), tot_l0))
    dv = sv.shell_deviation(sim)
    if dv: print('shell hold: cell means are %.3g (rms) and %.3g (max) from the shell shape = %.2f and %.2f yarn radii' % (dv['rms'], dv['max'], dv['rms']/sim.R, dv['max']/sim.R))
    np.savetxt(os.path.join(g.folder, 'relaxed_axis.csv'), np.vstack([np.c_[np.full(len(y.X), y.course), y.X] for y in sim.yarns]), delimiter=',', header='course,x,y,z', comments='', fmt=['%d', '%.6f', '%.6f', '%.6f'])
    write_obj(os.path.join(g.folder, 'relaxed_tube.obj'), [y.X for y in sim.yarns], sim.R, g.sides, 'relaxed state of %s' % os.path.basename(os.path.abspath(g.folder)))
    d = json.load(open(os.path.join(g.folder, 'state.json'))); d['summary'] = dict(start=e0, end=e1, min_dist_over_2R_start=d0/(2*sim.R), min_dist_over_2R_end=d1/(2*sim.R))
    if dv: d['summary']['shell'] = dict(w=g.shell_w, rms_dev_over_R=dv['rms']/sim.R, max_dev_over_R=dv['max']/sim.R)
    json.dump(d, open(os.path.join(g.folder, 'state.json.tmp'), 'w'), indent=1); os.replace(os.path.join(g.folder, 'state.json.tmp'), os.path.join(g.folder, 'state.json'))
    sys.exit(0 if sim.state == rs.S.SETTLED else 2)

if __name__ == '__main__':
    main()