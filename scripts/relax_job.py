#!/usr/bin/env python3
"""relax_job.py - run the baseline -> relaxation transition for one result folder. This is what the UI's Simulate button launches (as a subprocess).
Usage: python3 relax_job.py --folder webui/results/<id> --E <modulus> --mu <friction> --k-contact <kc> [--perturb 0.15] [--outer 40] [--maxiter 300]
Units follow the folder: mm if built with --ne --cpi --wpi (E in N/mm^2 = MPa, kc in N/mm), otherwise R-units (dimensionless).
Writes into the folder: state.json (progress, atomic, every outer iteration), relaxed_axis.csv (course,x,y,z), relaxed_tube.obj. With --shell also shell.npz. Appends one entry per run (with every round) to the 'simulations' list in meta.json.
Shell option (--shell): first the thin-shell model (shell_sim.py, Niu et al. 2025) is run on the stitch matrix to get the folded shape. The yarn is then moved onto that shape
(each vertex keeps its offset from the sheet along the sheet normal) and the loops are relaxed with the mean position of every stitch-sized cell held toward the shell shape,
with stiffness shell-w x EA/R per vertex (0 = start from the fold but do not hold it). Use --ref-scale 1 with it. By default the stress-free curvature kb0 is also rotated into the local frame of the shell (--no-shell-frame restores the fixed global-frame kb0).
Without the shell (loops that fold on their own): --ramp K grows the stress-free state from the built loop shape to the target in K quasi-static stages (--ramp-iter L-BFGS iterations each),
--seed-stitch a tilts front loops by +a R and back loops by -a R out of the plane before the random perturbation, --stall-tol t ends the outer loop when energy, rest lengths and positions have stopped changing
(reported as 'settled', with the reason in the state history; not strict convergence). The end summary now also gives the size change of the relaxed fabric and the rest length per stitch.
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relax_state as rs, relax_solver as sv
from build_knit import write_obj

SHELL_KAPPA_C, SHELL_H_C = 1.2, 0.5/1.2     # Niu et al.: kappa = 1 and h = 0.5 for a stitch 1.4 x 1.2. Written per course spacing C so they scale with the fabric.

def run_shell(g, meta, kxy=None):
    """Fold the stitch matrix with shell_sim. Returns the shell vertex grid (gy, gx, 3) and its layout; also saved as shell.npz in the result folder."""
    import shell_sim as ss
    M, tiles, W, C = meta['matrix'], meta['tiles'], meta['W'], meta['C']; kappa, h = SHELL_KAPPA_C/C, SHELL_H_C*C
    print('shell stage: %d x %d stitches, sub %d, %d growth steps, h %.4g, digit-1 curvature kx %.4g ky %.4g (1/mm, > 0 toward +z)' % (len(M[0])*tiles, len(M)*tiles, g.shell_sub, g.shell_steps, h, *(kxy if kxy else (-kappa, kappa))), flush=True)
    t = time.time()
    X0, X, tri, (nx, ny), log = ss.fold(M, W, C, kappa=kappa, h=h, sub=g.shell_sub, ntile=(tiles, tiles), steps=g.shell_steps, maxiter=g.shell_iter, verbose=True, kx=None if kxy is None else kxy[0], ky=None if kxy is None else kxy[1])
    Xg = X.reshape(ny*g.shell_sub + 1, nx*g.shell_sub + 1, 3)
    print('shell stage done in %.0f s: z-range %.4g, last step used %d iterations' % (time.time() - t, np.ptp(Xg[..., 2]), log[-1][2]), flush=True)
    np.savez(os.path.join(g.folder, 'shell.npz'), X=Xg, W=W, C=C, sub=g.shell_sub, nx=nx, ny=ny, kappa=kappa, h=h)
    return dict(X=Xg, W=W, C=C, sub=g.shell_sub, nx=nx, ny=ny)

def shell_field(sh):
    """Displacement U (shell position minus flat position) and unit normal N at every shell grid vertex; +z side = knit face."""
    Xg, W, C, sub = sh['X'], sh['W'], sh['C'], sh['sub']; gy, gx, _ = Xg.shape
    ii, jj = np.meshgrid(np.arange(gx), np.arange(gy)); flat = np.stack([ii*W/sub, jj*C/sub, np.zeros((gy, gx))], -1)
    N = np.cross(np.gradient(Xg, axis=1), np.gradient(Xg, axis=0)); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-30
    if N[..., 2].mean() < 0: N = -N                                            # orientation follows the grid (x-derivative cross y-derivative), so it stays continuous even where the sheet rolls past vertical; only the global sign is fixed
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

def shell_frames(sh):
    """Orthonormal frame of the folded sheet at every shell grid vertex: (t1, t2, N) with t1 along the sheet x direction (course direction), N the +z-side normal, t2 = N x t1.
    Flat sheet gives the identity frame. Same normal construction as shell_field."""
    Xg = sh['X']; t1 = np.gradient(Xg, axis=1); N = np.cross(np.gradient(Xg, axis=1), np.gradient(Xg, axis=0))
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-30
    if N[..., 2].mean() < 0: N = -N                                            # continuous orientation, as in shell_field
    t1 = t1 - np.sum(t1*N, -1, keepdims=True)*N; t1 /= np.linalg.norm(t1, axis=-1, keepdims=True) + 1e-30
    return t1, np.cross(N, t1), N

def rotate_kb0(sim, sh, P):
    """Rotate the stress-free curvature vectors kb0 from the flat-sheet frame into the local frame of the folded sheet, as a normal is carried by the surface
    transform (Sperl, Narain, Wojtan 2021 supplement, S2.1: directors move with the surface deformation J = [F, n]; here the frame is the rotation part of J,
    which is an adaptation of that rendering step to curvature vectors, not their result). Without it kb0 stays a fixed vector in the global frame, which is only
    valid for small tilts. P: baseline (flat) positions of all vertices, stacked by yarn. Returns the largest surface tilt in degrees."""
    t1, t2, N = shell_frames(sh); W, C, sub = sh['W'], sh['C'], sh['sub']; gy, gx, _ = t1.shape
    u = np.clip(P[:, 0]/(W/sub), 0, gx - 1); v = np.clip(P[:, 1]/(C/sub), 0, gy - 1)
    i0 = np.minimum(u.astype(int), gx - 2); j0 = np.minimum(v.astype(int), gy - 2); a = (u - i0)[:, None]; b = (v - j0)[:, None]
    bil = lambda F: F[j0, i0]*(1-a)*(1-b) + F[j0, i0+1]*a*(1-b) + F[j0+1, i0]*(1-a)*b + F[j0+1, i0+1]*a*b
    e1 = bil(t1); e3 = bil(N); e3 /= np.linalg.norm(e3, axis=1, keepdims=True) + 1e-30
    e1 -= np.sum(e1*e3, 1, keepdims=True)*e3; e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-30; e2 = np.cross(e3, e1)
    off = np.r_[0, np.cumsum([len(y.X) for y in sim.yarns])]; ix = np.concatenate([np.arange(o + 1, o + len(y.X) - 1) for y, o in zip(sim.yarns, off)])    # interior vertices, same order as kb0
    k = sim.kb0; sim.kb0 = k[:, 0:1]*e1[ix] + k[:, 1:2]*e2[ix] + k[:, 2:3]*e3[ix]
    return float(np.degrees(np.arccos(np.clip(e3[:, 2], -1, 1))).max())

def shell_cells(P, sh):
    """Stitch-sized cell (W x C, from the baseline x, y) of every vertex: (cell index per vertex, number of cells)."""
    ci = np.clip((P[:, 0]//sh['W']).astype(int), 0, sh['nx'] - 1); cj = np.clip((P[:, 1]//sh['C']).astype(int), 0, sh['ny'] - 1)
    uniq, cell = np.unique(cj*sh['nx'] + ci, return_inverse=True)
    return cell, len(uniq)

def warp_sheet(sim, kx, ky):
    """Bend the whole flat fabric onto a cylinder (kx: curvature along x, the course direction) and/or (ky: along y, the wale direction) before relaxing, positive = toward +z, about the
    centre of the sheet. Lengths along the mid-surface are kept; yarn at height z gets the factor (1 - z k). This is a START, not a model: nothing holds the shape afterwards, so the relaxation
    shows whether the yarn energy keeps the curvature, lets it grow, or flattens it. Units: 1/(fabric unit)."""
    P = np.vstack([y.X for y in sim.yarns]); c = 0.5*(P.max(0) + P.min(0))
    for y in sim.yarns:
        for ax, k in ((0, kx), (1, ky)):
            if not k: continue
            s = y.X[:, ax] - c[ax]; z = y.X[:, 2]; rho = 1.0/k; th = k*s
            y.X[:, ax] = c[ax] + (rho - z)*np.sin(th); y.X[:, 2] = rho - (rho - z)*np.cos(th)

def extent(sim):
    """Size of the yarn axes' bounding box (x, y, z) in fabric units."""
    P = np.vstack([y.X for y in sim.yarns]); return P.max(0) - P.min(0)

def fit_curvature(sim):
    """Least-squares quadratic fit z(x, y) of all yarn vertices: (kx, ky) = 2 x the x^2 and y^2 coefficients, 1/(fabric unit). Same as fold_metric.py."""
    P = np.vstack([y.X for y in sim.yarns]); x = P[:, 0] - P[:, 0].mean(); y = P[:, 1] - P[:, 1].mean()
    c = np.linalg.lstsq(np.c_[x*x, y*y, x*y, x, y, np.ones_like(x)], P[:, 2], rcond=None)[0]; return 2*c[0], 2*c[1]

def fold_search(sim, g):
    """No shell: look for a folded state of the yarn energy by trying large-scale bends. The flat fabric is not a minimum of the yarn energy for every input (measured: with straight stress-free yarn on
    plain stockinette the relaxed energy falls steadily as the sheet is bent, courses toward +z and wales toward -z), but the slope is far too weak for the optimiser to find from flat. Each trial
    bends the current relaxed fabric by a step in one of four directions (kx +/-, ky +/-), relaxes it for fold_iter iterations and keeps it only if its energy is lower than that of a CONTROL: the same
    fabric relaxed for the same number of iterations without bending (otherwise any trial would win just by getting more relaxation). The last successful
    direction is tried first and the step grows by 1.5 after a success; after a round with no success it halves; the search ends at fold_search accepted steps or when the step is below 1/16 of the
    first. What it keeps is chosen by the yarn energy alone; nothing is prescribed. Energies of trials are compared after the same short relaxation, so tiny gains are within noise (accepted only if
    more than 1e-4 relative). The result is a lower-energy state reachable by large-scale bending, not a proof that it is the global minimum."""
    ext = extent(sim); delta = g.fold_delta or 0.25/max(ext[0], ext[1]); step = delta; energy = lambda: sv.energy_parts(sim, g.k_contact, g.E)['total']
    snap = lambda: [y.X.copy() for y in sim.yarns]
    def put(X):
        for y, x in zip(sim.yarns, X): y.X = x.copy()
    sv.relax(sim, g.E, g.k_contact, maxiter=g.fold_iter, rounds=1); base = snap(); E0 = energy(); k0 = fit_curvature(sim); last = None; steps = []; dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    print('fold search: start E %.6g, curvature kx %+.4g ky %+.4g, first step %.3g' % (E0, k0[0], k0[1], delta), flush=True)
    for it in range(g.fold_search):
        done = False; put(base); sv.relax(sim, g.E, g.k_contact, maxiter=g.fold_iter, rounds=1); E0 = energy(); ctrl = snap()      # control: no bend, same iterations
        for d in ([last] if last else []) + [q for q in dirs if q != last]:
            put(base); warp_sheet(sim, d[0]*step, d[1]*step); sv.relax(sim, g.E, g.k_contact, maxiter=g.fold_iter, rounds=1); E = energy()
            if E < E0 - 1e-4*abs(E0):
                base = snap(); kc = fit_curvature(sim); steps.append(dict(step=it + 1, dir=list(d), size=step, E_before=E0, E=E, kx=float(kc[0]), ky=float(kc[1]))); E0 = E; last = d; done = True
                print('  fold step %d: bend (%+d, %+d) x %.3g -> E %.6g, curvature kx %+.4g ky %+.4g' % (it + 1, d[0], d[1], steps[-1]['size'], E, kc[0], kc[1]), flush=True); step *= 1.5; break
        if not done:
            base = ctrl; step *= 0.5; last = None; print('  no bend beats the control (E %.6g): step halved to %.3g' % (E0, step), flush=True)
            if step < delta/16: break
    put(base); kc = fit_curvature(sim); print('fold search: %d accepted steps, E %.6g, curvature kx %+.4g ky %+.4g /unit' % (len(steps), E0, kc[0], kc[1]), flush=True)
    return dict(delta=delta, fold_iter=g.fold_iter, steps=steps, E_start=steps[0]['E_before'] if steps else E0, E_end=E0, kx=float(kc[0]), ky=float(kc[1]))

def run_ramp(sim, g, kb_base, ratio):
    """Quasi-static growth of the stress-free state, for runs WITHOUT the shell. Stage k of K moves the reference curvature from the built loop shape (zero bending energy,
    nothing moves) toward its target and the rest lengths from the built lengths toward the target ratio, f = k/K of the way, and relaxes the positions after each stage.
    The idea is the growth method of Niu, Dion, Kamien (PNAS 2025), which ramps the preferred curvature in steps from the previous solution, and the bounded stepwise
    rest-shape updates of Sperl et al. (2022, Sec. 7.2); applying it to the yarn loops is this project's choice, not a published yarn-level result. Rest lengths are not
    transferred between segments during the ramp (slip starts in the outer loop). Returns one record per stage."""
    kb_T = sim.kb0.copy(); l0_T = [y.l0.copy() for y in sim.yarns]; rec = []
    if np.abs(kb_T - kb_base).max() < 1e-12 and abs(ratio - 1.0) < 1e-12:
        print('ramp skipped: the target stress-free state equals the built one (reference scale 1, no curl, rest length ratio 1), so there is nothing to grow', flush=True); return rec
    t = time.time()
    for k in range(1, g.ramp + 1):
        f = k/g.ramp
        sim.kb0 = (1.0 - f)*kb_base + f*kb_T
        for y, l in zip(sim.yarns, l0_T): y.l0[:] = l if k == g.ramp else l/ratio*(1.0 - f*(1.0 - ratio))
        sv.relax(sim, g.E, g.k_contact, maxiter=g.ramp_iter)
        e = sv.energy_parts(sim, g.k_contact, g.E); z = float(np.ptp(np.vstack([y.X for y in sim.yarns])[:, 2])/sim.R)
        rec.append(dict(stage=k, f=f, E=e['total'], stretch=e['stretch'], bend=e['bend'], contact=e['contact'], z_range_over_R=z, seconds=round(time.time() - t, 1)))
        print('ramp %d/%d (f %.2f): E %.6g (stretch %.4g bend %.4g contact %.4g; not comparable between stages), z-range %.2f R, %.0f s' % (k, g.ramp, f, e['total'], e['stretch'], e['bend'], e['contact'], z, time.time() - t), flush=True)
    return rec

def parse_law(s):
    """'strain:cN,strain:cN,...' (strain as a fraction, force in cN as a tensile tester reports it) -> [(strain, force in N)]."""
    try: return [(float(a), 0.01*float(b)) for a, b in (t.split(':') for t in s.split(','))]
    except ValueError: raise ValueError("stretch law must look like 'strain:cN,strain:cN', e.g. 0.01:80,0.03:200,0.06:330")

def yarn_inputs(g, R):
    """Install the yarn properties named on the command line into the solver and the state module, and return (EA, EI). --ea / --ei replace the solid-rod values; --stretch-law replaces the
    Hookean stretch term by a measured curve whose first-point slope is EA. Without them this is the same call as before. mm results only (the inputs are N and N mm^2)."""
    sv.STRETCH_SCALE = g.stretch_scale; sv.BEND_SCALE = g.bend_scale; sv.EA_FIXED = g.ea or None; sv.EI_FIXED = g.ei or None; law = None
    if g.ea and g.stretch_law: raise ValueError('give either --ea or --stretch-law (the law already fixes EA from its first point)')
    if (g.ea or g.stretch_law) and g.stretch_scale != 1.0: raise ValueError('--stretch-scale does not apply together with --ea or --stretch-law')
    if g.ei and g.bend_scale != 1.0: raise ValueError('--bend-scale does not apply together with --ei')
    if g.stretch_law:
        pr = parse_law(g.stretch_law); law = rs.build_law([p[0] for p in pr], [p[1] for p in pr]); sv.EA_FIXED = law['EA']
    rs.LAW = law; sv.set_law(law)
    return sv.section(g.E, R)

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--folder', required=True); a.add_argument('--E', type=float, required=True, help='effective yarn modulus')
    a.add_argument('--mu', type=float, required=True, help='yarn-yarn friction coefficient'); a.add_argument('--k-contact', type=float, required=True)
    a.add_argument('--perturb', type=float, default=0.15, help='out-of-plane symmetry-breaking amplitude, fraction of R (numerical device)')
    a.add_argument('--outer', type=int, default=40); a.add_argument('--maxiter', type=int, default=300); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--sides', type=int, default=8)
    a.add_argument('--curl', type=float, default=0.0, help='stored curl, turning angle (rad) per vertex, front loops +, back loops -')
    a.add_argument('--curl-aniso', action='store_true', help='anticlastic curl: course-direction yarn opposite to wale-direction yarn')
    a.add_argument('--ref-scale', type=float, default=0.0, help='stress-free curvature = ref_scale x baseline loop curvature: 0 straight yarn (default), 1 knitted loop shape')
    a.add_argument('--tension', type=float, default=0.0, help='feeder tension in cN (mm results only); the yarn is knitted stretched by this force')
    a.add_argument('--creep-tau', type=float, default=0.0, help='creep relaxation time in hours (0 = no creep)'); a.add_argument('--hours', type=float, default=0.0, help='simulated duration in hours (with creep)')
    a.add_argument('--shell', action='store_true', help='run the shell model first and relax the loops toward its folded shape')
    a.add_argument('--shell-w', type=float, default=1.0, help='hold strength on each stitch cell, x EA/R per vertex (0 = start from the fold only)')
    a.add_argument('--shell-kappa', choices=('guess', 'tile'), default='guess', help='shell curvature: the fixed guess of Niu et al. (kx = -k, ky = +k, k = 1.2/C) or the natural curvature computed by tile_sim.py from the yarn energy')
    a.add_argument('--kappa-scale', type=float, default=1.0, help='calibration factor on the shell curvature (negative flips the fold direction); fit it to a measured fold height'); a.add_argument('--tile-ref', type=float, default=0.5, help='rest-curvature scale used inside the tile analysis (unsourced)')
    a.add_argument('--seed-only', action='store_true', help='quick mode (needs --shell): place the yarn on the shell fold, then 3 short relaxation rounds (150 iterations each) to clear local overlaps; not an equilibrium');
    a.add_argument('--bend-scale', type=float, default=1.0, help='multiplies the bending stiffness EI only (1 = solid rod; spun yarns are far softer, see relax_solver.BEND_SCALE)'); a.add_argument('--length-ratio', type=float, default=1.0, help='extra rest-length ratio eta (<1: yarn knitted tighter than the built geometry; Ding et al. 2023, A.7), on top of the feeder-tension ratio')
    a.add_argument('--straight', type=float, default=0.0, help='straightening factor s >= 0: extra bending energy pulling the yarn straight, s x EI/2 |k|^2 on top of the loop-holding term (0 = off). Yarn length is not changed.')
    a.add_argument('--stretch-scale', type=float, default=1.0, help='multiplies the axial stiffness EA only (1 = solid rod); lower values let rest length migrate more')
    a.add_argument('--no-shell-frame', action='store_true', help='keep kb0 as fixed vectors in the global frame (previous behaviour) instead of rotating it into the local frame of the shell')
    a.add_argument('--ramp', type=int, default=0, help='no shell: grow the stress-free state (reference curvature and rest length ratio) from the built loop shape to the target in this many quasi-static stages (0 = off)')
    a.add_argument('--ramp-iter', type=int, default=200, help='L-BFGS iterations per ramp stage'); a.add_argument('--seed-stitch', type=float, default=0.0, help='no shell: out-of-plane seed, front loops +a R and back loops -a R (0 = off; negative flips)')
    a.add_argument('--ea', type=float, default=0.0, help='measured axial stiffness EA in N (0 = solid rod from --E). mm results only')
    a.add_argument('--ei', type=float, default=0.0, help='measured bending stiffness EI in N mm^2 (numerically equal to cN cm^2; 0 = solid rod from --E). mm results only')
    a.add_argument('--stretch-law', default='', help="measured stretch curve 'strain:cN,strain:cN,...' (e.g. 0.01:80,0.03:200,0.06:330); replaces Hooke, EA = first-point slope. mm results only")
    a.add_argument('--macro', action='store_true', help='no shell: let the optimiser move the three soft large-scale bending modes (course curvature, wale curvature, twist) as variables of their own, besides the vertices; same energy, same minimum, only reachable')
    a.add_argument('--fold-search', type=int, default=0, help='no shell: up to this many accepted large-scale bending steps searched by the yarn energy (0 = off). See fold_search in this file')
    a.add_argument('--fold-iter', type=int, default=200, help='iterations of the relaxation after each trial bend')
    a.add_argument('--fold-delta', type=float, default=0.0, help='first bend step in 1/mm (0 = 0.25 / sheet size)')
    a.add_argument('--warp-kx', type=float, default=0.0, help='no shell: start from the sheet bent into a cylinder, curvature along the courses (1/mm, + toward +z); a start, not held')
    a.add_argument('--warp-ky', type=float, default=0.0, help='no shell: same along the wales')
    a.add_argument('--stall-tol', type=float, default=0.0, help='early stop: relative energy change per outer round below this (with no rest-length change and little movement) for --stall-rounds rounds ends the run as settled (0 = off)')
    a.add_argument('--stall-rounds', type=int, default=3); a.add_argument('--stall-move', type=float, default=0.01, help='largest vertex move per outer round allowed for the early stop, x R')
    a.add_argument('--shell-sub', type=int, default=2, help='shell mesh cells per stitch side'); a.add_argument('--shell-steps', type=int, default=15); a.add_argument('--shell-iter', type=int, default=400)
    a.add_argument('--resume', action='store_true', help='continue from checkpoint.npz in the folder (written after every outer iteration; survives a power cut)')
    g = a.parse_args()
    def set_stage(name):
        """job_stage.json tells the UI which part of the job is running (prepare, tile, shell, ramp, fold, relax). Atomic; removed at the end."""
        try:
            sp = os.path.join(g.folder, 'job_stage.json'); json.dump(dict(stage=name, t=time.time()), open(sp + '.tmp', 'w')); os.replace(sp + '.tmp', sp)
        except OSError: pass
    set_stage('prepare')
    meta = json.load(open(os.path.join(g.folder, 'meta.json')))
    if (g.ea or g.ei or g.stretch_law) and meta.get('structure', {}).get('units') != 'mm': sys.exit('--ea, --ei and --stretch-law are in N and mm: they need a result built in mm (--ne --cpi --wpi)')
    if (g.macro or g.fold_search) and g.shell: sys.exit('--macro and --fold-search are for runs without the shell')
    sv.MACRO = bool(g.macro)
    try: EA, EI = yarn_inputs(g, meta['R'])
    except ValueError as err: sys.exit(str(err))
    print('yarn: EA %.4g N, EI %.4g N mm^2 (%.3g x the solid rod of modulus E; EA %.3g x), stretch law %s' % (EA, EI, EI/(g.E*math.pi*meta['R']**4/4.0), EA/(g.E*math.pi*meta['R']**2), ('measured, %d points' % (len(rs.LAW['K']) - 1)) if rs.LAW else 'Hooke'), flush=True)
    if g.tension and meta.get('structure', {}).get('units') != 'mm': sys.exit('feeder tension needs a result built in mm (--ne --cpi --wpi)')
    if g.curl and 'matrix' not in meta: sys.exit('curl needs the stitch matrix in meta.json (result built by an older pipeline)')
    if g.creep_tau and g.hours <= 0: sys.exit('creep needs --hours > 0')
    if g.shell and 'matrix' not in meta: sys.exit('the shell option needs the stitch matrix in meta.json (result built by an older pipeline)')
    if g.shell and g.ref_scale == 0: print('warning: --shell with --ref-scale 0 (straight-yarn reference): the loops will open against the shell hold; use --ref-scale 1', flush=True)
    if g.ramp and g.shell: sys.exit('--ramp is for runs without the shell')
    if g.seed_stitch and g.shell: sys.exit('--seed-stitch is for runs without the shell')
    if g.seed_stitch and 'matrix' not in meta: sys.exit('the stitch seed needs the stitch matrix in meta.json (result built by an older pipeline)')
    kxy = None
    ck_path = os.path.join(g.folder, 'checkpoint.npz'); resuming = bool(g.resume and os.path.exists(ck_path) and not g.seed_only)
    if g.resume and not resuming: sys.exit('--resume: no checkpoint.npz in %s' % g.folder)
    shp = os.path.join(g.folder, 'shell.npz'); reuse_shell = bool(g.shell and resuming and os.path.exists(shp))
    if g.shell and not reuse_shell:
        if g.shell_kappa == 'tile':
            set_stage('tile'); import tile_sim as ts; sv.STRETCH_SCALE = 1.0; kxy = ts.stitch_kappas(meta['W'], meta['C'], meta['R'], g.E, g.k_contact/(g.E*math.pi*meta['R']), g.tile_ref); sv.STRETCH_SCALE = g.stretch_scale
        else: kxy = (-SHELL_KAPPA_C/meta['C'], SHELL_KAPPA_C/meta['C'])
        kxy = (g.kappa_scale*kxy[0], g.kappa_scale*kxy[1])
    if reuse_shell:
        z = np.load(shp); shell = dict(X=z['X'], W=float(z['W']), C=float(z['C']), sub=int(z['sub']), nx=int(z['nx']), ny=int(z['ny'])); print('resume: shell fold reloaded from shell.npz', flush=True)
    else:
        if g.shell: set_stage('shell')
        shell = run_shell(g, meta, kxy) if g.shell else None
    set_stage('prepare')
    ten_strain = rs.strain_at(0.01*g.tension/EA)                              # strain under the feeder tension: T/EA for Hooke, from the measured curve with --stretch-law
    ratio = g.length_ratio/(1.0 + ten_strain)                                 # rest length = baseline length / (1 + strain), T in N (cN x 0.01)
    p = rs.Params(mu=g.mu, EA=EA, EI=EI, GJ=0.0, k_contact=g.k_contact, outer_max=g.outer, rest_length_ratio=ratio, curl=g.curl, curl_aniso=g.curl_aniso, ref_scale=g.ref_scale, creep_tau=g.creep_tau, hours=g.hours, stall_tol=g.stall_tol, stall_rounds=g.stall_rounds, stall_move=g.stall_move)       # GJ unused: twist not in the energy (see relax_solver)
    a_creep = 1.0 - math.exp(-(g.hours/g.outer)/g.creep_tau) if g.creep_tau else 0.0
    sim = rs.load(g.folder, p)
    if (g.curl or g.seed_stitch) and sim.yarns[0].s is None: sys.exit('curl and the stitch seed need the stitch types: meta.json must hold the matrix, tiles and structure W and C')
    rs.release(sim)
    stc = meta.get('structure', {}); ncol = len(meta['matrix'][0])*meta['tiles'] if 'matrix' in meta and 'tiles' in meta else 0
    colf = (lambda y: np.clip(((y.X[:-1, 0] + y.X[1:, 0])/2//stc['W']).astype(int), 0, ncol-1)) if ncol and 'W' in stc else None
    l0_start = [(colf(y), y.l0.copy()) for y in sim.yarns] if colf else None          # stitch column of each segment at the start, and its rest length
    e0 = sv.energy_parts(sim, g.k_contact, g.E); d0 = sv.min_axis_distance(sim)
    kb_base = sv._kb_np(sim, sv.pack(sim)[2])[0].copy() if g.ramp else None; ext0 = extent(sim); ramp_rec = []          # built loop curvature (the ramp starts here) and built size (for the size change at the end)
    if g.straight > 0:      # EI/2 |k - k0|^2 + s EI/2 |k|^2 = (1+s) EI/2 |k - k0/(1+s)|^2 + const: the same minimiser as a yarn that is (1+s) times stiffer and has 1/(1+s) of the loop curvature
        sim.kb0 = sv.baseline_curvature(sim, sv.pack(sim)[2])/(1.0 + g.straight); sim.bend_scale = 1.0 + g.straight
        print('straightening factor %g: stress-free curvature x %.4g, bending stiffness x %.4g (energies below are not comparable with runs at 0)' % (g.straight, 1/(1+g.straight), 1+g.straight), flush=True)
    if shell:
        U, N = shell_field(shell); P0 = np.vstack([y.X for y in sim.yarns]); T = shell_targets(shell, U, N, P0)
        cell, nc = shell_cells(P0, shell); cnt = np.bincount(cell, minlength=nc).astype(float)
        tgt = np.c_[[np.bincount(cell, weights=T[:, k], minlength=nc) for k in range(3)]].T/cnt[:, None]
        sim.shell = dict(cell=cell, tgt=tgt, cnt=cnt, ks=g.shell_w*(EA/g.stretch_scale)/sim.R)
        if not g.no_shell_frame:
            kn0 = float(np.linalg.norm(sim.kb0)); ang = rotate_kb0(sim, shell, P0)
            print('shell frame: kb0 rotated into the local frame of the shell (largest surface tilt %.1f deg, |kb0| %.6g -> %.6g)' % (ang, kn0, float(np.linalg.norm(sim.kb0))), flush=True)
        o = 0
        for y in sim.yarns: y.X[:] = T[o:o+len(y.X)]; o += len(y.X)                      # start the loops on the folded shape
        es = sv.energy_parts(sim, g.k_contact, g.E)
        print('shell seed: %d stitch cells, hold %g x EA/R; energy on the folded shape %.6g (stretch %.4g bend %.4g contact %.4g shell %.4g)' % (nc, g.shell_w, es['total'], es['stretch'], es['bend'], es['contact'], es['shell']), flush=True)
    if g.seed_stitch:
        for y in sim.yarns: y.X[:, 2] += g.seed_stitch*sim.R*y.s                  # front (digit 1) loops toward +z, back (digit 2) toward -z: the sign of the knit/purl imbalance, instead of random noise alone
    if (g.warp_kx or g.warp_ky):
        if g.shell: sys.exit('--warp-kx/--warp-ky are for runs without the shell')
        warp_sheet(sim, g.warp_kx, g.warp_ky); print('warp: sheet bent to kx %g, ky %g 1/mm before the relaxation' % (g.warp_kx, g.warp_ky), flush=True)
    rng = np.random.default_rng(g.seed)
    for y in sim.yarns: y.X[:, 2] += g.perturb*sim.R*rng.standard_normal(len(y.X))
    k0 = 0
    if resuming:
        ck = np.load(ck_path, allow_pickle=False); k0 = int(ck['k'])
        for i, y in enumerate(sim.yarns): y.X[:] = ck['X%d' % i]; y.l0[:] = ck['l%d' % i]
        sim.kb0 = ck['kb0'].copy(); sim.energy_log = json.loads(str(ck['elog']))
        print('resume: checkpoint loaded, continuing at outer iteration %d of %d' % (k0 + 1, g.outer), flush=True)
    print('curl %g, feeder tension %g cN (strain %.3g), creep tau %g h over %g h' % (g.curl, g.tension, ten_strain, g.creep_tau, g.hours), flush=True)
    print('start: E %.6g (stretch %.4g bend %.4g contact %.4g), min axis distance %.4f = %.1f%% of 2R' % (e0['total'], e0['stretch'], e0['bend'], e0['contact'], d0, 100*d0/(2*sim.R)), flush=True)
    mp = os.path.join(g.folder, 'meta.json')
    try: t_start = json.load(open(os.path.join(g.folder, 'job.json')))['started']          # wall clock since Simulate was pressed, same clock as the UI timer
    except Exception: t_start = time.time()
    def meta_update(fn):
        m = json.load(open(mp)); fn(m); json.dump(m, open(mp + '.tmp', 'w'), indent=1); os.replace(mp + '.tmp', mp)
    run_entry = dict(started=time.strftime('%Y-%m-%dT%H:%M:%S'), status='running', finished=None, seconds=None,
                     params=dict(E=g.E, mu=g.mu, k_contact=g.k_contact, curl=g.curl, curl_aniso=bool(g.curl_aniso), ref_scale=g.ref_scale, shell=bool(g.shell), shell_w=g.shell_w, shell_frame=bool(g.shell and not g.no_shell_frame), shell_kappa=g.shell_kappa, kappa_scale=g.kappa_scale, kappa_xy=kxy, stretch_scale=g.stretch_scale, bend_scale=g.bend_scale, straight=g.straight, length_ratio=g.length_ratio, seed_only=bool(g.seed_only), ramp=g.ramp, ramp_iter=g.ramp_iter, seed_stitch=g.seed_stitch, ea=g.ea, ei=g.ei, stretch_law=g.stretch_law, macro=bool(g.macro), fold_search=g.fold_search, fold_iter=g.fold_iter, fold_delta=g.fold_delta, stall_tol=g.stall_tol, stall_rounds=g.stall_rounds, stall_move=g.stall_move, tension_cN=g.tension, creep_tau_h=g.creep_tau, hours=g.hours, outer=g.outer, maxiter=g.maxiter, perturb=g.perturb, seed=g.seed),
                     start=dict(e0, min_dist_over_2R=d0/(2*sim.R)), rounds=[], resumed_at_outer=(k0 if resuming else None))
    meta_update(lambda m: m.__setitem__('simulations', (m.get('simulations') or []) + [run_entry]))
    def save_checkpoint(knext):
        """Atomic: positions, rest lengths, reference curvature and energy log after outer iteration knext-1. A power cut loses at most one outer iteration."""
        d = {'X%d' % i: y.X for i, y in enumerate(sim.yarns)}; d.update({'l%d' % i: y.l0 for i, y in enumerate(sim.yarns)})
        tmp = os.path.join(g.folder, 'checkpoint.tmp.npz'); np.savez(tmp, k=knext, kb0=sim.kb0, elog=json.dumps(sim.energy_log), **d); os.replace(tmp, ck_path)
        json.dump(dict(next_outer_iteration=knext, outer=g.outer, saved=time.strftime('%Y-%m-%dT%H:%M:%S')), open(os.path.join(g.folder, 'checkpoint.json.tmp'), 'w')); os.replace(os.path.join(g.folder, 'checkpoint.json.tmp'), os.path.join(g.folder, 'checkpoint.json'))
    def on_round(k, E, dmax):
        pr = sv.energy_parts(sim, g.k_contact, g.E)
        rec = dict(round=k+1, t=round(time.time()-t_start, 2), E=pr['total'], stretch=pr['stretch'], bend=pr['bend'], contact=pr['contact'], shell=pr['shell'], max_dl0=dmax, min_dist_over_2R=sv.min_axis_distance(sim)/(2*sim.R))
        meta_update(lambda m: m['simulations'][-1]['rounds'].append(rec))
        if not g.seed_only: save_checkpoint(k + 1)
    if g.ramp and not resuming:
        set_stage('ramp'); ramp_rec = run_ramp(sim, g, kb_base, ratio)
        if ramp_rec: meta_update(lambda m: m['simulations'][-1].update(ramp=ramp_rec)); save_checkpoint(0)          # a power cut after the ramp resumes at the first outer round with the grown state
    fold_rec = None
    if g.fold_search and not resuming:
        set_stage('fold'); fold_rec = fold_search(sim, g); meta_update(lambda m: m['simulations'][-1].update(fold_search=fold_rec)); save_checkpoint(0)
    t0 = time.time()
    def relax(s):
        sv.relax(s, g.E, g.k_contact, maxiter=g.maxiter, verbose=True)
        if a_creep: sv.creep(s, a_creep)
    energy = lambda s: sv.energy_parts(s, g.k_contact, g.E)['total']
    if g.seed_only:
        if not g.shell: sys.exit('--seed-only needs --shell')
        g.maxiter = min(g.maxiter, 150); sim.to(rs.S.RELAXING, 'quick mode')
        for k in range(3):
            relax(sim); sim.contacts = rs.build_contacts(sim); e_k = energy(sim); sim.energy_log.append(dict(iter=k, E=e_k, max_dl0=0.0)); on_round(k, e_k, 0.0)
        sim.to(rs.S.SETTLED, 'quick mode: 3 short rounds, not an equilibrium')
    else: set_stage('relax'); rs.run(sim, relax, energy, on_round, k0=k0)
    e1 = sv.energy_parts(sim, g.k_contact, g.E); d1 = sv.min_axis_distance(sim)
    tot_l0 = sum(float(y.l0.sum()) for y in sim.yarns)
    print('end (%s, %.0f s): E %.6g (stretch %.4g bend %.4g contact %.4g), min axis distance %.4f = %.1f%% of 2R, total rest length %.6g' % (sim.state.value, time.time()-t0, e1['total'], e1['stretch'], e1['bend'], e1['contact'], d1, 100*d1/(2*sim.R), tot_l0))
    ext1 = extent(sim); nst = len(meta['matrix'])*len(meta['matrix'][0])*meta['tiles']**2 if meta.get('matrix') and meta.get('tiles') else None
    dims = dict(shrink_x=float(ext1[0]/ext0[0] - 1), shrink_y=float(ext1[1]/ext0[1] - 1), z_range_over_R_start=float(ext0[2]/sim.R), z_range_over_R_end=float(ext1[2]/sim.R), rest_length_per_stitch=(tot_l0/nst if nst else None))
    print('relaxed size against the built size: x %+.1f%%, y %+.1f%% (folding also shrinks these); thickness (z range) %.2f R -> %.2f R; rest length per stitch %s' % (100*dims['shrink_x'], 100*dims['shrink_y'], dims['z_range_over_R_start'], dims['z_range_over_R_end'], '%.5g' % dims['rest_length_per_stitch'] if nst else 'n/a'), flush=True)
    dv = sv.shell_deviation(sim)
    if dv: print('shell hold: shell energy %.4g; cell means are %.3g (rms) and %.3g (max) from the shell shape = %.2f and %.2f yarn radii' % (e1['shell'], dv['rms'], dv['max'], dv['rms']/sim.R, dv['max']/sim.R))
    meta_update(lambda m: m['simulations'][-1].update(status=sim.state.value, finished=time.strftime('%Y-%m-%dT%H:%M:%S'), seconds=round(time.time()-t_start, 1), end=dict(e1, min_dist_over_2R=d1/(2*sim.R))))
    wale = None
    if l0_start:
        base = np.zeros(ncol); dl = np.zeros(ncol)
        for y, (c, l) in zip(sim.yarns, l0_start): np.add.at(base, c, l); np.add.at(dl, c, y.l0 - l)
        wale = (dl/np.maximum(base, 1e-12)).tolist(); w = np.array(wale)
        print('rest-length change per wale: min %.3e (wale %d), max %.3e (wale %d), total %.3e' % (w.min(), w.argmin(), w.max(), w.argmax(), dl.sum()/base.sum()))
    np.savetxt(os.path.join(g.folder, 'relaxed_axis.csv'), np.vstack([np.c_[np.full(len(y.X), y.course), y.X] for y in sim.yarns]), delimiter=',', header='course,x,y,z', comments='', fmt=['%d', '%.6f', '%.6f', '%.6f'])
    np.savetxt(os.path.join(g.folder, 'relaxed_l0.csv'), np.vstack([np.c_[np.full(len(y.l0), y.course), np.arange(len(y.l0)), y.l0] for y in sim.yarns]), delimiter=',', header='course,seg,l0', comments='', fmt=['%d', '%d', '%.10g'])     # rest length of every segment after the relaxation (segment i joins rows i and i+1 of relaxed_axis.csv of that course); read by the run viewer for the axial strain
    write_obj(os.path.join(g.folder, 'relaxed_tube.obj'), [y.X for y in sim.yarns], sim.R, g.sides, 'relaxed state of %s' % os.path.basename(os.path.abspath(g.folder)))
    d = json.load(open(os.path.join(g.folder, 'state.json'))); d['summary'] = dict(start=e0, end=e1, min_dist_over_2R_start=d0/(2*sim.R), min_dist_over_2R_end=d1/(2*sim.R), wale_length_change=wale)
    if dv: d['summary']['shell'] = dict(w=g.shell_w, rms_dev_over_R=dv['rms']/sim.R, max_dev_over_R=dv['max']/sim.R)
    d['summary']['dims'] = dims; d['summary']['settled_by'] = (sim.history[-1].get('note', '') if sim.state == rs.S.SETTLED else '')
    if ramp_rec: d['summary']['ramp'] = ramp_rec
    if fold_rec: d['summary']['fold_search'] = fold_rec
    json.dump(d, open(os.path.join(g.folder, 'state.json.tmp'), 'w'), indent=1); os.replace(os.path.join(g.folder, 'state.json.tmp'), os.path.join(g.folder, 'state.json'))
    for f in ('checkpoint.npz', 'checkpoint.json', 'job_stage.json'):
        try: os.remove(os.path.join(g.folder, f))
        except OSError: pass
    sys.exit(0 if sim.state == rs.S.SETTLED else 2)

if __name__ == '__main__':
    main()