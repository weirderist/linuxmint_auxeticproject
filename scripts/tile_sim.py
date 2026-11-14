#!/usr/bin/env python3
"""tile_sim.py - periodic yarn-level tile solver (stage 1 of the two-scale pipeline). Reuses build_knit (geometry) and relax_solver (energy) unchanged.
Idea and procedure follow Sperl, Sanchez-Banderas, Li, Wojtan, Otaduy (2022), "Estimation of Yarn-Level Simulation Models for Production Fabrics", ACM TOG 41(4) 65:
  Sec 7.1  one periodic tile, coarse deformation q = (s, c, kappa) plus nodal yarn positions; Psi = (1/A) sum W_i                (eq. 8, 9)
  Sec 7.2  rest-shape estimation: alternate equilibrium solves with resetting rest length and rest curvature to the current state,
           rest length change bounded to 20 %; the paper reports that a stable, slightly pre-tensed rest state is essential
  Sec 5.1  stretch with zero orthogonal stress: compression c = argmin Psi (eq. 3); here nu = -eps_y / eps_x = c / s
  Sec 9.1  jersey knits showed curling in their yarn model as an offset bending stress at zero curvature; the paper leaves curling as future work.
What is NOT from the paper: this tile is one repeat of YOUR stitch matrix (not the paper's 8 cells), the energy is relax_solver's (Hookean stretch + Bergou
bending + vertex penalty contact, not the paper's two-phase stretch and barrier contact), no sliding constraint (rest lengths are fixed inside a solve),
and the natural-curvature search is this script's own addition: it minimises the tile energy over kappa as well, so kappa* is the curvature at zero bending stress.
Geometry: x = course direction (wale spacing W = 25.4/WPI), y = wale direction (course spacing C = 25.4/CPI). The tile is Lx = nx W by Ly = ny C. Each course is
cut out of a 3 x 3 repeat (periodicity check printed) and closed into a loop through its +x image. Contacts use the 3 x 3 neighbouring images.
Cylindrical bending (as in the paper, Sec 7.1): mode 'x' bends about an axis parallel to y (curvature along x), mode 'y' the reverse. Image of a vertex = rigid
rotation about the cylinder axis, so the lattice stays exactly periodic. A saddle (both curvatures at once) is not representable because the two lattice rotations
do not commute; kappa_x* and kappa_y* are therefore found one at a time, each with the other direction flat.
Sign: kappa > 0 bends toward +z (digit 1 = front loops are at +z). All unsourced inputs (mu is not used here; E, kc_ratio, ref_scale, ds) are as in relax_job.
Usage: python3 tile_sim.py --matrix "1 1/1 1" --ne 10 --cpi 14 --wpi 18 --out tile_out [--E 3700] [--kc-ratio 30] [--ref-scale 1] [--rounds 6] [--strain 0.05,0.1,0.2]
"""
import argparse, json, math, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relax_solver as sv, build_knit as bk
jax.config.update('jax_enable_x64', True)

BLK = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]          # image blocks; block index = (i+1)*3 + (j+1); (0,0) is block 4
sinc = lambda a: jnp.sinc(a/jnp.pi)

def xform(P, i, j, Lx, Ly, kap, mode):
    """Image (i, j) of the tile vertices P. mode 'x': rotation i*kap*Lx about the axis through (0, *, 1/kap) parallel to y, translation j*Ly along y.
    mode 'y': the same with the roles of x and y swapped. Written with sinc so kap = 0 is smooth: sin(a)/kap = i L sinc(a), (1-cos a)/kap = i L (a/2) sinc(a/2)^2."""
    if mode == 'x':
        a = i*kap*Lx; x, y, z = P[:, 0], P[:, 1], P[:, 2]
        return jnp.stack([x*jnp.cos(a) - z*jnp.sin(a) + i*Lx*sinc(a), y + j*Ly, x*jnp.sin(a) + z*jnp.cos(a) + i*Lx*0.5*a*sinc(a/2)**2], 1)
    a = j*kap*Ly; x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return jnp.stack([x + i*Lx, y*jnp.cos(a) - z*jnp.sin(a) + j*Ly*sinc(a), y*jnp.sin(a) + z*jnp.cos(a) + j*Ly*0.5*a*sinc(a/2)**2], 1)

# ------------------------------------------------------------------ tile geometry
def build_tile(M, W, C, R, ds, cache, swap_labels=True):
    """One closed loop per course of one matrix repeat. Returns (list of (n,3) loops, Lx, Ly, periodicity residual)."""
    nx, ny = len(M[0]), len(M); Lx, Ly = nx*W, ny*C
    solved, rep = bk.optimise_all(W/R, C/R, cache_dir=cache, verbose=False)
    rows, used = bk.build_rows(bk.tile_matrix(M, 3), solved, W, C, R, swap_labels=swap_labels)
    loops, res = [], 0.0
    for c in range(ny, 2*ny):
        P = bk.catmull_rom(rows[c], 16); i0 = int(np.argmax(P[:, 0] >= nx*W + 0.25*W))
        tgt = P[i0] + np.array([Lx, 0.0, 0.0]); j = i0 + 1 + int(np.argmin(np.linalg.norm(P[i0+1:] - tgt, axis=1)))
        res = max(res, float(np.linalg.norm(P[j] - tgt)))
        poly = P[i0:j+1].copy(); poly[-1] = poly[0] + [Lx, 0.0, 0.0]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]; n = max(8, int(round(s[-1]/ds)))
        u = np.arange(n)*s[-1]/n; loops.append(np.c_[[np.interp(u, s, poly[:, k]) for k in range(3)]].T)
    return loops, Lx, Ly, res

def topology(ns, N):
    """Segment and bend index arrays into the 9N image-stacked vertex array. Each yarn is a closed loop: vertex n-1 connects to image (+1, 0) of vertex 0."""
    seg, tri, o = [], [], 0
    for n in ns:
        ext = [(((k-1)//n)+1)*3*N + N + o + (k-1) % n for k in range(n+2)]        # block of image (i, 0) is (i+1)*3 + 1
        seg += [(ext[k], ext[k+1]) for k in range(1, n+1)]; tri += [(ext[k-1], ext[k], ext[k+1]) for k in range(1, n+1)]; o += n
    return np.array(seg), np.array(tri)

class Tile:
    def __init__(s, loops, Lx, Ly, R, E, kc, ref_scale, kb_mode='x'):
        s.R, s.Lx, s.Ly, s.kc = R, Lx, Ly, kc; s.EA, s.EI = sv.section(E, R)
        s.ns = [len(l) for l in loops]; s.N = sum(s.ns); s.X = np.vstack(loops); s.seg, s.tri = topology(s.ns, s.N)
        s.yid = np.concatenate([[k]*n for k, n in enumerate(s.ns)]); s.off = np.r_[0, np.cumsum(s.ns)]
        s.q = np.zeros(3); s.mode = 'x'; s.vg = None
        s.l0 = s.seg_len(); s.l_base = s.l0.copy(); s.arc = np.concatenate([np.r_[0, np.cumsum(s.l0[o:o+n-1])] for o, n in zip(s.off[:-1], s.ns)])
        s.Ltot = np.array([s.l0[o:o+n].sum() for o, n in zip(s.off[:-1], s.ns)])
        s.kb0 = ref_scale*s.kb(); s.cell = np.zeros(9*s.N, int)
    def set_mode(s, mode):
        if mode == s.mode and s.vg is not None: return
        s.mode = mode; N, seg, tri, EA, EI, kc, R2, cell = s.N, s.seg, s.tri, s.EA, s.EI, s.kc, 2*s.R, s.cell
        tgt, cnt, Lx0, Ly0, Lk = np.zeros((1, 3)), np.ones(1), s.Lx, s.Ly, (s.Lx if mode == 'x' else s.Ly)
        def Efun(x, q, l0, lbar, kb0, pairs, pm):
            X = x.reshape(-1, 3); Lx = Lx0*(1 + q[0]); Ly = Ly0*(1 + q[1]); kap = q[2]/Lk          # q[2] = kappa * L0 = bending angle across the tile
            Xall = jnp.concatenate([xform(X, i, j, Lx, Ly, kap, mode) for (i, j) in BLK])
            return sv._energy(Xall.ravel(), l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, kc, R2, cell, tgt, cnt, 0.0)
        s.vg = jax.jit(jax.value_and_grad(Efun, argnums=(0, 1)))
    def images(s):
        Lx, Ly = s.Lx*(1 + s.q[0]), s.Ly*(1 + s.q[1]); Lk = s.Lx if s.mode == 'x' else s.Ly
        return np.asarray(jnp.concatenate([xform(jnp.asarray(s.X), i, j, Lx, Ly, s.q[2]/Lk, s.mode) for (i, j) in BLK]))
    def seg_len(s):
        out = []; o = 0
        for n in s.ns:
            Y = s.X[o:o+n]; Z = np.vstack([Y, Y[:1] + [s.Lx, 0, 0]]); out.append(np.linalg.norm(np.diff(Z, axis=0), axis=1)); o += n
        return np.concatenate(out)
    def kb(s):
        Xall = s.images() if hasattr(s, 'mode') and hasattr(s, 'vg') else None
        if Xall is None:                                                    # called before the first mode is set: flat images by hand
            parts = [s.X + [i*s.Lx, j*s.Ly, 0] for (i, j) in BLK]; Xall = np.vstack(parts)
        e1 = Xall[s.tri[:, 1]] - Xall[s.tri[:, 0]]; e2 = Xall[s.tri[:, 2]] - Xall[s.tri[:, 1]]
        n1 = np.linalg.norm(e1, axis=1); n2 = np.linalg.norm(e2, axis=1)
        return 2.0*np.cross(e1, e2)/(n1*n2 + np.sum(e1*e2, 1) + 1e-12)[:, None]
    def lbar(s):
        out = []
        for o, n in zip(s.off[:-1], s.ns): l = s.l0[o:o+n]; out.append(0.5*(np.roll(l, 1) + l))
        return np.concatenate(out)
    def pairs(s, pad=4096, skin=1.5, skip_vert=6):
        """Vertex a of the tile with vertex b of image (i, j), closer than skin*2R; each unordered pair once; neighbours along one yarn skipped."""
        N = s.N; Xall = s.images(); Xt = Xall[4*N:5*N]; lists = cKDTree(Xall).query_ball_point(Xt, skin*2*s.R)
        cnt = np.array([len(l) for l in lists]); a = np.repeat(np.arange(N), cnt); b = np.concatenate([np.asarray(l, int) for l in lists])
        v, i, j = b % N, (b//N)//3 - 1, (b//N) % 3 - 1
        keep = (a < v) | ((a == v) & ((i > 0) | ((i == 0) & (j > 0))))
        same = (s.yid[a] == s.yid[v]) & (j == 0)
        keep &= ~(same & (np.abs(s.arc[a] - (s.arc[v] + i*s.Ltot[s.yid[v]])) < skip_vert*np.median(s.l_base)))
        pr = np.c_[4*N + a[keep], b[keep]]; n = max(pad, int(np.ceil(len(pr)/pad))*pad)
        out = np.zeros((n, 2), int); m = np.zeros(n); out[:len(pr)] = pr; m[:len(pr)] = 1.0
        return out, m, len(pr)
    def evalg(s, x, q, pr):
        v, (gx, gq) = s.vg(x, q, s.l0, s.lbar(), s.kb0, pr[0], pr[1]); return float(v), np.asarray(gx, float), np.asarray(gq, float)
    def solve(s, free=(), maxiter=1500, rounds=4, verbose=False):
        """Minimise over nodal positions (and the coarse variables listed in free: 0 = s_x, 1 = s_y, 2 = bending angle). Contact pairs rebuilt between rounds."""
        free = list(free); n3 = 3*s.N
        for r in range(rounds):
            pr = s.pairs()
            def f(z):
                q = s.q.copy(); q[free] = z[n3:]; v, gx, gq = s.evalg(z[:n3], q, pr); return v, np.r_[gx, gq[free]]
            res = minimize(f, np.r_[s.X.ravel(), s.q[free]], jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxcor=30, ftol=1e-13, gtol=1e-9))
            s.X = res.x[:n3].reshape(-1, 3).copy(); s.q[free] = res.x[n3:]
            if verbose: print('     solve round %d: %d pairs, E %.6g, %d its' % (r+1, pr[2], res.fun, res.nit), flush=True)
            if res.nit < maxiter: break
        return float(res.fun)
    def stress(s):
        """Coarse stress per rest area at the current state: dE/dq / (Lx Ly). q[2] is the bending angle, so the third entry is dE/d(kappa) / A = bending moment per area."""
        pr = s.pairs(); v, gx, gq = s.evalg(s.X.ravel(), s.q, pr); A = s.Lx*s.Ly
        return dict(E=v, sx=gq[0]/A, sy=gq[1]/A, m=gq[2]/(A*(s.Lx if s.mode == 'x' else s.Ly)), gx_max=float(np.abs(gx).max()))
    def copy(s):
        import copy as cp; t = cp.copy(s); t.X, t.q, t.l0, t.kb0 = s.X.copy(), s.q.copy(), s.l0.copy(), s.kb0.copy(); return t

# ------------------------------------------------------------------ Sperl Sec 7.2: rest-shape estimation
def rest_shape(t, rounds=6, a=1.0, bound=0.2, it_first=200, it_last=1500, log=print):
    """Alternate flat equilibrium solves (q = 0) with resetting l0 and kb0 a fraction a of the way to the current state (a = 1: fully). The change of l0 is bounded
    to +-bound of the baseline length (paper: 20 %). First rounds are short (contact forces are strong and localised), the last run toward convergence."""
    t.set_mode('x'); hist = []
    for k in range(rounds):
        its = int(it_first + (it_last - it_first)*k/max(1, rounds - 1)); E = t.solve((), maxiter=its, rounds=2)
        St = t.stress(); l = t.seg_len(); kb = t.kb()
        t.l0 = np.clip(t.l0 + a*(l - t.l0), t.l_base*(1 - bound), t.l_base*(1 + bound)); t.kb0 = t.kb0 + a*(kb - t.kb0)
        hist.append(dict(round=k+1, E=E, sx=St['sx'], sy=St['sy'], max_dl0_rel=float(np.abs(t.l0/t.l_base - 1).max())))
        log('  rest round %d: E %.5g (%d its), flat stress sx %.4g sy %.4g, max |l0/l_base-1| %.3f' % (k+1, E, its, St['sx'], St['sy'], hist[-1]['max_dl0_rel']))
    return hist

# ------------------------------------------------------------------ analyses
def natural_curvature(t0, mode, delta=0.05, maxiter=1500, log=print):
    """Bending about one axis, other direction flat. Returns kappa at zero bending stress (free minimisation), the flat bending moment per area, and the
    bending stiffness per area from central differences of the relaxed energy at +-delta (radians across the tile)."""
    t = t0.copy(); t.q[:] = 0; t.mode = None; t.vg = None; t.set_mode(mode); ax = 0 if mode == 'x' else 1; L = t.Lx if mode == 'x' else t.Ly; A = t.Lx*t.Ly
    t.X = t.X - np.r_[[t.X[:, k].mean() if k in (ax, 2) else 0.0 for k in range(3)]]; base = t.X.copy()          # centre the tile on the cylinder axis plane
    E = {}
    for name, ang in (('0', 0.0), ('+', delta), ('-', -delta)):
        t.X = base.copy(); t.q[:] = 0; t.q[2] = ang; E[name] = t.solve((), maxiter=maxiter); log('   %s bend %+.3g rad: E %.6g' % (mode, ang, E[name]))
    k_b = (E['+'] + E['-'] - 2*E['0'])/(delta/L)**2/A; m0 = (E['+'] - E['-'])/(2*delta/L)/A
    kq = -m0/k_b if k_b > 0 else float('nan')
    t.X = base.copy(); t.q[:] = 0; Ef = t.solve((2,), maxiter=maxiter*2, rounds=6); kap = t.q[2]/L
    log('   %s: flat moment/area %.4g, stiffness/area %.4g -> quadratic kappa* %.4g; free minimisation kappa* %.4g (E %.6g -> %.6g)' % (mode, m0, k_b, kq, kap, E['0'], Ef))
    return dict(mode=mode, kappa_free=float(kap), kappa_quadratic=float(kq), kappa_free_times_R=float(kap*t.R), flat_moment_per_area=float(m0), bending_stiffness_per_area=float(k_b),
                E_flat=E['0'], E_free=Ef, E_plus=E['+'], E_minus=E['-'], delta=delta)

def poisson_curve(t0, strains, direction='x', maxiter=1500, log=print):
    """Stretch along x (or y) with the orthogonal stretch free (zero orthogonal stress, paper eq. 3). nu = -eps_perp / s."""
    t = t0.copy(); t.q[:] = 0; t.mode = None; t.vg = None; t.set_mode('x'); i_s, i_f = (0, 1) if direction == 'x' else (1, 0); out = []
    for s in strains:
        prev = t.q.copy(); t.q[i_s] = s; sc = np.ones(3); sc[i_s] = (1 + s)/(1 + prev[i_s]); t.X = t.X*sc
        E = t.solve((i_f,), maxiter=maxiter, rounds=4); eps = t.q[i_f]; out.append(dict(strain=float(s), eps_perp=float(eps), nu=float(-eps/s), E=E))
        log('   stretch %.3g along %s: orthogonal strain %+.4g, nu %+.4g, E %.6g' % (s, direction, eps, -eps/s, E))
    return out

def run_tile(M, W, C, R, E=3700.0, kc_ratio=30.0, ref_scale=1.0, ds_over_R=1.0, rounds=6, reset=1.0, strains=(0.05, 0.1, 0.2), delta=0.05, maxiter=1500, perturb=0.02,
             seed=0, do=('rest', 'curl', 'poisson'), cache=None, labels_swapped=True, log=lambda m: print(m, flush=True), save=None, max_verts=3000):
    cache = cache or os.path.join(os.path.dirname(os.path.abspath(bk.__file__)), 'cell_cache'); t0 = time.time()
    loops, Lx, Ly, res = build_tile(M, W, C, R, ds_over_R*R, cache, labels_swapped)
    log('tile %d x %d stitches, Lx %.4g Ly %.4g, %d courses, %d vertices; periodicity residual %.2g (%.2g R)' % (len(M[0]), len(M), Lx, Ly, len(loops), sum(len(l) for l in loops), res, res/R))
    if sum(len(l) for l in loops) > max_verts: sys.exit('tile has %d vertices, above --max-verts %d: use a smaller matrix or a larger --ds' % (sum(len(l) for l in loops), max_verts))
    t = Tile(loops, Lx, Ly, R, E, kc_ratio*E*math.pi*R, ref_scale)
    t.X = t.X + perturb*R*np.random.default_rng(seed).standard_normal(t.X.shape)*np.array([0, 0, 1.0]); out = dict(Lx=Lx, Ly=Ly, R=R, n_vertices=t.N, periodicity_residual=res)
    st0 = (t.set_mode('x'), t.stress())[1]; log('start: E %.6g (stretch+bend+contact), flat stress sx %.4g sy %.4g' % (st0['E'], st0['sx'], st0['sy'])); out['start'] = st0
    save and save(out)
    if 'rest' in do: out['rest'] = rest_shape(t, rounds, reset, log=log); out['rest_state'] = t.stress(); save and save(out)
    if 'curl' in do:
        out['curl'] = []
        for m in ('x', 'y'): out['curl'].append(natural_curvature(t, m, delta, maxiter, log)); save and save(out)
    if 'poisson' in do:
        out['poisson_x'] = poisson_curve(t, strains, 'x', maxiter, log); save and save(out); out['poisson_y'] = poisson_curve(t, strains, 'y', maxiter, log)
    out['seconds'] = time.time() - t0; return out, t

def main():
    a = argparse.ArgumentParser(); a.add_argument('--folder', help='web UI result folder: matrix, W, C, R are read from its meta.json and the outputs are written into it'); a.add_argument('--max-verts', type=int, default=3000)
    a.add_argument('--matrix', help='rows separated by /, digits 1 and 2, first row = bottom course (one repeat = one tile)')
    a.add_argument('--ne', type=float); a.add_argument('--cpi', type=float); a.add_argument('--wpi', type=float)
    a.add_argument('--W', type=float, default=16.0); a.add_argument('--C', type=float, default=9.0); a.add_argument('--R', type=float, default=1.0)
    a.add_argument('--E', type=float, default=3700.0, help='yarn modulus, MPa for mm results'); a.add_argument('--kc-ratio', type=float, default=30.0)
    a.add_argument('--ref-scale', type=float, default=1.0, help='initial rest curvature = ref_scale x baseline curvature (paper: rest = registered geometry, so 1)')
    a.add_argument('--ds', type=float, default=1.0, help='vertex spacing in yarn radii'); a.add_argument('--rounds', type=int, default=6, help='rest-shape reset rounds (Sperl 7.2)')
    a.add_argument('--reset', type=float, default=1.0, help='fraction of the way l0 and kb0 move to the current state per reset')
    a.add_argument('--strain', default='0.05,0.1,0.2'); a.add_argument('--delta', type=float, default=0.05, help='bending angle across the tile for the stiffness difference (rad)')
    a.add_argument('--maxiter', type=int, default=1500); a.add_argument('--perturb', type=float, default=0.02); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--do', default='rest,curl,poisson'); a.add_argument('--labels', choices=('swapped', 'printed'), default='swapped'); a.add_argument('--out', default='tile_out')
    g = a.parse_args(); W, C, R = g.W, g.C, g.R
    if g.folder:
        meta = json.load(open(os.path.join(g.folder, 'meta.json'))); M = meta['matrix']; W, C, R = meta['W'], meta['C'], meta['R']; g.out = g.folder
        if (meta.get('structure') or {}).get('swap_types'): M = [[3 - v for v in r] for r in M]
        g.labels = meta.get('labels', g.labels)
    elif not g.matrix: sys.exit('give --matrix or --folder')
    else: M = [[int(ch) for ch in row.replace(' ', '')] for row in g.matrix.split('/')]
    if g.folder: pass
    elif g.ne and g.cpi and g.wpi: R = 25.4/(28.0*math.sqrt(g.ne))/2; W = 25.4/g.wpi; C = 25.4/g.cpi
    elif g.ne or g.cpi or g.wpi: sys.exit('give all of --ne --cpi --wpi, or none')
    os.makedirs(g.out, exist_ok=True); rp = os.path.join(g.out, 'tile_result.json'); t_start = time.time()
    def save(o, status='running'):
        d = dict(o, status=status, args=vars(g), matrix=M, elapsed=time.time() - t_start); json.dump(d, open(rp + '.tmp', 'w'), indent=1); os.replace(rp + '.tmp', rp)
    out, t = run_tile(M, W, C, R, g.E, g.kc_ratio, g.ref_scale, g.ds, g.rounds, g.reset, tuple(float(x) for x in g.strain.split(',')), g.delta, g.maxiter, g.perturb, g.seed, tuple(g.do.split(',')), None, g.labels == 'swapped', save=save, max_verts=g.max_verts)
    save(out, 'done')
    np.savetxt(os.path.join(g.out, 'tile_axis.csv'), np.vstack([np.c_[np.full(n, k), t.X[o:o+n]] for k, (o, n) in enumerate(zip(t.off[:-1], t.ns))]), delimiter=',', header='course,x,y,z', comments='', fmt=['%d', '%.6f', '%.6f', '%.6f'])
    np.savez(os.path.join(g.out, 'tile_state.npz'), X=t.X, q=t.q, l0=t.l0, kb0=t.kb0, ns=t.ns, Lx=t.Lx, Ly=t.Ly, R=t.R)
    print('done in %.0f s; wrote %s/tile_result.json, tile_axis.csv, tile_state.npz' % (out['seconds'], g.out))

if __name__ == '__main__':
    main()
