#!/usr/bin/env python3
"""build_knit.py - 8-unit-cell yarn-level weft-knit builder (replaces the nine-node builders).
Method: Wadekar, Perumal, Dion, Kontsos, Breen (2020), "An Optimized Yarn-Level Geometric Model for Finite Element Analysis of Weft-Knitted
Fabrics", Computer Aided Geometric Design.  8 unit cells (a leg spline and a head spline, uniform Catmull-Rom, nodes 0-6), initial control points
P = (a_x*W, a_y*C, a_z*R) from the paper's Table 1, optimised once with the paper's cost function (eq. 2-9, Table 5 weights), then tiled
(eq. 10) for any knit/purl pattern.  Defaults are the paper's: W = 16, C = 9, R = 1.  Cost is evaluated in units of R, as in the paper.

Usage: python3 build_knit.py --matrix "1 1 2 2 1/2 1 1 2 2" --tiles 4 --out run_knit [--name "my fabric"] [--id custom_id]
       (digits: 1 = front loop = purl frame, 2 = back loop = knit frame; rows separated by '/', first row = bottom course)
       Optional physical sizing instead of W, C, R:  --ne 10 --cpi 14 --wpi 18   (W = 25.4/WPI, C = 25.4/CPI, R = d/2, d = 25.4/(28 sqrt Ne) mm)
Output: fab_axis.csv (course,x,y,z), fab_nodes.csv (control points), fab_tube.obj (swept tube), meta.json (written last, atomically).
Tube: circular cross-section of radius R swept along each course axis with rotation-minimising frames (double reflection method,
      Wang, Juttler, Zheng, Liu 2008, "Computation of Rotation Minimizing Frames", ACM Trans. Graphics 27(1)).

Choices the paper leaves open (each checked by measuring the yarn-to-yarn distance of the assembled fabric):
  * Its Table 3 and Table 4 disagree on some copies; Table 4's copy sources are used (same sign).
  * Leg Node 5 and Head Node 1 are held at (0.25W, C, 0) and (0.25W, 0, 0).
  * Optimised nodes stay inside their half-stitch cell (0 <= x <= W/2, 0 <= y <= C); the paper gives no explicit bounds.
  * K_Join: sigma = local chord length; rho, tau from C1 continuity at theta = pi/4 (the paper defers these to Breen et al. 1994).
"""
import argparse, hashlib, json, math, os, sys, time
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import minimum_filter
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# ---- Table 1 of the paper (dimensionless coefficients a): x*W, y*C, z*R. None = fixed by neighbouring cells (copied)
T1 = {
 'A': {'LN': [(0, .20, 1.50), (.16, .25, 1.08), (.28, .41, 1.00), (.32, .72, 0.00), (.25, 1.00, 0.00)],
       'HN': [(.25, 0, 0), (.19, .33, -.63), (.23, .59, 1.00), (.34, .75, 2.08), (.50, .80, 2.33)]},
 'B': {'LN': [None, (.16, .25, 1.08), (.28, .41, 1.00), (.30, .72, -.33), (.25, 1.00, 0)],
       'HN': [(.25, 0, 0), (.19, .33, .83), (.23, .59, 1.50), (.34, .75, 2.00), (.50, .80, 2.08)]},
 'C': {'LN': [None, (.16, .25, .83), (.28, .41, 1.00), None, None],
       'HN': [None, None, (.23, .59, 1.00), (.34, .75, 2.08), (.50, .80, 2.33)]},
 'D': {'LN': [None, (.16, .25, .83), (.28, .41, 1.00), None, None],
       'HN': [None, None, (.23, .59, 1.50), (.34, .75, 2.08), (.50, .80, 2.33)]}}
GROUP = {1: 'A', 5: 'A', 2: 'B', 6: 'B', 3: 'C', 7: 'C', 4: 'D', 8: 'D'}
# ---- Table 4 (dependencies on previously optimised cells)
SRC_LN45 = {3: 1, 4: 2, 5: 1, 6: 2, 7: 1, 8: 2}
SRC_HN12 = {3: 2, 4: 1, 5: 1, 6: 2, 7: 2, 8: 1}
SRC_HN5 = {5: 1, 6: 2, 7: 3, 8: 4}
S_LEFT = {c: (1.0 if c <= 4 else -1.0) for c in range(1, 9)}                 # z-sign of Leg Node 0 (left neighbour P or K)
S_ABOVE = {1: 1., 2: -1., 3: 1., 4: -1., 5: 1., 6: -1., 7: 1., 8: -1.}        # z-sign of the stitch above (Table 2)
# paper constants (Table 5)
ALPHA, BETA, BETA0, GAMMA, GAMMA0, ETA = 1000.0, 1e-4, 1e6, 1.0, 0.75, 1000.0

def node_table(cid, Wn, Cn, solved):
    """Full set of 10 nodes of a cell (R-units, R = 1) from Table 1, overridden by copied nodes; plus the list of free variables."""
    g = T1[GROUP[cid]]
    N = {}
    for nm in ('LN', 'HN'):
        for i in range(5):
            a = g[nm][i]
            N[(nm, i+1)] = None if a is None else np.array([a[0]*Wn, a[1]*Cn, a[2]], float)
    N[('LN', 1)] = np.array([0.0, 0.20*Cn, 0.0]) if cid >= 5 else (N[('LN', 1)] if N[('LN', 1)] is not None else None)
    N[('LN', 5)] = np.array([0.25*Wn, Cn, 0.0]); N[('HN', 1)] = np.array([0.25*Wn, 0.0, 0.0])      # boundary nodes fixed (simplification, see notes)
    free = []
    if cid == 1: free.append(('LN', 1, [2]))
    if 2 <= cid <= 4:
        N[('LN', 1)] = np.array([0.0, 0.20*Cn, solved[1][('LN', 1)][2]])
    if cid in SRC_LN45:
        for i in (4, 5):
            N[('LN', i)] = solved[SRC_LN45[cid]][('LN', i)].copy()
        N[('LN', 5)] = np.array([0.25*Wn, Cn, 0.0])
    if cid in SRC_HN12:
        for i in (2,):
            N[('HN', i)] = solved[SRC_HN12[cid]][('HN', i)].copy()
    free += [('LN', 2, [0, 1, 2]), ('LN', 3, [0, 1, 2]), ('HN', 3, [0, 1, 2]), ('HN', 4, [0, 1, 2])]
    if cid in (1, 2): free += [('LN', 4, [0, 1, 2]), ('HN', 2, [0, 1, 2])]
    if cid in SRC_HN5: N[('HN', 5)] = np.array([0.5*Wn, 0.80*Cn, solved[SRC_HN5[cid]][('HN', 5)][2]])
    else: free.append(('HN', 5, [2]))
    return N, free

def assemble(cid, N, Cn):
    """Control points 0..6 of both splines, with the master-slave links (Table 3 and the text of section 3.1)."""
    L = np.zeros((7, 3)); H = np.zeros((7, 3))
    for i in range(1, 6): L[i] = N[('LN', i)]; H[i] = N[('HN', i)]
    L[0] = [2*L[1][0] - L[2][0], L[2][1], S_LEFT[cid]*L[2][2]]                    # LN0: reflection of LN2 about the x of LN1
    L[6] = [H[2][0], H[2][1] + Cn, S_ABOVE[cid]*H[2][2]]                          # LN6 follows HN2 (shifted one course up)
    H[0] = [L[4][0], L[4][1] - Cn, S_ABOVE[cid]*L[4][2]]                          # HN0 follows LN4 (shifted one course down)
    H[6] = [2*H[5][0] - H[4][0], H[4][1], H[4][2]]                                # HN6: reflection of HN4 about the x of HN5
    return L, H

def bezier_segments(P):
    """Uniform Catmull-Rom control points 0..6 -> 4 cubic Bezier segments between nodes 1..5."""
    segs = []
    for i in range(1, 5):
        P0, P1, P2, P3 = P[i-1], P[i], P[i+1], P[i+2]
        segs.append((P1, P1 + (P2 - P0)/6.0, P2 - (P3 - P1)/6.0, P2))
    return segs

TT = np.arange(100)/100.0
def sample(segs, n=100):
    t = (np.arange(n)/n)[:, None]
    out = []
    for b0, b1, b2, b3 in segs:
        out.append((1-t)**3*b0 + 3*(1-t)**2*t*b1 + 3*(1-t)*t**2*b2 + t**3*b3)
    return np.vstack(out)

def arclens(segs, n=100):
    t = (np.arange(n+1)/n)[:, None]; r = []
    for b0, b1, b2, b3 in segs:
        pts = (1-t)**3*b0 + 3*(1-t)**2*t*b1 + 3*(1-t)*t**2*b2 + t**3*b3
        r.append(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return np.array(r)

def kjoin(P):
    """Discrete joint curvature K_Join at nodes 1..5. sigma = local chord length; rho, tau fixed by C1 continuity at theta = pi/4 (the paper leaves them to Breen et al. 1994)."""
    ks = []
    for i in range(1, 6):
        u, v = P[i-1] - P[i], P[i+1] - P[i]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        th = math.acos(max(-1.0, min(1.0, float(u @ v)/(lu*lv + 1e-12)))); sg = 0.5*(lu + lv)
        if th > math.pi/4: ks.append((2.0/sg)*math.cos(th/2))
        else:
            rho = -(1.0/sg)*math.sin(math.pi/8); tau = (2.0/sg)*math.cos(math.pi/8)
            ks.append(-(math.pi/4)**2*rho/max(th, 1e-6) + tau + (math.pi/4)*rho)
    return np.array(ks)

def fbend(P):
    f = 0.0
    for b0, b1, b2, b3 in bezier_segments(P):
        A, B = b0 - 2*b1 + b2, b1 - 2*b2 + b3
        f += 12.0*(A @ A + A @ B + B @ B)                                         # exact integral of |C''|^2 over a cubic Bezier
    return f + BETA0*float((kjoin(P)**4).sum())

def contacts(Lp, Hp, sep=12):
    D = cdist(Lp, Hp); n = D.shape[0]
    mins = np.argwhere(D <= minimum_filter(D, size=3, mode='nearest') + 1e-12)
    ctr = np.array([(D.shape[0]-1)/2, (D.shape[1]-1)/2])
    order = np.argsort(np.linalg.norm(mins - ctr, axis=1)); mins = mins[order]
    c1 = mins[0]; c2 = c1
    for m in mins[1:]:
        if np.linalg.norm(m - c1) >= sep: c2 = m; break
    return D, tuple(c1), tuple(c2)

def cost(v, cid, N0, free, Cn, L0, alpha, parts=False):
    N = {k: x.copy() for k, x in N0.items()}; k = 0
    for nm, i, comps in free:
        for c in comps: N[(nm, i)][c] = v[k]; k += 1
    L, H = assemble(cid, N, Cn)
    sl, sh = bezier_segments(L), bezier_segments(H)
    Lp, Hp = sample(sl), sample(sh)
    D, c1, c2 = contacts(Lp, Hp)
    D1, D2, Dm = D[c1], D[c2], D.min()
    fd = (D1 - 2.0)**2 + (D2 - 2.0)**2 + (0.0 if min(abs(Dm - D1), abs(Dm - D2)) < 1e-9 else (Dm - 2.0)**2)
    fb = fbend(L) + fbend(H)
    ln = np.concatenate([arclens(sl), arclens(sh)])
    fl = float(((GAMMA0*L0 - ln)**2).sum())
    fc = 1.0/(1.0 + math.hypot(c1[0]-c2[0], c1[1]-c2[1]))
    total = alpha*fd + BETA*fb + GAMMA*fl + ETA*fc
    if parts: return total, dict(D1=D1, D2=D2, Dmin=Dm, fd=fd, fb=fb, fl=fl, fc=fc)
    return total

def initial_chords(cid, N0, free, Cn):
    L, H = assemble(cid, N0, Cn)
    return np.array([np.linalg.norm(s[3]-s[0]) for s in bezier_segments(L)] + [np.linalg.norm(s[3]-s[0]) for s in bezier_segments(H)])

def optimise_cell(cid, Wn, Cn, solved, tol=0.03, max_rounds=6, verbose=True):
    N0, free = node_table(cid, Wn, Cn, solved)
    for (nm, i), val in N0.items():
        if val is None: raise ValueError('cell %d node %s%d undefined' % (cid, nm, i))
    v = np.array([N0[(nm, i)][c] for nm, i, comps in free for c in comps])
    L0 = initial_chords(cid, N0, free, Cn); alpha = ALPHA; info = None
    for rnd in range(max_rounds):
        bnds = []
        for nm, i, comps in free:
            for c in comps: bnds.append((0.0, 0.5*Wn) if c == 0 else ((0.0, Cn) if c == 1 else (None, None)))   # keep every node inside its half-stitch cell
        r = minimize(cost, np.clip(v, [b[0] if b[0] is not None else -1e9 for b in bnds], [b[1] if b[1] is not None else 1e9 for b in bnds]),
                     args=(cid, N0, free, Cn, L0, alpha), method='L-BFGS-B', bounds=bnds, options=dict(maxiter=600, ftol=1e-12, gtol=1e-8, maxfun=60000))
        v = r.x; _, info = cost(v, cid, N0, free, Cn, L0, alpha, parts=True)
        ok = abs(info['D1']-2) <= tol*2 and abs(info['D2']-2) <= tol*2 and info['Dmin'] >= 2*(1-tol)
        if verbose: print('   cell %d round %d alpha %.0e: D1 %.3f D2 %.3f Dmin %.3f  %s' % (cid, rnd+1, alpha, info['D1'], info['D2'], info['Dmin'], 'OK' if ok else 'repeat with larger alpha'), flush=True)
        if ok: break
        alpha *= 10.0
    k = 0; N = {key: x.copy() for key, x in N0.items()}
    for nm, i, comps in free:
        for c in comps: N[(nm, i)][c] = v[k]; k += 1
    return N, dict(info, ok=bool(ok), alpha=alpha)

def optimise_all(Wn, Cn, cache_dir=None, verbose=True):
    key = 'cells_W%.4f_C%.4f.json' % (Wn, Cn)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True); path = os.path.join(cache_dir, key)
        if os.path.exists(path):
            raw = json.load(open(path))
            return ({int(c): {(k.split(':')[0], int(k.split(':')[1])): np.array(x) for k, x in d.items()} for c, d in raw['cells'].items()},
                    {int(c): r for c, r in raw['report'].items()})
    solved, rep = {}, {}
    for cid in range(1, 9):
        solved[cid], rep[cid] = optimise_cell(cid, Wn, Cn, solved, verbose=verbose)
    if cache_dir:
        cells_json = {str(c): {'%s:%d' % k: x.tolist() for k, x in d.items()} for c, d in solved.items()}
        rep_json = {str(c): {k: (v if isinstance(v, bool) else float(v)) for k, v in r.items()} for c, r in rep.items()}
        json.dump(dict(cells=cells_json, report=rep_json), open(path, 'w'))
    return solved, rep


CODE = {('P', 'P'): 0, ('K', 'K'): 1, ('P', 'K'): 2, ('K', 'P'): 3}      # (stitch above, stitch below) of Table 2 -> cell offset 0..3

def cell_id(T, r, t, side, swap_labels=False):
    """Unit cell for the left (side=-1) or right (+1) half of stitch (r,t). Missing neighbours copy the boundary stitch's own type."""
    S, Wd = len(T), len(T[0]); me = T[r][t]
    def nb(rr, tt):
        if rr < 0 or rr >= S or tt < 0 or tt >= Wd: return me
        return T[rr][tt]
    rel = lambda x: 'P' if x == me else 'K'                                # neighbour type seen in the purl frame of this stitch
    above, below, sd = rel(nb(r+1, t)), rel(nb(r-1, t)), rel(nb(r, t+side))
    if swap_labels: above, below = below, above
    return 1 + (4 if sd == 'K' else 0) + CODE[(above, below)]

def build_rows(T, cells, W, C, R, swap_labels=False, close_top=True):
    """Return one control-point polyline per course (mm), plus the cell ids used. Yarn row r = legs of stitches in row r + heads (crowns) held in row r+1."""
    S, Wd = len(T), len(T[0]); pz = lambda x: 1.0 if x == 1 else -1.0
    rows, used = [], set()
    Tx = [list(r) for r in T]
    if close_top: Tx.append(list(T[-1]))                                      # virtual row above the last course: closes the top loops
    for r in range(S):
        pts = []
        for t in range(Wd):
            for half in (-1, +1):
                pass
            cl = cell_id(Tx, r, t, -1, swap_labels); cr = cell_id(Tx, r, t, +1, swap_labels); used |= {cl, cr}
            hl = cell_id(Tx, r+1, t, -1, swap_labels) if (r+1 < len(Tx)) else None
            hr = cell_id(Tx, r+1, t, +1, swap_labels) if (r+1 < len(Tx)) else None
            if hl: used |= {hl, hr}
            p0 = pz(T[r][t]); p1 = pz(Tx[r+1][t]) if r+1 < len(Tx) else p0
            def G(nm, i, cid, half, p, row):
                x, y, z = cells[cid][(nm, i)]
                X = t*W + x*R if half < 0 else (t+1)*W - x*R
                return np.array([X, row*C + y*R, p*z*R])
            seq = [G('LN', i, cl, -1, p0, r) for i in range(1, 6)]
            if hl: seq += [G('HN', i, hl, -1, p1, r+1) for i in range(2, 6)] + [G('HN', i, hr, +1, p1, r+1) for i in (4, 3, 2)]
            seq += [G('LN', i, cr, +1, p0, r) for i in (5, 4, 3, 2, 1)]
            for q in seq:
                if pts and np.linalg.norm(q - pts[-1]) < 1e-9: continue
                pts.append(q)
        rows.append(np.array(pts))
    return rows, used

def catmull_rom(P, per=16):
    """Uniform Catmull-Rom through P; end tangents from reflected end points."""
    P = np.vstack([2*P[0]-P[1], P, 2*P[-1]-P[-2]]); out = []
    t = (np.arange(per)/per)[:, None]
    for i in range(1, len(P)-2):
        p0, p1, p2, p3 = P[i-1], P[i], P[i+1], P[i+2]
        out.append(0.5*((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t**2 + (-p0+3*p1-3*p2+p3)*t**3))
    out.append(P[-2][None]); return np.vstack(out)

def verify(curves, R, margin_x, arc_skip=3.0, border_rows=1):
    """Minimum distance between yarn axes (not on the same yarn within arc_skip*2R), interior of the fabric. Returns overall min and min by row gap."""
    pts = np.vstack(curves); rid = np.concatenate([[k]*len(c) for k, c in enumerate(curves)])
    arc = np.concatenate([np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))] for c in curves])
    xmin, xmax = pts[:, 0].min()+margin_x, pts[:, 0].max()-margin_x
    cen = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (rid >= border_rows) & (rid < len(curves)-border_rows)
    tree = cKDTree(pts); best = 1e9; byg = {}
    for i in np.where(cen)[0][::2]:
        dd, jj = tree.query(pts[i], k=40)
        for dist, j in zip(dd, jj):
            if j == i or j >= len(pts): continue
            if rid[j] == rid[i] and abs(arc[j]-arc[i]) < arc_skip*2*R: continue
            g = int(abs(rid[j]-rid[i])); byg[g] = min(byg.get(g, 1e9), dist); best = min(best, dist); break
    return best, byg


# ---------------------------------------------------------------- tube sweep
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v/np.where(n < 1e-15, 1.0, n)

def rmf_frames(P):
    """Rotation-minimising frames (r, s) along polyline P by the double reflection method (Wang et al. 2008). Tangent t, with (r, s, t) right-handed."""
    n = len(P)
    T = _unit(np.gradient(P, axis=0))
    a = np.zeros(3); a[int(np.argmin(np.abs(T[0])))] = 1.0
    r = _unit((a - (a @ T[0])*T[0])[None])[0]
    Rr = np.empty((n, 3)); Rr[0] = r
    for i in range(n-1):
        v1 = P[i+1] - P[i]; c1 = v1 @ v1
        if c1 < 1e-24: Rr[i+1] = r; continue
        rL = r - (2.0/c1)*(v1 @ r)*v1
        tL = T[i] - (2.0/c1)*(v1 @ T[i])*v1
        v2 = T[i+1] - tL; c2 = v2 @ v2
        r = rL - (2.0/c2)*(v2 @ rL)*v2 if c2 > 1e-24 else rL
        r = r - (r @ T[i+1])*T[i+1]; r = r/ (np.linalg.norm(r) + 1e-15)          # remove numerical drift
        Rr[i+1] = r
    S = np.cross(T, Rr)
    return T, Rr, S

def sweep_tube(curve, R, sides):
    """Circular tube of radius R about curve. Returns vertices, normals, triangles (0-based) with end caps."""
    keep = np.r_[True, np.linalg.norm(np.diff(curve, axis=0), axis=1) > 1e-9]
    P = curve[keep]; n = len(P)
    T, Rr, S = rmf_frames(P)
    phi = 2*np.pi*np.arange(sides)/sides
    ring = np.cos(phi)[None, :, None]*Rr[:, None, :] + np.sin(phi)[None, :, None]*S[:, None, :]     # n x sides x 3 (unit normals)
    V = (P[:, None, :] + R*ring).reshape(-1, 3); N = ring.reshape(-1, 3)
    i = np.arange(n-1)[:, None]; j = np.arange(sides)[None, :]
    a = (i*sides + j).ravel(); b = (i*sides + (j+1) % sides).ravel(); c = a + sides; d = b + sides
    F = np.vstack([np.c_[a, b, d], np.c_[a, d, c]])
    c0 = len(V); c1 = c0 + 1
    V = np.vstack([V, P[0], P[-1]]); N = np.vstack([N, -T[0], T[-1]])
    js = np.arange(sides); jn = (js+1) % sides; last = (n-1)*sides
    F = np.vstack([F, np.c_[jn, js, np.full(sides, c0)], np.c_[last+js, last+jn, np.full(sides, c1)]])
    return V, N, F

def write_obj(path, curves_tube, R, sides, header):
    """curves_tube: list of axis polylines. Writes one OBJ (one object per course)."""
    Vs, Ns, Fs, off = [], [], [], 0
    for k, cv in enumerate(curves_tube):
        V, N, F = sweep_tube(cv, R, sides)
        Vs.append(V); Ns.append(N); Fs.append(F + off); off += len(V)
    V = np.vstack(Vs); N = np.vstack(Ns); F = np.vstack(Fs) + 1
    with open(path, 'w') as f:
        f.write('# %s\n' % header)
        np.savetxt(f, V, fmt='v %.5f %.5f %.5f')
        np.savetxt(f, N, fmt='vn %.4f %.4f %.4f')
        np.savetxt(f, F, fmt='f %d %d %d')                                  # plain indices: the web viewer's parseOBJ reads 'f a b c' only
    return len(V), len(F)


def tile_matrix(M, n):
    return [[M[r % len(M)][c % len(M[0])] for c in range(n*len(M[0]))] for r in range(n*len(M))]

def fnum(x, nd=4): return float('%.*g' % (nd, x))

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--matrix', required=True, help='rows separated by /, digits 1 (front) and 2 (back); first row = bottom course')
    a.add_argument('--W', type=float, default=16.0); a.add_argument('--C', type=float, default=9.0); a.add_argument('--R', type=float, default=1.0)
    a.add_argument('--ne', type=float); a.add_argument('--cpi', type=float); a.add_argument('--wpi', type=float)
    a.add_argument('--tiles', type=int, default=25, help='repeats of the matrix in x and y')
    a.add_argument('--out', default='run_knit'); a.add_argument('--cache', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cell_cache'))
    a.add_argument('--labels', choices=('swapped', 'printed'), default='swapped'); a.add_argument('--swap-types', action='store_true')
    a.add_argument('--per', type=int, default=16, help='curve samples per control-point interval (axis CSV)')
    a.add_argument('--name', default=None, help='human-readable label stored in meta.json (default: matrix and tiles)')
    a.add_argument('--id', default=None, help='result id stored in meta.json (default: timestamp + short hash); independent of --name and of the folder name')
    a.add_argument('--sides', type=int, default=8, help='tube cross-section polygon sides')
    a.add_argument('--tube-per', type=int, default=6, help='tube samples per control-point interval')
    a.add_argument('--tube-max-verts', type=int, default=400000, help='tube is coarsened, then skipped, above this vertex count')
    a.add_argument('--no-tube', action='store_true')
    g = a.parse_args()
    M0 = [[int(ch) for ch in row.replace(' ', '')] for row in g.matrix.split('/')]
    assert len({len(r) for r in M0}) == 1 and all(v in (1, 2) for r in M0 for v in r), 'matrix must be rectangular with digits 1 and 2'
    if g.swap_types: M0 = [[3 - v for v in r] for r in M0]
    W, C, R = g.W, g.C, g.R
    physical = False
    if (g.ne or g.cpi or g.wpi) and not (g.ne and g.cpi and g.wpi): sys.exit('error: give all of --ne --cpi --wpi, or none')
    if g.ne and g.cpi and g.wpi:
        R = 25.4/(28.0*math.sqrt(g.ne))/2; W = 25.4/g.wpi; C = 25.4/g.cpi; physical = True
    print('W = %.4f  C = %.4f  R = %.4f   (W/R = %.2f, C/R = %.2f)' % (W, C, R, W/R, C/R))
    t0 = time.time(); solved, rep = optimise_all(W/R, C/R, cache_dir=g.cache, verbose=True)
    opt_time = time.time()-t0
    print('unit cells ready (%.0f s). Contact distances in units of 2R:' % opt_time)
    for c in range(1, 9): print('   cell %d: D1 %.3f  D2 %.3f  Dmin %.3f  %s' % (c, rep[c]['D1']/2, rep[c]['D2']/2, rep[c]['Dmin']/2, 'converged' if rep[c]['ok'] else 'NOT fully converged'))
    sw = g.labels == 'swapped'
    T = tile_matrix(M0, g.tiles)
    rows, used = build_rows(T, solved, W, C, R, swap_labels=sw)
    curves = [catmull_rom(p, g.per) for p in rows]
    Ts = tile_matrix(M0, 3); rs, us = build_rows(Ts, solved, W, C, R, swap_labels=sw)       # 3 x 3 repeat sample for verification
    vmin, vby = verify([catmull_rom(p, 16) for p in rs], R, margin_x=2*W, border_rows=len(M0))
    os.makedirs(g.out, exist_ok=True)
    np.savetxt(os.path.join(g.out, 'fab_axis.csv'), np.vstack([np.c_[np.full(len(c), k), c] for k, c in enumerate(curves)]), delimiter=',', header='course,x,y,z', comments='', fmt=['%d', '%.6f', '%.6f', '%.6f'])
    np.savetxt(os.path.join(g.out, 'fab_nodes.csv'), np.vstack([np.c_[np.full(len(p), k), np.arange(len(p)), p] for k, p in enumerate(rows)]), delimiter=',', header='course,idx,x,y,z', comments='', fmt=['%d', '%d', '%.6f', '%.6f', '%.6f'])

    # ---- tube sweep (coarsened or skipped if too large for the viewer)
    tube = None
    if not g.no_tube:
        per, sides = max(2, g.tube_per), max(3, g.sides)
        est = lambda p, s: sum((len(r)-1)*p + 1 for r in rows)*s + 2*len(rows)
        while est(per, sides) > g.tube_max_verts and per > 2: per -= 1
        while est(per, sides) > g.tube_max_verts and sides > 4: sides -= 1
        if est(per, sides) > g.tube_max_verts:
            print('WARNING: tube would have about %d vertices (limit %d); fab_tube.obj not written. Use fewer --tiles.' % (est(per, sides), g.tube_max_verts))
        else:
            if (per, sides) != (g.tube_per, g.sides): print('note: tube coarsened to %d samples per interval, %d sides to stay under %d vertices' % (per, sides, g.tube_max_verts))
            tcurves = [catmull_rom(p, per) for p in rows]
            nv, nf = write_obj(os.path.join(g.out, 'fab_tube.obj'), tcurves, R, sides, 'build_knit.py tube, radius %.6f, %d sides, %d courses' % (R, sides, len(tcurves)))
            tube = dict(file='fab_tube.obj', radius=fnum(R, 6), sides=sides, samples_per_interval=per, vertices=nv, triangles=nf, frames='rotation-minimising (double reflection, Wang et al. 2008)')
            print('wrote %s/fab_tube.obj (%d vertices, %d triangles)' % (g.out, nv, nf))

    bad = sorted(c for c in used if not rep[c]['ok'])
    status = 'PASS' if vmin >= 0.97*2*R else ('WARN (slight interpenetration)' if vmin >= 0.80*2*R else 'FAIL (yarns collide)')
    now = datetime.now()
    name = g.name or ('%s x%d' % ('/'.join(''.join(map(str, r)) for r in M0), g.tiles))
    rid = g.id or (now.strftime('%Y%m%d-%H%M%S') + '-' + hashlib.sha1(json.dumps([M0, g.tiles, W, C, R, g.labels, now.isoformat()]).encode()).hexdigest()[:4])
    units = 'mm' if physical else 'R-units (R = %g, dimensionless)' % R
    structure = dict(units=units, W=fnum(W, 6), C=fnum(C, 6), R=fnum(R, 6), W_over_R=fnum(W/R), C_over_R=fnum(C/R), yarn_diameter=fnum(2*R, 6),
                     contact_distance_2R=fnum(2*R, 6), ne=g.ne, cpi=g.cpi, wpi=g.wpi, labels=g.labels, swap_types=bool(g.swap_types),
                     matrix=M0, tiles=g.tiles, courses=len(rows), stitches_per_course=len(T[0]))
    unit_cells = dict(method='Wadekar et al. 2020, 8 unit cells; uniform Catmull-Rom, control points 0-6, nodes 1-5 interpolated',
                      optimiser='L-BFGS-B, node bounds 0<=x<=W/2, 0<=y<=C, contact tolerance 3%, alpha x10 per repeat',
                      weights=dict(alpha=ALPHA, beta=BETA, beta0=BETA0, gamma=GAMMA, gamma0=GAMMA0, eta=ETA), optimisation_seconds=fnum(opt_time, 3),
                      cells={str(c): dict(used=(c in used), converged=bool(rep[c]['ok']), D1_over_2R=fnum(rep[c]['D1']/2), D2_over_2R=fnum(rep[c]['D2']/2), Dmin_over_2R=fnum(rep[c]['Dmin']/2),
                                          final_alpha=fnum(rep[c]['alpha'], 3), nodes_in_R=({'%s%d' % k: [fnum(float(v), 5) for v in x] for k, x in solved[c].items()} if True else None))
                             for c in range(1, 9)})
    info = ['Name: %s' % name, 'Id: %s' % rid, 'Created: %s' % now.isoformat(timespec='seconds'),
            'Fabric: %d courses x %d stitches (matrix %dx%d, %d tiles)' % (len(rows), len(T[0]), len(M0), len(M0[0]), g.tiles),
            'Units: %s' % units,
            'W (wale spacing) = %.4f   C (course spacing) = %.4f   R (yarn radius) = %.4f' % (W, C, R),
            'W/R = %.2f   C/R = %.2f   yarn diameter 2R = %.4f' % (W/R, C/R, 2*R)]
    if physical: info.append('Ne %g   CPI %g   WPI %g' % (g.ne, g.cpi, g.wpi))
    info += ['Table 2 labels: %s   type swap: %s' % (g.labels, 'yes' if g.swap_types else 'no'),
             'Unit cells: 8, Catmull-Rom (nodes 0-6), L-BFGS-B; weights alpha %g beta %g beta0 %g gamma %g gamma0 %g eta %g' % (ALPHA, BETA, BETA0, GAMMA, GAMMA0, ETA)]
    for c in range(1, 9):
        info.append('  cell %d%s: D1 %.3f  D2 %.3f  Dmin %.3f (x2R)  %s' % (c, '' if c in used else ' (unused)', rep[c]['D1']/2, rep[c]['D2']/2, rep[c]['Dmin']/2, 'converged' if rep[c]['ok'] else 'NOT fully converged'))
    info.append('Verification (3x3 repeat): min axis distance %.1f%% of 2R -> %s' % (100*vmin/(2*R), status))
    if tube: info.append('Tube: radius %.4f, %d sides, %d vertices' % (R, tube['sides'], tube['vertices']))
    meta = dict(schema=2, id=rid, name=name, created=now.isoformat(timespec='seconds'),
                matrix=M0, tiles=g.tiles, courses=len(rows), stitches_per_course=len(T[0]), W=W, C=C, R=R, labels=g.labels, cells_used=sorted(used),
                cells_not_converged_used=bad, min_axis_distance=float(vmin), min_axis_distance_over_2R=float(vmin/(2*R)),
                structure=structure, unit_cells=unit_cells, verification=dict(status=status.split()[0], min_axis_distance=float(vmin), over_2R=float(vmin/(2*R)), sample='3x3 repeat'),
                tube=tube, files=['fab_axis.csv', 'fab_nodes.csv'] + (['fab_tube.obj'] if tube else []), info=info)
    if physical: meta.update(ne=g.ne, cpi=g.cpi, wpi=g.wpi, d=2*R)
    tmp = os.path.join(g.out, 'meta.json.tmp')
    json.dump(meta, open(tmp, 'w'), indent=1); os.replace(tmp, os.path.join(g.out, 'meta.json'))     # written last, atomically
    print('fabric: %d courses x %d stitches, cells used %s%s' % (len(rows), len(T[0]), sorted(used), ('; not fully converged: %s' % bad) if bad else ''))
    print('RESULT: minimum yarn-axis distance %.4f = %.1f%% of 2R (contact distance) -> %s' % (vmin, 100*vmin/(2*R), status))
    print('wrote %s/fab_axis.csv (%d points), fab_nodes.csv, meta.json  [id %s, name "%s"]' % (g.out, sum(len(c) for c in curves), rid, name))
    print('--- info ---'); print('\n'.join(info))

if __name__ == '__main__':
    main()