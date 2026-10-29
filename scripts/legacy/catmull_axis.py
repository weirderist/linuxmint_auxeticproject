#!/usr/bin/env python3
"""catmull_axis.py - yarn axis (centripetal Catmull-Rom) + tube sweep for one fabric patch. Units: mm.
Assumptions (not cited formulas): hook radius rh = needle-arc radius rn = w/4; calf length Lc is solved so the
analytic loop length equals --l. --zb is a PLACEHOLDER out-of-plane bulge (sin profile, zero at loop ends);
at zb=0 the front/back mirror is a no-op. With zb>0 the bulge adds length and Lc is NOT re-solved.
Peirce R is not used. Interlocking between courses is the connector step, not done here."""
import argparse, math, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cpi", type=float, default=14.0)
ap.add_argument("--wpi", type=float, default=18.0)
ap.add_argument("--l", type=float, default=3.2)
ap.add_argument("--d", type=float, default=0.2869)
ap.add_argument("--zb", type=float, default=0.0)
ap.add_argument("--matrix", default="FFFFBBBB,FFFFBBBB,BBBBFFFF,BBBBFFFF")
ap.add_argument("--ds", type=float, default=0.05)    # dense-node spacing
ap.add_argument("--tol", type=float, default=0.01)   # loop-length tolerance (fraction)
ap.add_argument("--nodes9", action="store_true")     # use only the 9 nodes as through-points
ap.add_argument("--out", default="axis")
a = ap.parse_args()

w, c, d, M = 25.4 / a.wpi, 25.4 / a.cpi, a.d, 16
rh = rn = w / 4
Lc = (a.l - math.pi * (rh + rn)) / 2
if Lc <= 0: sys.exit("l too short for this w: calf length would be <= 0")
H = rh + Lc + rn

def arc(cx, cy, r, t0, t1):
    n = max(2, math.ceil(abs(t1 - t0) * r / a.ds)); t = np.linspace(t0, t1, n + 1)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
def line(p, q):
    n = max(2, math.ceil(math.dist(p, q) / a.ds)); t = np.linspace(0, 1, n + 1)[:, None]
    return np.asarray(p) * (1 - t) + np.asarray(q) * t
segs = [arc(0, rh, rh, -math.pi/2, 0), line((rh, rh), (rh, rh + Lc)),
        arc(rh + rn, rh + Lc, rn, math.pi, 0), line((rh + 2*rn, rh + Lc), (rh + 2*rn, rh)),
        arc(2*rh + 2*rn, rh, rh, math.pi, 1.5*math.pi)]
loop2d = np.vstack([segs[0]] + [s[1:] for s in segs[1:]])
analytic = math.pi * (rh + rn) + 2 * Lc
if a.nodes9:
    q = math.pi / 4
    loop2d = np.array([(0, 0), (rh*math.cos(q), rh - rh*math.sin(q)), (rh, rh), (rh, rh + Lc),
        (rh + rn, H), (rh + 2*rn, rh + Lc), (rh + 2*rn, rh),
        (2*rh + 2*rn - rh*math.cos(q), rh - rh*math.sin(q)), (w, 0)])

ch = np.r_[0, np.cumsum(np.linalg.norm(np.diff(loop2d, axis=0), axis=1))]
zprof = a.zb * np.sin(math.pi * ch / ch[-1])

def build_course(types, y0):
    arr, idx = None, []
    for k, t in enumerate(types):
        P = np.c_[loop2d[:, 0] + k * w, loop2d[:, 1] + y0, (1 if t == "F" else -1) * zprof]
        if arr is None: i0, arr = 0, P
        else:
            i0 = len(arr) - 1; arr[-1] = 0.5 * (arr[-1] + P[0]); arr = np.vstack([arr, P[1:]])
        idx.append((i0, len(arr) - 1))
    return arr, idx

def catmull_rom(P, m=M, alpha=0.5):
    P = np.vstack([2*P[0] - P[1], P, 2*P[-1] - P[-2]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i-1], P[i], P[i+1], P[i+2]
        t0 = 0.0; t1 = t0 + np.linalg.norm(p1 - p0)**alpha
        t2 = t1 + np.linalg.norm(p2 - p1)**alpha; t3 = t2 + np.linalg.norm(p3 - p2)**alpha
        t = np.linspace(t1, t2, m, endpoint=False)[:, None]
        A1 = (t1-t)/(t1-t0)*p0 + (t-t0)/(t1-t0)*p1
        A2 = (t2-t)/(t2-t1)*p1 + (t-t1)/(t2-t1)*p2
        A3 = (t3-t)/(t3-t2)*p2 + (t-t2)/(t3-t2)*p3
        B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
        B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3
        out.append((t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2)
    out.append(P[-2][None]); return np.vstack(out)

def min_radius(C):
    A, B, D = C[:-2], C[1:-1], C[2:]
    ab, bd, ad = (np.linalg.norm(B-A, axis=1), np.linalg.norm(D-B, axis=1), np.linalg.norm(D-A, axis=1))
    area2 = np.linalg.norm(np.cross(B-A, D-A), axis=1)      # = 2*area
    return np.min(ab*bd*ad / np.maximum(2*area2, 1e-12))

def tube(C, r, nr=12):
    T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
    n = np.cross(T[0], [0, 0, 1.0]); n /= np.linalg.norm(n); N = [n]
    for i in range(1, len(C)):
        v = np.cross(T[i-1], T[i]); s = np.linalg.norm(v)
        if s > 1e-12:
            ax = v / s; ang = math.atan2(s, float(T[i-1] @ T[i]))
            n = n*math.cos(ang) + np.cross(ax, n)*math.sin(ang) + ax*(ax @ n)*(1 - math.cos(ang))
        n = n - T[i] * (n @ T[i]); n /= np.linalg.norm(n); N.append(n)
    N = np.array(N); B = np.cross(T, N); ph = np.linspace(0, 2*math.pi, nr, endpoint=False)
    V = (C[:, None, :] + r * (np.cos(ph)[None, :, None] * N[:, None, :] + np.sin(ph)[None, :, None] * B[:, None, :]))
    i = np.arange(len(C) - 1)[:, None]; j = np.arange(nr)[None, :]
    a0, a1 = i*nr + j, i*nr + (j + 1) % nr
    F = np.concatenate([np.stack([a0, a1, a0 + nr], -1).reshape(-1, 3), np.stack([a1, a1 + nr, a0 + nr], -1).reshape(-1, 3)])
    return V.reshape(-1, 3), F

rows, verts, faces, Ls, rmins, off = [], [], [], [], [], 0
for j, types in enumerate(a.matrix.split(",")):
    nodes, idx = build_course(types, j * c)
    C = catmull_rom(nodes)
    for (i0, i1) in idx: Ls.append(np.linalg.norm(np.diff(C[i0*M:i1*M + 1], axis=0), axis=1).sum())
    rmins.append(min_radius(C))
    rows += [(j, *p) for p in C]
    V, F = tube(C[::8], d / 2, 10); verts.append(V); faces.append(F + off); off += len(V)
np.savetxt(a.out + "_axis.csv", np.array(rows), delimiter=",", header="course,x,y,z", comments="", fmt="%.6f")
V, F = np.vstack(verts), np.vstack(faces)
with open(a.out + "_tube.obj", "w") as f:
    f.write("\n".join("v %.5f %.5f %.5f" % tuple(v) for v in V) + "\n")
    f.write("\n".join("f %d %d %d" % tuple(t + 1) for t in F) + "\n")

Ls = np.array(Ls); err = (Ls - analytic) / analytic
ok = bool(np.all(np.abs(err) <= a.tol))
print(f"mode={'9 nodes' if a.nodes9 else 'dense'}  w={w:.4f} c={c:.4f}  rh=rn={rh:.4f}  Lc={Lc:.4f}")
print(f"analytic loop length = {analytic:.4f} (target l={a.l})")
print(f"spline loop length   = min {Ls.min():.4f} / mean {Ls.mean():.4f} / max {Ls.max():.4f}  (max |err| {np.abs(err).max()*100:.3f}%)")
print(f"loop length check (tol {a.tol*100:.1f}%): {'PASS' if ok else 'FAIL -> switch spline type'}")
print(f"info: R_min = {min(rmins):.4f} (tube radius {d/2:.4f});  loop height H = {H:.4f} vs course spacing c = {c:.4f}")
print(f"wrote {a.out}_axis.csv, {a.out}_tube.obj  ({len(V)} verts)")
