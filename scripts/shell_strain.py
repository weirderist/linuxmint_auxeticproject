#!/usr/bin/env python3
"""shell_strain.py - strain and shape fields of a shell fold (shell.npz) for the 3D viewer. numpy only (no JAX, no solver).

The folded sheet is the vertex grid X (gy, gx, 3) of shell_sim.fold; its rest state is the flat grid (i W/sub, j C/sub). For every triangle (same triangulation as shell_sim.build, shell_metrics.tri_index)
the deformation gradient F (3 x 2) maps the rest edges to the folded edges; its singular values are the principal stretches s1 >= s2 of the MEMBRANE (the shell's stretch energy is the cost of s != 1;
the folds themselves are bending, which costs no stretch). Fields (per vertex = mean over the adjacent triangles):
  area_strain   s1 s2 - 1            stretch_max   s1 - 1            stretch_min   s2 - 1
  z             height of the sheet (mm)            tilt          angle between the surface normal and z (degrees)
Colours: strain fields diverge about 0 (tension red, compression blue, UNSOURCED display choice; range = +-(clip-th percentile of |value|)); z and tilt use one sequential ramp over the data range.
Legend dict = the contract of webui/viewer_core.js legendHTML (label, vmin, vmax, ticks [lo, mid, hi], stops, end_labels, note, data_min, data_max)."""
import numpy as np
import shell_metrics as sm

FIELDS = ('area_strain', 'stretch_max', 'stretch_min', 'z', 'tilt')
LABEL = dict(area_strain='membrane area strain (folded / flat area - 1)', stretch_max='largest principal stretch - 1 (membrane)', stretch_min='smallest principal stretch - 1 (membrane)',
             z='height z (mm)', tilt='surface tilt from horizontal (degrees)')
NEG = np.array([42, 120, 214], float); NEUTRAL = np.array([245, 245, 240], float); POS = np.array([214, 70, 50], float)          # UNSOURCED display colours: compression blue, tension red
SEQ = np.array([[250, 238, 168], [240, 170, 70], [200, 80, 90], [110, 50, 140], [30, 30, 90]], float)


def load(path):
    z = np.load(path); return np.asarray(z['X'], float), float(z['W']), float(z['C']), int(z['sub'])


def triangle_kinematics(X, W, C, sub):
    """(F, s1, s2, normals) per triangle: triangle vertex indices F (nt, 3), principal stretches, unit normals (nt, 3)."""
    gy, gx, _ = X.shape; V = X.reshape(-1, 3); T = sm.tri_index(gx - 1, gy - 1).astype(int)
    ii, jj = np.meshgrid(np.arange(gx), np.arange(gy)); P0 = np.stack([ii * W / sub, jj * C / sub], -1).reshape(-1, 2)
    p = V[T]; e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]; q = P0[T]; E1, E2 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    Dm = np.stack([E1, E2], -1); Ds = np.stack([e1, e2], -1); Fg = Ds @ np.linalg.inv(Dm); s = np.linalg.svd(Fg, compute_uv=False)
    n = np.cross(e1, e2); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return T, s[:, 0], s[:, 1], n


def to_vertices(T, tri_vals, nv):
    out = np.zeros(nv); cnt = np.zeros(nv)
    for k in range(3): np.add.at(out, T[:, k], tri_vals); np.add.at(cnt, T[:, k], 1.0)
    return out / np.maximum(cnt, 1.0)


def vertex_field(X, W, C, sub, name):
    if name not in FIELDS: raise ValueError('unknown field %r (one of %s)' % (name, ', '.join(FIELDS)))
    nv = X.shape[0] * X.shape[1]
    if name == 'z': return X[..., 2].reshape(-1).copy()
    T, s1, s2, n = triangle_kinematics(X, W, C, sub)
    if name == 'area_strain': return to_vertices(T, s1 * s2 - 1.0, nv)
    if name == 'stretch_max': return to_vertices(T, s1 - 1.0, nv)
    if name == 'stretch_min': return to_vertices(T, s2 - 1.0, nv)
    return to_vertices(T, np.degrees(np.arccos(np.clip(np.abs(n[:, 2]), 0, 1))), nv)          # tilt


def _ramp(stops, t):
    t = np.clip(t, 0, 1) * (len(stops) - 1); i = np.minimum(t.astype(int), len(stops) - 2); u = (t - i)[:, None]; return stops[i] * (1 - u) + stops[i + 1] * u


def colours(name, vals, clip=98.0):
    """(uint8 RGB (nv, 3), legend dict)."""
    dmin, dmax = float(vals.min()), float(vals.max())
    if name in ('area_strain', 'stretch_max', 'stretch_min'):
        c = float(np.percentile(np.abs(vals), clip)) or 1e-12; t = np.clip(vals / c, -1, 1)[:, None]
        rgb = np.where(t >= 0, NEUTRAL + (POS - NEUTRAL) * t, NEUTRAL + (NEG - NEUTRAL) * (-t))
        leg = dict(vmin=-c, vmax=c, ticks=[-c, 0.0, c], stops=[NEG.astype(int).tolist(), NEUTRAL.astype(int).tolist(), POS.astype(int).tolist()], end_labels=['compression', 'tension'], clip_percentile=clip,
                   note='range = +-%g-th percentile of |value| (UNSOURCED display choice); membrane strain of the flat-to-folded map, not stretch-test strain' % clip)
    else:
        lo, hi = dmin, dmax if dmax > dmin else dmin + 1e-12; rgb = _ramp(SEQ, (vals - lo) / (hi - lo))
        leg = dict(vmin=lo, vmax=hi, ticks=[lo, 0.5 * (lo + hi), hi], stops=SEQ.astype(int).tolist(), end_labels=['low', 'high'], note='range = data minimum to maximum')
    leg.update(label=LABEL[name], field=name, data_min=dmin, data_max=dmax)
    return np.asarray(rgb).round().astype(np.uint8), leg
