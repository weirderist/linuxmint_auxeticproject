#!/usr/bin/env python3
"""shell_metrics.py - descriptors and the mesh export of one shell fold (shell.npz). numpy only: no JAX, no solver, runs anywhere.

describe(X, W, C, sub, fold_min_turn_rad): X is the folded vertex grid (gy, gx, 3) of shell_sim.fold (row j = course position y, column i = wale position x; flat position (i W/sub, j C/sub)).
  plan_x, plan_y, thickness : extents of the folded sheet (mm); flat_x, flat_y: extents before folding; shrink_x, shrink_y = plan / flat - 1.
  fold_kx_per_mm, fold_ky_per_mm: curvatures of the global quadratic fit z = a x^2 + b y^2 + c xy + d x + e y + f (same fit as fold_metric.fit); fold_turn_*_rad = |k| x extent; fold_direction as in campaign.py.
  wave_*: sideways snaking of the wale columns. For every column of the central half of the width: x minus flat x, linear trend removed, FFT along the length.
          wave_k = dominant wave number (periods over the sheet length), wave_power_fraction = its share of the spectral power, wave_amp_median_mm = half the peak-to-peak
          amplitude (median over those columns), mid_over_end = largest sideways deviation in the middle third of the length over that of the two outer thirds.
  These definitions are this project's own; the Blender reference was measured separately and may differ (UNSOURCED agreement).
kmsh_bytes(X): the shell as the KMSH v1 mesh of build_knit.write_obj ('KMSH', version 1, nv, nf as uint32 LE, float32 xyz, uint32 triangle indices, 0-based), triangles with the alternating
  diagonals of shell_sim.build, so the web viewer's loadTubeBase reads it unchanged.
"""
import numpy as np


def tri_index(nx_cells, ny_cells):
    """Triangles (nt, 3) of the (ny_cells x nx_cells) grid, in the order and with the diagonals of shell_sim.build."""
    gx = nx_cells + 1
    j, i = np.meshgrid(np.arange(ny_cells), np.arange(nx_cells), indexing='ij'); j = j.ravel(); i = i.ravel()
    a = j * gx + i; b = a + 1; d = a + gx; c = d + 1; even = ((i + j) % 2 == 0)[:, None]
    t1 = np.where(even, np.c_[a, b, c], np.c_[a, b, d]); t2 = np.where(even, np.c_[a, c, d], np.c_[b, c, d])
    out = np.empty((2 * len(a), 3), np.uint32); out[0::2] = t1; out[1::2] = t2
    return out


def kmsh_bytes(X):
    gy, gx, _ = X.shape; V = np.asarray(X, '<f4').reshape(-1, 3); F = tri_index(gx - 1, gy - 1)
    return b'KMSH' + np.array([1, len(V), len(F)], dtype='<u4').tobytes() + V.tobytes() + F.astype('<u4').tobytes()


def _f(v):
    v = float(v); return v if np.isfinite(v) else None


def describe(X, W, C, sub, fold_min_turn_rad=0.2):
    X = np.asarray(X, float); gy, gx, _ = X.shape
    if not np.all(np.isfinite(X)): raise ValueError('non-finite vertex positions')
    flat_x = (np.arange(gx) * W / sub)[None, :]; dx = X[..., 0] - flat_x
    out = dict(grid_x=gx, grid_y=gy, plan_x=_f(np.ptp(X[..., 0])), plan_y=_f(np.ptp(X[..., 1])), thickness=_f(np.ptp(X[..., 2])), flat_x=_f((gx - 1) * W / sub), flat_y=_f((gy - 1) * C / sub))
    out['shrink_x'] = _f(out['plan_x'] / out['flat_x'] - 1); out['shrink_y'] = _f(out['plan_y'] / out['flat_y'] - 1)
    x = X[..., 0].ravel() - X[..., 0].mean(); y = X[..., 1].ravel() - X[..., 1].mean(); z = X[..., 2].ravel()
    c = np.linalg.lstsq(np.c_[x * x, y * y, x * y, x, y, np.ones_like(x)], z, rcond=None)[0]
    kx, ky = 2 * c[0], 2 * c[1]; tx, ty = abs(kx) * out['plan_x'], abs(ky) * out['plan_y']
    out.update(fold_kx_per_mm=_f(kx), fold_ky_per_mm=_f(ky), fold_turn_x_rad=_f(tx), fold_turn_y_rad=_f(ty),
               fold_direction='both' if tx >= fold_min_turn_rad and ty >= fold_min_turn_rad else 'course' if tx >= fold_min_turn_rad else 'wale' if ty >= fold_min_turn_rad else 'none')
    if gy >= 8 and gx >= 4:
        r = np.arange(gy); cols = np.arange(gx // 4, max(gx // 4 + 1, 3 * gx // 4)); ks = []; fr = []; am = []
        for i in cols:
            col = dx[:, i]; res = col - np.polyval(np.polyfit(r, col, 1), r); sp = np.abs(np.fft.rfft(res))[1:gy // 2]
            if len(sp) == 0 or (sp ** 2).sum() <= 1e-30: continue
            k = int(np.argmax(sp)) + 1; ks.append(k); fr.append(sp[k - 1] ** 2 / (sp ** 2).sum()); am.append((res.max() - res.min()) / 2)
        if ks:
            t = gy // 3; dev = lambda a: np.abs(a - a.mean(0)).max()
            out.update(wave_k=_f(np.median(ks)), wave_power_fraction=_f(np.median(fr)), wave_amp_median_mm=_f(np.median(am)), mid_over_end=_f(dev(dx[t:2 * t]) / max(dev(dx[:t]), dev(dx[2 * t:]), 1e-12)))
    return out
