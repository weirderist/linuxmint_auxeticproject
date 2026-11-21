#!/usr/bin/env python3
"""Tests of scripts/shell_strain.py (membrane strain and shape fields of a shell, numpy only) with exact cases: flat sheet, uniform stretch, a sheet rolled onto a cylinder (isometric), legend contract.
Run from the repository root: python tests/test_shell_strain.py"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import shell_strain as ss
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)
W, C, sub = 1.5875, 1.1545, 2; gy, gx = 25, 21
jj, ii = np.meshgrid(np.arange(gy), np.arange(gx), indexing='ij'); flat = np.stack([ii * W / sub, jj * C / sub, np.zeros((gy, gx))], -1)
F = {k: ss.vertex_field(flat, W, C, sub, k) for k in ss.FIELDS}
check('flat sheet: no strain, no tilt, no height', all(np.abs(F[k]).max() < 1e-9 for k in ss.FIELDS), str({k: float(np.abs(v).max()) for k, v in F.items()}))
st = flat.copy(); st[..., 0] *= 1.1
check('uniform 10 % stretch along x: largest stretch 0.1, smallest 0, area strain 0.1 (exact)', np.allclose(ss.vertex_field(st, W, C, sub, 'stretch_max'), 0.1, atol=1e-9) and np.abs(ss.vertex_field(st, W, C, sub, 'stretch_min')).max() < 1e-9 and np.allclose(ss.vertex_field(st, W, C, sub, 'area_strain'), 0.1, atol=1e-9))
R = 20.0; cy = flat.copy(); th = flat[..., 0] / R; cy[..., 0] = R * np.sin(th); cy[..., 2] = R * (1 - np.cos(th)); s_ = ss.vertex_field(cy, W, C, sub, 'stretch_max')
check('rolled onto a cylinder (isometric): membrane strain below 2e-3 although the sheet is strongly curved (height %.1f mm)' % np.ptp(cy[..., 2]), np.abs(s_).max() < 2e-3 and np.ptp(cy[..., 2]) > 5, 'max |stretch| %.2e' % np.abs(s_).max())
tilt = ss.vertex_field(cy, W, C, sub, 'tilt')
check('tilt of the cylinder reaches about the rolled angle (degrees), flat tilt 0', abs(tilt.max() - np.degrees(th.max())) < 5.0, 'max tilt %.1f vs %.1f' % (tilt.max(), np.degrees(th.max())))
nv = gy * gx
for k in ss.FIELDS:
    v = ss.vertex_field(cy, W, C, sub, k); rgb, lg = ss.colours(k, v)
    check('colours %s: uint8 (nv, 3), legend contract (label, vmin <= vmax, 3 ticks, stops, end labels)' % k, rgb.shape == (nv, 3) and rgb.dtype == np.uint8 and lg['vmin'] <= lg['vmax'] and len(lg['ticks']) == 3 and len(lg['stops']) >= 3 and lg['label'] and len(lg['end_labels']) == 2)
rgb, lg = ss.colours('area_strain', np.array([-1.0, 0.0, 1.0]))
check('strain colours: zero is the neutral colour, compression blue, tension red', tuple(rgb[1]) == tuple(ss.NEUTRAL.astype(np.uint8)) and rgb[0][2] > rgb[0][0] and rgb[2][0] > rgb[2][2])
try: ss.vertex_field(flat, W, C, sub, 'bogus'); check('unknown field is refused', False)
except ValueError: check('unknown field is refused', True)
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
