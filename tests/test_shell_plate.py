"""test_shell_plate.py - the plate bending term of scripts/shell_model/shell_sim.py (bending='plate', 2026-11-26) against the exact Kirchhoff plate energy. NEEDS JAX: run it where the
solver runs (the server), from the repository root:  venv/bin/python tests/test_shell_plate.py
Checks: (1) bending energy of imposed quadratic shapes z = (a x^2 + 2 c x y + b y^2)/2 against  integral D/2 [nu tr(K)^2 + (1 - nu) tr(K^2)] dA  over the triangles that carry the term (interior ones for the plate),  K = [[a, c], [c, b]], for a cylinder, a sphere,
a grid-aligned saddle and a 45-degree twist, at three mesh sizes: the ratio must be close to 1 and the same for all four shapes (no grid-direction bias); the old hinge term is printed for comparison.
(2) sign: with the rest curvature K0 = the imposed K the plate energy is far below that of the opposite shape. (3) a free flat sheet given a rest curvature relaxes to a cylinder of that curvature."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'shell_model'))
import numpy as np
import shell_sim as ss
from scipy.optimize import minimize

fails = []
def check(msg, ok, extra=''):
    print(('PASS  ' if ok else 'FAIL  ') + msg + (('   ' + str(extra)) if extra != '' else '')); ok or fails.append(msg)

h, nu, Y, W, C = 0.5, 0.4, 1.0, 1.4, 1.2
D = Y * h**3 / (12 * (1 - nu**2))
def setup(sub, n=12, bending='plate'):
    M = [[1] * n for _ in range(n)]; X0, tri, hin, sg, (nx, ny) = ss.build(M, W, C, sub, (1, 1)); A0, a0, ell = ss.rest_data(X0, tri, hin, sg)
    vg, U, parts = ss.make_energy(X0, tri, hin, A0, a0, ell, h, Y, nu, 0.0, None, bending, return_parts=True)
    return X0, tri, hin, sg, A0, parts

def impose(X0, a, b, c):
    X = X0.copy(); x = X[:, 0] - X[:, 0].mean(); y = X[:, 1] - X[:, 1].mean(); X[:, 2] = 0.5 * (a * x * x + 2 * c * x * y + b * y * y); return X

k = 0.02          # small curvature (k x sheet size << 1), so the imposed lift is nearly isometric and the linear plate energy applies
shapes = dict(cylinder=(k, 0, 0), sphere=(k, k, 0), saddle=(k, -k, 0), twist=(0, 0, k))
print('bending energy / exact plate energy (12 x 12 stitches of 1.4 x 1.2, h 0.5, nu 0.4):')
ratios = {}
for sub in (2, 4, 8):
    for bending in ('plate', 'hinge'):
        X0, tri, hin, sg, A0, parts = setup(sub, bending=bending); row = []
        area = (A0 * ss.plate_data(X0, tri, hin, A0)[2]).sum() if bending == 'plate' else A0.sum()          # the plate term acts on the interior triangles (boundary strip weight 0)
        th0 = np.zeros((len(tri), 2, 2)) if bending == 'plate' else np.zeros(len(hin))
        for name, (a, b, c) in shapes.items():
            Kt = np.array([[a, c], [c, b]]); exact = area * 0.5 * D * (nu * np.trace(Kt)**2 + (1 - nu) * np.trace(Kt @ Kt))
            Eb = float(parts(impose(X0, a, b, c).ravel(), th0)[1]); row.append('%s %.3f' % (name, Eb / exact)); ratios[(sub, bending, name)] = Eb / exact
        print('  sub %d %-5s: %s' % (sub, bending, ', '.join(row)))
for sub in (4, 8):
    r = [ratios[(sub, 'plate', n)] for n in shapes]
    check('plate, sub %d: every shape within 10 %% of the exact energy' % sub, all(0.90 <= x <= 1.10 for x in r), ['%.3f' % x for x in r])
    check('plate, sub %d: no grid-direction bias (aligned saddle and 45-degree twist within 5 %%)' % sub, abs(ratios[(sub, 'plate', 'saddle')] / ratios[(sub, 'plate', 'twist')] - 1) < 0.05)
check('plate is mesh-independent (every shape changes by < 1 % from sub 2 to sub 8; the hinge term is not and is 2-10 x too stiff)', all(abs(ratios[(8, 'plate', n)] / ratios[(2, 'plate', n)] - 1) < 0.01 for n in shapes))

# (2) sign convention: rest curvature +K is satisfied by z = +K-shape, not by its mirror
X0, tri, hin, sg, A0, parts = setup(4)
K0 = ss.rest_tensor(tri, sg, k, -k, 1.0)
Ep = float(parts(impose(X0, k, -k, 0).ravel(), K0)[1]); Em = float(parts(impose(X0, -k, k, 0).ravel(), K0)[1])
check('sign: the shape that matches the rest curvature has far lower bending energy than its mirror', Ep < 0.05 * Em, 'matching %.3g, mirror %.3g' % (Ep, Em))

# (3) a free sheet relaxes to its rest curvature (cylinder along x), plate term
M = [[1] * 10 for _ in range(6)]; X0, tri, hin, sg, (nx, ny) = ss.build(M, W, C, 2, (1, 1)); A0, a0, ell = ss.rest_data(X0, tri, hin, sg)
vg, U = ss.make_energy(X0, tri, hin, A0, a0, ell, h, Y, nu, 0.0, None, 'plate'); kk = 0.1; K0 = ss.rest_tensor(tri, sg, kk, 0.0, 1.0)
x = X0.copy(); x[:, 2] += 1e-4 * np.random.default_rng(0).standard_normal(len(x))
r = minimize(lambda z: tuple(np.asarray(v, float) for v in vg(z, K0)), x.ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=5000, ftol=1e-15, gtol=1e-11))
P = r.x.reshape(-1, 3); xc, yc = P[:, 0] - P[:, 0].mean(), P[:, 1] - P[:, 1].mean()
cf = np.linalg.lstsq(np.c_[xc * xc, yc * yc, xc * yc, xc, yc, np.ones_like(xc)], P[:, 2], rcond=None)[0]
check('free sheet with rest curvature kx = %.2f relaxes to it (fitted kx %.4f, ky %.4f; plan bending of a large curvature is not exactly quadratic)' % (kk, 2 * cf[0], 2 * cf[1]), abs(2 * cf[0] / kk - 1) < 0.15 and abs(2 * cf[1]) < 0.2 * kk)

print('ALL OK' if not fails else 'FAILURES: ' + '; '.join(fails))
