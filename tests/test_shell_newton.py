"""test_shell_newton.py - the Newton solver of scripts/shell_model/shell_sim.py (solver='newton', 2026-11-27). NEEDS JAX: run it where the solver runs (the server), from the repository root:
    venv/bin/python tests/test_shell_newton.py
Checks: (1) the sparse element Hessian of make_hessian equals the JAX Hessian-vector product of the full energy of make_energy, for bending 'hinge' and 'plate', with and without boundary springs,
at a random non-flat state; (2) newton_min reaches a stationary point on a small knit/purl sheet: every growth step converged (status 0), the gradient is far below that of L-BFGS at the same
iteration budget, and the final energy is not above that of a long L-BFGS run; (3) the default solver is still L-BFGS (same result as solver='lbfgs')."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'shell_model'))
import numpy as np, jax, jax.numpy as jnp
import shell_sim as ss

fails = []
def check(msg, ok, extra=''):
    print(('PASS  ' if ok else 'FAIL  ') + msg + (('   ' + str(extra)) if extra != '' else '')); ok or fails.append(msg)

W, C, h, nu, Y = 1.4, 1.2, 0.5, 0.4, 1.0
M = [[1, 1, 2, 2, 1, 1], [1, 2, 2, 1, 1, 2], [2, 2, 1, 1, 2, 2], [2, 1, 1, 2, 2, 1], [1, 1, 2, 2, 1, 1]]
rng = np.random.default_rng(3)
for bending in ('hinge', 'plate'):
    for springs in (0.0, 1e-3):
        X0, tri, hin, sg, (nx, ny) = ss.build(M, W, C, 2, (1, 1)); A0, a0, ell = ss.rest_data(X0, tri, hin, sg)
        bnd = None
        if springs > 0:
            gx, gy = nx*2 + 1, ny*2 + 1; idx = np.arange(len(X0)).reshape(gy, gx); bnd = np.unique(np.r_[idx[0], idx[-1], idx[:, 0], idx[:, -1]])
        vg, U = ss.make_energy(X0, tri, hin, A0, a0, ell, h, Y, nu, springs, bnd, bending)
        hess = ss.make_hessian(X0, tri, hin, A0, a0, ell, h, Y, nu, springs, bnd, bending)
        th0 = ss.rest_angles(X0, hin, sg, -0.6, 0.6, 0.7) if bending == 'hinge' else ss.rest_tensor(tri, sg, -0.6, 0.6, 0.7)
        x = (X0 + 0.15*rng.standard_normal(X0.shape)).ravel()
        g = jax.grad(lambda z: U(z, th0)); Hs = hess(x, th0); err = 0.0
        for _ in range(4):
            v = rng.standard_normal(len(x)); hv = np.asarray(jax.jvp(g, (jnp.asarray(x),), (jnp.asarray(v),))[1]); err = max(err, np.abs(Hs @ v - hv).max()/np.abs(hv).max())
        check('Hessian = JAX Hessian-vector product (%s, springs %g)' % (bending, springs), err < 1e-9, 'max rel. error %.2e' % err)
        check('Hessian symmetric (%s, springs %g)' % (bending, springs), abs(Hs - Hs.T).max() <= 1e-12*abs(Hs).max())

def rigid_diff(A, B):
    """Largest vertex distance between two sheets after the best rigid motion of B onto A (Kabsch): a free sheet may drift or turn as a whole without any change of energy."""
    a = A - A.mean(0); b = B - B.mean(0); u, _, vt = np.linalg.svd(b.T @ a); d = np.sign(np.linalg.det(u @ vt)); R = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((b @ R - a)**2).sum(1)).max())

for bending in ('hinge', 'plate'):
    kw = dict(kappa=0.5, h=0.5, nu=0.4, sub=2, ntile=(1, 1), steps=6, verbose=False, bending=bending)
    st = []; X0, Xn, tri, _, logn = ss.fold(M, W, C, solver='newton', maxiter=60, stats=st, **kw)
    _, Xl, _, _, logl = ss.fold(M, W, C, maxiter=400, **kw)
    _, Xll, _, _, logll = ss.fold(M, W, C, maxiter=20000, **kw)
    check('Newton: every growth step converged (%s)' % bending, all(r[5] == 0 for r in logn), [(r[2], r[5]) for r in logn])
    check('Newton: end gradient < 1e-6 (%s)' % bending, logn[-1][4] < 1e-6, 'newton %.2e, lbfgs-400 %.2e, lbfgs-20000 %.2e' % (logn[-1][4], logl[-1][4], logll[-1][4]))
    check('Newton: energy not above long L-BFGS (%s)' % bending, logn[-1][1] <= logll[-1][1]*(1 + 1e-9), 'U newton %.10g, lbfgs-20000 %.10g' % (logn[-1][1], logll[-1][1]))
    print('      shape difference newton vs lbfgs-20000: %.3g mm raw, %.3g mm after rigid alignment; vs lbfgs-400: %.3g mm aligned' % (np.abs(Xn - Xll).max(), rigid_diff(Xn, Xll), rigid_diff(Xn, Xl)))
    st2 = []; _, Xf, _, _, logf = ss.fold(M, W, C, solver='fire+newton', maxiter=2000, newton_its=10, check_stability=True, stats=st2, **kw)
    check('FIRE+Newton: energy within 1e-7 of Newton (%s)' % bending, abs(logf[-1][1] - logn[-1][1]) <= 1e-7*abs(logn[-1][1]), 'U %.10g vs %.10g, shape %.2e mm aligned' % (logf[-1][1], logn[-1][1], rigid_diff(Xf, Xn)))
    w = np.array(st2[-1][1]); d = st2[-1][2]
    check('stability: no clearly negative eigenvalue (%s)' % bending, w.min() > -1e-6*d, 'lowest %s (scale %.3g)' % (np.round(w[:8]/d, 9).tolist(), d))
    _, Xd, _, _, logd = ss.fold(M, W, C, maxiter=400, solver='lbfgs', **kw)
    check('default solver is L-BFGS (%s)' % bending, np.array_equal(Xd, Xl))

print('ALL OK' if not fails else 'FAILED: %d' % len(fails)); sys.exit(1 if fails else 0)
