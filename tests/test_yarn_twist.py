#!/usr/bin/env python3
"""test_yarn_twist.py - the twist drive of scripts/yarn_model/relax_solver.py (2026-11-27; preferred torsion tau0 = live x 2 pi TPI / 25.4, sign Z +, S -). NEEDS JAX:
    python tests/test_yarn_twist.py      (from the repository root; the workstation venv has JAX)
Checks: (1) off (TPI 0 or live 0): no arguments, the energy is the old one bit for bit; (2) the discrete torsion of a helix equals its analytic torsion c / (r^2 + c^2) (within 2 %);
(3) the energy gradient with the term equals central finite differences; (4) a flat curved yarn given a preferred torsion relaxes OUT of the plane with torsion of the preferred sign,
Z and S opposite (the skew drive), and with live = 0 it stays flat."""
import os, sys, copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'yarn_model'))
import numpy as np, jax, jax.numpy as jnp
import relax_solver as sv

fails = []
def check(msg, ok, extra=''):
    print(('PASS  ' if ok else 'FAIL  ') + msg + (('   ' + str(extra)) if extra != '' else '')); ok or fails.append(msg)

class Y: pass
class Sim: pass
def P(**kw):
    p = type('P', (), dict(ref_scale=1.0, curl=0.0, tpi=0.0, twist_dir='Z', twist_live=0.0))(); [setattr(p, k, v) for k, v in kw.items()]; return p
def sim_of(X, **kw):
    s = Sim(); s.R = 0.05; s.params = P(**kw); y = Y(); y.X = np.asarray(X, float); y.l0 = np.linalg.norm(np.diff(y.X, axis=0), axis=1); y.s = np.r_[0, np.cumsum(y.l0)]; s.yarns = [y]; return s
def args_of(sim, E=1.0):
    off, seg, tri, l0, lbar = sv.pack(sim); pairs, pm, _ = sv.near_pairs(sim, off); kb0 = sv.baseline_curvature(sim, tri); EA, EI = sv.bend_stiffness(sim, E)
    return (l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, 1.0, 2*sim.R, *sv.shell_args(sim)), sv.twist_args(sim, tri, E), tri

# (1) off
t = np.linspace(0, 3*np.pi, 60); helix = np.c_[1.0*np.cos(t), 1.0*np.sin(t), 0.3*t]
s0 = sim_of(helix); a0, ta0, tri = args_of(s0); s1 = sim_of(helix, tpi=10.0, twist_live=0.0); a1, ta1, _ = args_of(s1)
check('(1) TPI given but live 0: no twist arguments, energy identical', ta1 == () and float(sv._energy(helix.ravel(), *a0)) == float(sv._energy(helix.ravel(), *a1, *ta1)))
check('(1) TPI 0: no twist arguments', args_of(sim_of(helix, tpi=0.0, twist_live=1.0))[1] == ())

# (2) discrete torsion of a helix: r = 1, c = 0.3 -> tau = c / (r^2 + c^2)
s2 = sim_of(helix, tpi=10.0, twist_live=1.0); a2, ta2, tri = args_of(s2); kb = jnp.asarray(sv._kb_np(s2, tri)[0])
_, tau, w = sv._torsion(jnp.asarray(helix), tri, kb, jnp.asarray(ta2[0]), ta2[2]); tau = np.asarray(tau)[5:-5]; exact = 0.3/(1 + 0.09)
check('(2) helix torsion = c/(r^2+c^2) within 2 %', abs(tau.mean()/exact - 1) < 0.02, 'measured %.5f exact %.5f' % (tau.mean(), exact))
check('(2) Z twist gives tau0 > 0, S gives tau0 < 0', ta2[2] > 0 and args_of(sim_of(helix, tpi=10.0, twist_live=1.0, twist_dir='S'))[1][2] < 0, ta2[2])

# (3) gradient against central differences
x = helix.ravel() + 0.01*np.random.default_rng(0).standard_normal(helix.size); f = lambda z: float(sv._energy(z, *a2, *ta2))
g = np.asarray(jax.grad(lambda z: sv._energy(z, *a2, *ta2))(jnp.asarray(x))); idx = np.random.default_rng(1).choice(x.size, 12, replace=False); h = 1e-6
fd = np.array([(f(x + h*np.eye(x.size)[i]) - f(x - h*np.eye(x.size)[i]))/(2*h) for i in idx]); err = np.abs(fd - g[idx]).max()/max(np.abs(g[idx]).max(), 1e-30)
check('(3) gradient = central differences (rel. 1e-5)', err < 1e-5, 'rel err %.2e' % err)

# (4) a flat arc (radius 2, rest = its own curvature, tiny z noise) with a preferred torsion relaxes out of the plane, sign as tau0
ang = np.linspace(0, 1.5*np.pi, 70); arc = np.c_[2*np.cos(ang), 2*np.sin(ang), 1e-4*np.random.default_rng(2).standard_normal(70)]
res = {}
for dirn, live in (('Z', 1.0), ('S', 1.0), ('Z', 0.0)):
    s = sim_of(arc, tpi=8.0, twist_dir=dirn, twist_live=live); sv.SOLVER = 'lbfgs'; sv.relax(s, 1.0, 1.0, maxiter=3000, rounds=3)
    a, ta, tri = args_of(s); X = s.yarns[0].X; zr = float(np.ptp(X[:, 2]))
    tau = float(np.mean(np.asarray(sv._torsion(jnp.asarray(X), tri, jnp.asarray(sv._kb_np(s, tri)[0]), jnp.asarray(ta[0]) if ta else jnp.asarray(args_of(sim_of(arc, tpi=8.0, twist_live=1.0))[1][0]), 0.0)[1])[5:-5]))
    res[(dirn, live)] = (zr, tau)
print('relaxed arc: z-range / mean torsion  Z: %.4f / %+.4f   S: %.4f / %+.4f   off: %.4f / %+.4f   (tau0 Z = %+.4f rad/mm)' % (*res[('Z', 1.0)], *res[('S', 1.0)], *res[('Z', 0.0)], 1.0*2*np.pi*8/25.4))
off = max(res[('Z', 0.0)][0], 1e-6)
check('(4) with the drive the arc leaves the plane (z-range at least 20 x the no-drive case, which stays flat < 1e-2 mm)', res[('Z', 1.0)][0] > 20*off and res[('S', 1.0)][0] > 20*off and res[('Z', 0.0)][0] < 1e-2, [round(v[0], 4) for v in res.values()])
check('(4) torsion follows the sign of the drive (Z +, S -)', res[('Z', 1.0)][1] > 0 > res[('S', 1.0)][1], (res[('Z', 1.0)][1], res[('S', 1.0)][1]))
print('ALL OK' if not fails else 'FAILED: %s' % fails); sys.exit(1 if fails else 0)
