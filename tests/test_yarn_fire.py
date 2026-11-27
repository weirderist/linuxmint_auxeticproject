#!/usr/bin/env python3
"""test_yarn_fire.py - the FIRE minimiser of scripts/yarn_model/relax_solver.py (SOLVER = 'fire', 2026-11-27) against SciPy L-BFGS-B on the SAME yarn energy. NEEDS JAX (numpy, scipy, jax):
    python tests/test_yarn_fire.py      (run from the repository root; the workstation venv has JAX)
Checks: (1) two crossing yarns, compressed rest lengths, perturbed start: both minimisers converge (FIRE to max |grad| < FIRE_GTOL_REL x EA); (2) FIRE and L-BFGS reach the same energy
(relative difference < 1e-6); (3) the energy is not above the start and every part (stretch, bend, contact) agrees; (4) the minimum is a minimum: the lowest Hessian eigenvalue of the yarn
energy (dense, small case) is not negative beyond the rigid modes."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'yarn_model'))
import numpy as np, jax, copy
import relax_solver as sv

fails = []
def check(msg, ok, extra=''):
    print(('PASS  ' if ok else 'FAIL  ') + msg + (('   ' + str(extra)) if extra != '' else '')); ok or fails.append(msg)

class Y: pass
class P: ref_scale = 0.0; curl = 0.0
class Sim: pass

def make_sim(seed=0):
    rng = np.random.default_rng(seed); sim = Sim(); sim.R = 0.1; sim.params = P(); sim.yarns = []
    A = Y(); n = 12; A.X = np.c_[np.arange(n)*1.0, np.zeros(n), np.zeros(n)]; A.l0 = np.full(n - 1, 0.8)          # compressed rest lengths: the yarn must buckle
    B = Y(); B.X = np.c_[np.full(n, 5.5), np.arange(n)*1.0, np.full(n, 0.3)]; B.l0 = np.full(n - 1, 0.8)
    for y in (A, B): y.X = y.X + 0.02*rng.standard_normal(y.X.shape); y.s = np.r_[0, np.cumsum(y.l0)]
    sim.yarns = [A, B]; return sim

def run(sim, solver, E=1.0, kc=2.0):
    sv.SOLVER = solver; t = time.time(); sv.relax(sim, E, kc, maxiter=(300 if solver == 'lbfgs' else sv.FIRE_MAXIT), rounds=1); dt = time.time() - t
    return sim, dt

def parts_and_grad(sim, E=1.0, kc=2.0):
    parts = sv.energy_parts(sim, kc, E); off, seg, tri, l0, lbar = sv.pack(sim); pairs, pm, _ = sv.near_pairs(sim, off); kb0 = sv.baseline_curvature(sim, tri)
    EA, EI = sv.bend_stiffness(sim, E); x = np.vstack([y.X for y in sim.yarns]).ravel(); args = (l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, kc, 2*sim.R, *sv.shell_args(sim))
    g = np.asarray(sv._vg(x, *args)[1]); return parts, np.abs(g).max(), EA, x, args

base = make_sim(0)
start = parts_and_grad(base)[0]['total']
s_l, t_l = run(copy.deepcopy(base), 'lbfgs'); s_f, t_f = run(copy.deepcopy(base), 'fire')
pl, gl, EA, xl, argsl = parts_and_grad(s_l); pf, gf, _, xf, argsf = parts_and_grad(s_f)
print('start energy %.6g | L-BFGS %.8g in %.2f s, max|grad| %.2e | FIRE %.8g in %.2f s, max|grad| %.2e (gtol %.2e)' % (start, pl['total'], t_l, gl, pf['total'], t_f, gf, sv.FIRE_GTOL_REL*EA))
check('(3) energy does not rise from the start (L-BFGS and FIRE)', pl['total'] <= start + 1e-12 and pf['total'] <= start + 1e-12)
check('(1) FIRE converged: max |grad| below its tolerance', gf < sv.FIRE_GTOL_REL*EA*1.001, '%.2e' % gf)
rel = abs(pf['total'] - pl['total'])/max(start, 1e-30)          # relative to the start energy: both minima here are close to zero, so a relative test of the minimum itself would be meaningless
check('(2) FIRE and L-BFGS reach the same energy (difference < 1e-6 of the start energy)', rel < 1e-6, 'diff/start %.2e' % rel)
for k in ('stretch', 'bend', 'contact'):
    d = abs(pf[k] - pl[k]); check('(2) %s part agrees (1e-6 of the start energy)' % k, d <= 1e-6*start, 'diff %.2e' % d)

# (4) stability at the FIRE minimum (dense Hessian of this small case)
import jax.numpy as jnp
H = np.asarray(jax.hessian(lambda z: sv._vg(z, *argsf)[0])(jnp.asarray(xf)))
w = np.linalg.eigvalsh(0.5*(H + H.T)); scale = np.abs(w).max()
neg = w[w < -1e-8*scale]
check('(4) no negative curvature beyond the rigid modes at the FIRE minimum', len(neg) == 0, 'lowest %s (scale %.2e)' % (np.round(w[:6]/scale, 9).tolist(), scale))

sv.SOLVER = 'lbfgs'
print('ALL OK' if not fails else 'FAILED: %s' % fails); sys.exit(1 if fails else 0)
