#!/usr/bin/env python3
"""relax_solver.py - quasi-static relaxation of the yarn polylines held in a relax_state.Sim.
Energy (positions x, rest lengths l0 held fixed inside one solve; slip changes l0 between solves, see relax_state.migrate):
  E = E_stretch + E_bend + E_contact
  E_stretch = sum_i EA/2 (l_i - l0_i)^2 / l0_i                         Hookean axial law (unsourced assumption, EA user-supplied)
  E_bend    = sum_i EI/(2 lbar_i) |kb_i - kb0_i|^2, kb_i = 2 e_{i-1} x e_i / (|e_{i-1}||e_i| + e_{i-1}.e_i), lbar_i = (l0_{i-1}+l0_i)/2
              (discrete curvature binormal and energy of Bergou, Wardetzky, Robinson, Audoly, Grinspun 2008, Discrete Elastic Rods)
  E_contact = sum_pairs kc/2 max(0, 2R - |x_p - x_q|)^2                 penalty, vertex-vertex, no tangential friction. Form is a numerical
              device, not sourced; kc is user-supplied.
Stress-free (reference) curvature, taken once from the first state seen (before any perturbation):
  kb0 = ref_scale * (curvature of the baseline loops) + curl bias
  ref_scale = 0 (DEFAULT, the latent-energy model): the stress-free yarn is straight, so the tightly packed baseline loops start heavily pre-stressed and store
              high bending energy; releasing it drives buckling and folding. Measured side effect (R-unit test, 3x3 fabric): with no other drive the fabric
              rolls up strongly and keeps creeping (z-range 4.7 R -> 25..35 R, energy still falling after 3000+ L-BFGS iterations); it can also unravel loops.
  ref_scale = 1: the yarn is stress-free in its knitted loop shape; the baseline has zero bending energy and nothing moves unless a curl bias is added.
  0 < ref_scale < 1: partial memory of the loop shape; the response was smooth in the same test (z-range 4.6 R -> 6.6 R at 0.6 and -> 8.5 R at 0.3, 7200 iterations).
  All values are modelling choices, not sourced: calibrate against a known fabric.
Curl bias (curl > 0): a turning angle per vertex (rad) bending the yarn out of the fabric plane, toward +z for front (digit 1) loops and -z for back (digit 2) loops.
  curl_aniso: the sign also flips between wale-direction yarn and course-direction yarn (anticlastic, as single-jersey edge curl: Niu, Dion, Kamien, PNAS 2025).
  A same-sign bias asks the sheet to dome, which needs in-plane stretch; in tests it stalled at a local minimum with most of the stored energy left.
Creep: between solves, rest lengths l0 and reference curvatures kb0 drift toward the current state, x += a (x_now - x), a = 1 - exp(-dt/tau)
  (a single-time-constant Maxwell element). Cotton yarn creep is really a spread of times, so this is a simplification; tau is user-supplied.
Shell hold (optional, sim.shell set by relax_job.py --shell): E_shell = ks/2 sum_cells n_c |mean position of the vertices in the cell - shell target|^2, one cell per stitch (W x C).
  The target is the folded shape from shell_sim.py (Niu et al. 2025). Only the cell mean is held, so loop shape, contact and slip inside a cell stay free. ks is a numerical
  choice (relax_job --shell-w x EA/R). Use ref_scale = 1 with it: with the straight-yarn reference the loops open and fight the hold.
Twist is NOT in this energy: for an isotropic circular section with free ends and no torque source the twist angles can always be set to
minimise the twist energy to zero, so the term would add nothing. Gradients by JAX reverse-mode autodiff (exact), solver scipy L-BFGS-B.
Section properties of a solid circular rod of radius R and effective modulus E: EA = E pi R^2, EI = E pi R^4 / 4 (standard).
"""
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.spatial import cKDTree
jax.config.update('jax_enable_x64', True)

def section(E, R): return E*np.pi*R**2, E*np.pi*R**4/4.0                      # EA, EI

def _energy(x, l0, seg, tri, lbar, kb0, pairs, pmask, EA, EI, kc, R2, cell, tgt, cnt, ks):
    X = x.reshape(-1, 3)
    e = X[seg[:, 1]] - X[seg[:, 0]]; l = jnp.sqrt(jnp.sum(e*e, 1) + 1e-30)
    Es = 0.5*EA*jnp.sum((l - l0)**2/l0)
    e1 = X[tri[:, 1]] - X[tri[:, 0]]; e2 = X[tri[:, 2]] - X[tri[:, 1]]
    n1 = jnp.sqrt(jnp.sum(e1*e1, 1) + 1e-30); n2 = jnp.sqrt(jnp.sum(e2*e2, 1) + 1e-30)
    kb = 2.0*jnp.cross(e1, e2)/(n1*n2 + jnp.sum(e1*e2, 1) + 1e-12)[:, None]
    kd = kb - kb0; Eb = 0.5*EI*jnp.sum(jnp.sum(kd*kd, 1)/lbar)
    d = jnp.sqrt(jnp.sum((X[pairs[:, 0]] - X[pairs[:, 1]])**2, 1) + 1e-30)
    Ec = 0.5*kc*jnp.sum(pmask*jnp.maximum(0.0, R2 - d)**2)
    mean = jax.ops.segment_sum(X, cell, num_segments=tgt.shape[0])/cnt[:, None]          # mean position of the vertices in each stitch cell
    Esh = 0.5*ks*jnp.sum(cnt*jnp.sum((mean - tgt)**2, 1))                                  # hold on the shell fold (ks = 0 when no shell)
    return Es + Eb + Ec + Esh
_vg = jax.jit(jax.value_and_grad(_energy))

def pack(sim):
    off = np.r_[0, np.cumsum([len(y.X) for y in sim.yarns])]; seg, tri, l0, lbar = [], [], [], []
    for y, o in zip(sim.yarns, off):
        n = len(y.X); seg += [(o+i, o+i+1) for i in range(n-1)]; l0 += list(y.l0)
        tri += [(o+i-1, o+i, o+i+1) for i in range(1, n-1)]; lbar += [0.5*(y.l0[i-1] + y.l0[i]) for i in range(1, n-1)]
    return off, np.array(seg), np.array(tri), np.array(l0), np.array(lbar)

def _kb_np(sim, tri):
    X = np.vstack([y.X for y in sim.yarns]); e1 = X[tri[:, 1]] - X[tri[:, 0]]; e2 = X[tri[:, 2]] - X[tri[:, 1]]
    n1 = np.linalg.norm(e1, axis=1); n2 = np.linalg.norm(e2, axis=1)
    return 2.0*np.cross(e1, e2)/(n1*n2 + np.sum(e1*e2, 1) + 1e-12)[:, None], e1/n1[:, None] + e2/n2[:, None]

def baseline_curvature(sim, tri):
    """kb0 per interior vertex, computed once from the first positions seen and kept on the Sim: ref_scale * (baseline loop curvature) + curl bias.
    With ref_scale = 0 and curl = 0 this is exactly the all-zero (straight yarn) reference."""
    if not hasattr(sim, 'kb0'):
        kb, t = _kb_np(sim, tri); kb = getattr(sim.params, 'ref_scale', 0.0)*kb
        if sim.params.curl:
            sg = np.concatenate([y.s[1:-1] for y in sim.yarns]); t = t/np.linalg.norm(t, axis=1)[:, None]
            b = np.cross(t, [0.0, 0.0, 1.0]); nb = np.linalg.norm(b, axis=1)[:, None]
            dsg = np.where(np.abs(t[:, 1]) >= np.abs(t[:, 0]), 1.0, -1.0) if getattr(sim.params, 'curl_aniso', False) else 1.0      # anticlastic: wale-direction yarn one way, course-direction yarn the other
            kb = kb + sim.params.curl*(sg*dsg)[:, None]*np.where(nb > 0.3, b/np.maximum(nb, 1e-9), 0.0)     # b x t = +z: bends toward +z
        sim.kb0 = kb
    return sim.kb0

def creep(sim, a):
    """One creep step, a = 1 - exp(-dt/tau): rest lengths and reference curvatures move a fraction a toward the current state."""
    off, seg, tri, l0, lbar = pack(sim); kb0 = baseline_curvature(sim, tri); kb, _ = _kb_np(sim, tri); sim.kb0 = kb0 + a*(kb - kb0)
    for y in sim.yarns: y.l0 = y.l0 + a*(np.linalg.norm(np.diff(y.X, axis=0), axis=1) - y.l0)

def near_pairs(sim, off, skin=1.5, skip=6, pad=8192):
    P = np.vstack([y.X for y in sim.yarns]); yid = np.concatenate([[k]*len(y.X) for k, y in enumerate(sim.yarns)])
    pr = cKDTree(P).query_pairs(skin*2*sim.R, output_type='ndarray')
    keep = (yid[pr[:, 0]] != yid[pr[:, 1]]) | (np.abs(pr[:, 0] - pr[:, 1]) > skip); pr = pr[keep]
    n = max(pad, int(np.ceil(len(pr)/pad))*pad); out = np.zeros((n, 2), int); m = np.zeros(n); out[:len(pr)] = pr; m[:len(pr)] = 1.0
    return out, m, len(pr)

def shell_args(sim, hold=True):
    """(cell, tgt, cnt, ks) for _energy. No shell set: one dummy cell and ks = 0, so the term is zero."""
    sh = getattr(sim, 'shell', None)
    if sh is None: return np.zeros(sum(len(y.X) for y in sim.yarns), int), np.zeros((1, 3)), np.ones(1), 0.0
    return sh['cell'], sh['tgt'], sh['cnt'], (sh['ks'] if hold else 0.0)

def shell_deviation(sim):
    """RMS and max distance between each stitch cell's mean position and its shell target (fabric units); None without a shell."""
    sh = getattr(sim, 'shell', None)
    if sh is None: return None
    X = np.vstack([y.X for y in sim.yarns]); n = len(sh['cnt'])
    mean = np.c_[[np.bincount(sh['cell'], weights=X[:, k], minlength=n) for k in range(3)]].T/sh['cnt'][:, None]
    d = np.linalg.norm(mean - sh['tgt'], axis=1)
    return dict(rms=float(np.sqrt((d**2).mean())), max=float(d.max()))

def energy_parts(sim, kc, E):
    EA, EI = section(E, sim.R); off, seg, tri, l0, lbar = pack(sim); pairs, pm, _ = near_pairs(sim, off)
    x = np.vstack([y.X for y in sim.yarns]).ravel(); kb0 = baseline_curvature(sim, tri); sa, s0 = shell_args(sim), shell_args(sim, False)
    tot = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, kc, 2*sim.R, *sa))
    st = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, EA, 0.0, 0.0, 0.0, *s0)); bd = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, 0.0, EI, 0.0, 0.0, *s0))
    sh = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, 0.0, 0.0, 0.0, 0.0, *sa))
    return dict(total=tot, stretch=st, bend=bd, shell=sh, contact=tot - st - bd - sh)

def relax(sim, E, kc, maxiter=300, rounds=4, verbose=False):
    """Minimise positions at fixed rest lengths. Contact pairs are rebuilt between rounds (positions move)."""
    EA, EI = section(E, sim.R)
    for r in range(rounds):
        off, seg, tri, l0, lbar = pack(sim); pairs, pm, npair = near_pairs(sim, off); kb0 = baseline_curvature(sim, tri); sa = shell_args(sim)
        x0 = np.vstack([y.X for y in sim.yarns]).ravel()
        f = lambda x: tuple(np.asarray(v, float) for v in _vg(x, l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, kc, 2*sim.R, *sa))
        res = minimize(f, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxcor=30, ftol=1e-13, gtol=1e-9))
        X = res.x.reshape(-1, 3)
        for y, o in zip(sim.yarns, off): y.X = X[o:o+len(y.X)].copy()
        if verbose: print('   relax round %d: %d pairs, E %.6g, %d its, %s' % (r+1, npair, res.fun, res.nit, res.message if isinstance(res.message, str) else res.message.decode()), flush=True)
        if res.nit < maxiter: break

def min_axis_distance(sim, skip=6):
    P = np.vstack([y.X for y in sim.yarns]); yid = np.concatenate([[k]*len(y.X) for k, y in enumerate(sim.yarns)]); ix = np.concatenate([np.arange(len(y.X)) for y in sim.yarns])
    pr = cKDTree(P).query_pairs(2*sim.R*1.5, output_type='ndarray')
    pr = pr[(yid[pr[:, 0]] != yid[pr[:, 1]]) | (np.abs(ix[pr[:, 0]] - ix[pr[:, 1]]) > skip)]
    return float(np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1).min()) if len(pr) else float('inf')