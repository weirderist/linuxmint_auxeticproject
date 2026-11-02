#!/usr/bin/env python3
"""relax_solver.py - modified for high-stress latent energy storage and buckling.
Changes from original: baseline curvature kb0 is set to 0 (straight reference state), 
meaning the initial tightly packed loops are heavily pre-stressed and store high bending energy.
When released, this stored strain energy drives natural out-of-plane buckling and folding.
"""
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.spatial import cKDTree
jax.config.update('jax_enable_x64', True)

def section(E, R): return E*np.pi*R**2, E*np.pi*R**4/4.0

def _energy(x, l0, seg, tri, lbar, kb0, pairs, pmask, EA, EI, kc, R2):
    X = x.reshape(-1, 3)
    e = X[seg[:, 1]] - X[seg[:, 0]]; l = jnp.sqrt(jnp.sum(e*e, 1) + 1e-30)
    Es = 0.5*EA*jnp.sum((l - l0)**2/l0)
    e1 = X[tri[:, 1]] - X[tri[:, 0]]; e2 = X[tri[:, 2]] - X[tri[:, 1]]
    n1 = jnp.sqrt(jnp.sum(e1*e1, 1) + 1e-30); n2 = jnp.sqrt(jnp.sum(e2*e2, 1) + 1e-30)
    kb = 2.0*jnp.cross(e1, e2)/(n1*n2 + jnp.sum(e1*e2, 1) + 1e-12)[:, None]
    
    # kd uses kb0 = 0 (straight state reference), forcing initial loops to store max bending energy
    kd = kb - kb0; Eb = 0.5*EI*jnp.sum(jnp.sum(kd*kd, 1)/lbar)
    
    d = jnp.sqrt(jnp.sum((X[pairs[:, 0]] - X[pairs[:, 1]])**2, 1) + 1e-30)
    Ec = 0.5*kc*jnp.sum(pmask*jnp.maximum(0.0, R2 - d)**2)
    return Es + Eb + Ec
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
    """CRITICAL FIX: Setting kb0 to 0 (or near-zero) makes a straight yarn stress-free. 
    Thus, the initial knitted loops store massive internal bending strain energy."""
    if not hasattr(sim, 'kb0'):
        # Initialize kb0 as a zero array so the initial curved loops carry high bending potential
        _, t = _kb_np(sim, tri)
        sim.kb0 = np.zeros((len(tri), 3))
    return sim.kb0

def creep(sim, a):
    off, seg, tri, l0, lbar = pack(sim); kb0 = baseline_curvature(sim, tri); kb, _ = _kb_np(sim, tri); sim.kb0 = kb0 + a*(kb - kb0)
    for y in sim.yarns: y.l0 = y.l0 + a*(np.linalg.norm(np.diff(y.X, axis=0), axis=1) - y.l0)

def near_pairs(sim, off, skin=1.5, skip=6, pad=8192):
    P = np.vstack([y.X for y in sim.yarns]); yid = np.concatenate([[k]*len(y.X) for k, y in enumerate(sim.yarns)])
    pr = cKDTree(P).query_pairs(skin*2*sim.R, output_type='ndarray')
    keep = (yid[pr[:, 0]] != yid[pr[:, 1]]) | (np.abs(pr[:, 0] - pr[:, 1]) > skip); pr = pr[keep]
    n = max(pad, int(np.ceil(len(pr)/pad))*pad); out = np.zeros((n, 2), int); m = np.zeros(n); out[:len(pr)] = pr; m[:len(pr)] = 1.0
    return out, m, len(pr)

def energy_parts(sim, kc, E):
    EA, EI = section(E, sim.R); off, seg, tri, l0, lbar = pack(sim); pairs, pm, _ = near_pairs(sim, off)
    x = np.vstack([y.X for y in sim.yarns]).ravel(); kb0 = baseline_curvature(sim, tri)
    tot = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, kc, 2*sim.R))
    st = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, EA, 0.0, 0.0, 0.0)); bd = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, 0.0, EI, 0.0, 0.0))
    return dict(total=tot, stretch=st, bend=bd, contact=tot - st - bd)

def relax(sim, E, kc, maxiter=300, rounds=4, verbose=False):
    EA, EI = section(E, sim.R)
    for r in range(rounds):
        off, seg, tri, l0, lbar = pack(sim); pairs, pm, npair = near_pairs(sim, off); kb0 = baseline_curvature(sim, tri)
        x0 = np.vstack([y.X for y in sim.yarns]).ravel()
        f = lambda x: tuple(np.asarray(v, float) for v in _vg(x, l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, kc, 2*sim.R))
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