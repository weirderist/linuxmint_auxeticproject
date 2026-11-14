#!/usr/bin/env python3
"""shell_sim.py - stage 1: thin-shell fold prediction from the knit/purl pattern (open source, JAX + SciPy).
Model: Niu, Dion, Kamien (2025), "Geometric modeling of knitted fabrics", PNAS 122(7) e2416536122. The fabric is an elastic sheet with a flat rest metric and a
field of natural (preferred) curvatures set by the stitch type: knit kappa_x = -k, kappa_y = +k, purl the opposite (Fig. 3 of that paper). Folds are the minimiser of
  U = membrane (stretch) energy + bending energy relative to the natural curvature,
found by quasi-static growth: the curvature magnitude is ramped from 0 to its full value and each step starts from the previous solution (the paper's method).
Membrane energy, per triangle (their Eq. 1, first term): (h/4) |a0^-1 (a - a0)|_e^2 , |X|_e^2 = Y nu/(1-nu^2) tr^2 X + Y/(1+nu) tr(X^2), integrated over the rest area.
Bending energy: DISCRETE-SHELL HINGES (Grinspun et al. 2003); the rest dihedral angle of each hinge is the angle its four vertices would have if lifted onto the local
  quadric z = (kx dx^2 + ky dy^2)/2 (checked: an exact cylinder of the preferred curvature has bending+stretch energy -> 0 under mesh refinement, ~h^4).
  This is NOT the paper's finite-difference second-fundamental-form scheme; it is a simpler stand-in with the same preferred curvatures. Bending stiffness D = Y h^3/(12(1-nu^2)).
Not modelled (as in the paper): self-intersection. Free boundaries, with optional weak springs to the flat positions (paper: 1e-4 per unit length) for unbalanced patterns.
Units: lengths in mm (stitch width W, height C); kappa in 1/mm; h in mm. Y only scales the energy.
"""
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)

def build(M, W, C, sub=1, ntile=(1, 1)):
    """M: matrix rows bottom->top of digits 1/2. Returns rest positions, triangles, hinge data, per-triangle stitch sign s (+1 digit 1, -1 digit 2)."""
    T = np.tile(np.array(M), (ntile[1], ntile[0])); ny, nx = T.shape
    gx, gy = nx*sub + 1, ny*sub + 1
    X0 = np.array([(i*W/sub, j*C/sub, 0.0) for j in range(gy) for i in range(gx)])
    tri, sg = [], []
    for j in range(ny*sub):
        for i in range(nx*sub):
            a = j*gx + i; b = a + 1; d = a + gx; c = d + 1; s = 1.0 if T[j//sub, i//sub] == 1 else -1.0
            if (i + j) % 2 == 0: tri += [(a, b, c), (a, c, d)]
            else: tri += [(a, b, d), (b, c, d)]
            sg += [s, s]
    tri = np.array(tri); sg = np.array(sg)
    edges = {}
    for t, (p, q, r) in enumerate(tri):
        for (u, v, w) in ((p, q, r), (q, r, p), (r, p, q)):
            edges.setdefault((min(u, v), max(u, v)), []).append((t, u, v, w))
    hin = []                          # (i, j, k, l, t1, t2): shared edge i->j, k opposite in the triangle on the left of i->j, l opposite in the other
    for (u, v), lst in edges.items():
        if len(lst) != 2: continue
        (t1, a1, b1, o1), (t2, a2, b2, o2) = lst
        if (a1, b1) == (u, v): i, j, k, l = u, v, o1, o2; ta, tb = t1, t2
        else: i, j, k, l = u, v, o2, o1; ta, tb = t2, t1
        hin.append((i, j, k, l, ta, tb))
    hin = np.array(hin)
    return X0, tri, hin, sg, (nx, ny)

def rest_data(X0, tri, hin, sg):
    p = X0[tri]; e1 = p[:, 1, :2] - p[:, 0, :2]; e2 = p[:, 2, :2] - p[:, 0, :2]
    A0 = 0.5*np.abs(e1[:, 0]*e2[:, 1] - e1[:, 1]*e2[:, 0]); a0 = np.stack([np.stack([(e1*e1).sum(1), (e1*e2).sum(1)], 1), np.stack([(e1*e2).sum(1), (e2*e2).sum(1)], 1)], 1)
    i, j, k, l, ta, tb = hin.T; ell = np.linalg.norm(X0[j, :2] - X0[i, :2], axis=1)
    return A0, a0, ell

def rest_angles(X0, hin, sg, kx, ky, scale=1.0):
    """Preferred dihedral angle of every hinge: the angle the same four vertices would have if lifted onto the quadric z = (kx dx^2 + ky dy^2)/2 centred on the hinge
    midpoint, with kx, ky averaged over the two adjacent triangles (so they vanish at a knit/purl interface). >0 bends toward +z."""
    i, j, k, l, ta, tb = hin.T; m = (X0[i] + X0[j])/2; kxh = scale*kx*(sg[ta] + sg[tb])/2; kyh = scale*ky*(sg[ta] + sg[tb])/2
    def lift(q):
        dx = q[:, 0] - m[:, 0]; dy = q[:, 1] - m[:, 1]; return np.stack([q[:, 0], q[:, 1], 0.5*(kxh*dx**2 + kyh*dy**2)], 1)
    xi, xj, xk, xl = lift(X0[i]), lift(X0[j]), lift(X0[k]), lift(X0[l]); ev = xj - xi; eh = ev/np.linalg.norm(ev, axis=1)[:, None]
    n1 = np.cross(xj - xi, xk - xi); n2 = np.cross(xi - xj, xl - xj); n1 /= np.linalg.norm(n1, axis=1)[:, None]; n2 /= np.linalg.norm(n2, axis=1)[:, None]
    return np.arctan2(-(np.cross(n1, n2)*eh).sum(1), (n1*n2).sum(1))

def make_energy(X0, tri, hin, A0, a0, ell, h, Y, nu, spring_w, bnd):
    a0inv = np.linalg.inv(a0); D = Y*h**3/(12*(1 - nu**2)); kb = D/2.0
    i, j, k, l, ta, tb = hin.T; wbend = 3*ell**2/(A0[ta] + A0[tb])
    c1 = Y*nu/(1 - nu**2); c2 = Y/(1 + nu); x0 = jnp.array(X0)
    def U(x, th0):
        X = x.reshape(-1, 3); p = X[tri]; e1 = p[:, 1] - p[:, 0]; e2 = p[:, 2] - p[:, 0]
        a = jnp.stack([jnp.stack([(e1*e1).sum(1), (e1*e2).sum(1)], 1), jnp.stack([(e1*e2).sum(1), (e2*e2).sum(1)], 1)], 1)
        Xm = jnp.einsum('tij,tjk->tik', a0inv, a - a0); tr = Xm[:, 0, 0] + Xm[:, 1, 1]; tr2 = jnp.einsum('tij,tji->t', Xm, Xm)
        Es = 0.5*jnp.sum(A0*(h/4.0)*(c1*tr**2 + c2*tr2))
        xi, xj, xk, xl = X[i], X[j], X[k], X[l]; ev = xj - xi; eh = ev/jnp.sqrt((ev*ev).sum(1, keepdims=True) + 1e-30)
        n1 = jnp.cross(xj - xi, xk - xi); n2 = jnp.cross(xi - xj, xl - xj)
        n1 = n1/jnp.sqrt((n1*n1).sum(1, keepdims=True) + 1e-30); n2 = n2/jnp.sqrt((n2*n2).sum(1, keepdims=True) + 1e-30)
        th = jnp.arctan2(-(jnp.cross(n1, n2)*eh).sum(1), (n1*n2).sum(1))              # >0: surface bends toward +z (concave up)
        Eb = jnp.sum(kb*wbend*(th - th0)**2)
        Esp = 0.5*spring_w*jnp.sum(((X - x0)[bnd]**2).sum(1)) if bnd is not None else 0.0
        return Es + Eb + Esp
    return jax.jit(jax.value_and_grad(U)), jax.jit(U)

def fold(M, W, C, kappa=1.0, h=0.5, nu=0.4, Y=1.0, sub=1, ntile=(1, 1), steps=20, maxiter=400, springs=0.0, seed=0, verbose=True, kx=None, ky=None):
    """kx, ky: natural curvatures of a digit-1 stitch (1/length, > 0 bends toward +z); digit 2 gets the negatives. None = the paper's (-kappa, +kappa)."""
    kx0, ky0 = (-kappa, +kappa) if kx is None else (kx, ky)
    X0, tri, hin, sg, (nx, ny) = build(M, W, C, sub, ntile)
    A0, a0, ell = rest_data(X0, tri, hin, sg)
    bnd = None
    if springs > 0:
        gx, gy = nx*sub + 1, ny*sub + 1; idx = np.arange(len(X0)).reshape(gy, gx); bnd = np.unique(np.r_[idx[0], idx[-1], idx[:, 0], idx[:, -1]])
    vg, U = make_energy(X0, tri, hin, A0, a0, ell, h, Y, nu, springs, bnd)
    x = X0.copy(); x[:, 2] += 1e-3*np.random.default_rng(seed).standard_normal(len(x)); x = x.ravel(); log = []
    for s in np.linspace(1.0/steps, 1.0, steps):
        th0 = rest_angles(X0, hin, sg, kx0, ky0, s)
        r = minimize(lambda z: tuple(np.asarray(v, float) for v in vg(z, th0)), x, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxcor=30, ftol=1e-14, gtol=1e-9)); x = r.x
        Xc = x.reshape(-1, 3); log.append((float(s), float(r.fun), int(r.nit), float(np.ptp(Xc[:, 2]))))
        if verbose: print('  growth %.2f: U %.5g, %d its, z-range %.3f mm' % log[-1], flush=True)
    return X0, x.reshape(-1, 3), tri, (nx, ny), log#!/usr/bin/env python3
"""shell_sim.py - stage 1: thin-shell fold prediction from the knit/purl pattern (open source, JAX + SciPy).
Model: Niu, Dion, Kamien (2025), "Geometric modeling of knitted fabrics", PNAS 122(7) e2416536122. The fabric is an elastic sheet with a flat rest metric and a
field of natural (preferred) curvatures set by the stitch type: knit kappa_x = -k, kappa_y = +k, purl the opposite (Fig. 3 of that paper). Folds are the minimiser of
  U = membrane (stretch) energy + bending energy relative to the natural curvature,
found by quasi-static growth: the curvature magnitude is ramped from 0 to its full value and each step starts from the previous solution (the paper's method).
Membrane energy, per triangle (their Eq. 1, first term): (h/4) |a0^-1 (a - a0)|_e^2 , |X|_e^2 = Y nu/(1-nu^2) tr^2 X + Y/(1+nu) tr(X^2), integrated over the rest area.
Bending energy: DISCRETE-SHELL HINGES (Grinspun et al. 2003); the rest dihedral angle of each hinge is the angle its four vertices would have if lifted onto the local
  quadric z = (kx dx^2 + ky dy^2)/2 (checked: an exact cylinder of the preferred curvature has bending+stretch energy -> 0 under mesh refinement, ~h^4).
  This is NOT the paper's finite-difference second-fundamental-form scheme; it is a simpler stand-in with the same preferred curvatures. Bending stiffness D = Y h^3/(12(1-nu^2)).
Not modelled (as in the paper): self-intersection. Free boundaries, with optional weak springs to the flat positions (paper: 1e-4 per unit length) for unbalanced patterns.
Units: lengths in mm (stitch width W, height C); kappa in 1/mm; h in mm. Y only scales the energy.
"""
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)

def build(M, W, C, sub=1, ntile=(1, 1)):
    """M: matrix rows bottom->top of digits 1/2. Returns rest positions, triangles, hinge data, per-triangle stitch sign s (+1 digit 1, -1 digit 2)."""
    T = np.tile(np.array(M), (ntile[1], ntile[0])); ny, nx = T.shape
    gx, gy = nx*sub + 1, ny*sub + 1
    X0 = np.array([(i*W/sub, j*C/sub, 0.0) for j in range(gy) for i in range(gx)])
    tri, sg = [], []
    for j in range(ny*sub):
        for i in range(nx*sub):
            a = j*gx + i; b = a + 1; d = a + gx; c = d + 1; s = 1.0 if T[j//sub, i//sub] == 1 else -1.0
            if (i + j) % 2 == 0: tri += [(a, b, c), (a, c, d)]
            else: tri += [(a, b, d), (b, c, d)]
            sg += [s, s]
    tri = np.array(tri); sg = np.array(sg)
    edges = {}
    for t, (p, q, r) in enumerate(tri):
        for (u, v, w) in ((p, q, r), (q, r, p), (r, p, q)):
            edges.setdefault((min(u, v), max(u, v)), []).append((t, u, v, w))
    hin = []                          # (i, j, k, l, t1, t2): shared edge i->j, k opposite in the triangle on the left of i->j, l opposite in the other
    for (u, v), lst in edges.items():
        if len(lst) != 2: continue
        (t1, a1, b1, o1), (t2, a2, b2, o2) = lst
        if (a1, b1) == (u, v): i, j, k, l = u, v, o1, o2; ta, tb = t1, t2
        else: i, j, k, l = u, v, o2, o1; ta, tb = t2, t1
        hin.append((i, j, k, l, ta, tb))
    hin = np.array(hin)
    return X0, tri, hin, sg, (nx, ny)

def rest_data(X0, tri, hin, sg):
    p = X0[tri]; e1 = p[:, 1, :2] - p[:, 0, :2]; e2 = p[:, 2, :2] - p[:, 0, :2]
    A0 = 0.5*np.abs(e1[:, 0]*e2[:, 1] - e1[:, 1]*e2[:, 0]); a0 = np.stack([np.stack([(e1*e1).sum(1), (e1*e2).sum(1)], 1), np.stack([(e1*e2).sum(1), (e2*e2).sum(1)], 1)], 1)
    i, j, k, l, ta, tb = hin.T; ell = np.linalg.norm(X0[j, :2] - X0[i, :2], axis=1)
    return A0, a0, ell

def rest_angles(X0, hin, sg, kx, ky, scale=1.0):
    """Preferred dihedral angle of every hinge: the angle the same four vertices would have if lifted onto the quadric z = (kx dx^2 + ky dy^2)/2 centred on the hinge
    midpoint, with kx, ky averaged over the two adjacent triangles (so they vanish at a knit/purl interface). >0 bends toward +z."""
    i, j, k, l, ta, tb = hin.T; m = (X0[i] + X0[j])/2; kxh = scale*kx*(sg[ta] + sg[tb])/2; kyh = scale*ky*(sg[ta] + sg[tb])/2
    def lift(q):
        dx = q[:, 0] - m[:, 0]; dy = q[:, 1] - m[:, 1]; return np.stack([q[:, 0], q[:, 1], 0.5*(kxh*dx**2 + kyh*dy**2)], 1)
    xi, xj, xk, xl = lift(X0[i]), lift(X0[j]), lift(X0[k]), lift(X0[l]); ev = xj - xi; eh = ev/np.linalg.norm(ev, axis=1)[:, None]
    n1 = np.cross(xj - xi, xk - xi); n2 = np.cross(xi - xj, xl - xj); n1 /= np.linalg.norm(n1, axis=1)[:, None]; n2 /= np.linalg.norm(n2, axis=1)[:, None]
    return np.arctan2(-(np.cross(n1, n2)*eh).sum(1), (n1*n2).sum(1))

def make_energy(X0, tri, hin, A0, a0, ell, h, Y, nu, spring_w, bnd):
    a0inv = np.linalg.inv(a0); D = Y*h**3/(12*(1 - nu**2)); kb = D/2.0
    i, j, k, l, ta, tb = hin.T; wbend = 3*ell**2/(A0[ta] + A0[tb])
    c1 = Y*nu/(1 - nu**2); c2 = Y/(1 + nu); x0 = jnp.array(X0)
    def U(x, th0):
        X = x.reshape(-1, 3); p = X[tri]; e1 = p[:, 1] - p[:, 0]; e2 = p[:, 2] - p[:, 0]
        a = jnp.stack([jnp.stack([(e1*e1).sum(1), (e1*e2).sum(1)], 1), jnp.stack([(e1*e2).sum(1), (e2*e2).sum(1)], 1)], 1)
        Xm = jnp.einsum('tij,tjk->tik', a0inv, a - a0); tr = Xm[:, 0, 0] + Xm[:, 1, 1]; tr2 = jnp.einsum('tij,tji->t', Xm, Xm)
        Es = 0.5*jnp.sum(A0*(h/4.0)*(c1*tr**2 + c2*tr2))
        xi, xj, xk, xl = X[i], X[j], X[k], X[l]; ev = xj - xi; eh = ev/jnp.sqrt((ev*ev).sum(1, keepdims=True) + 1e-30)
        n1 = jnp.cross(xj - xi, xk - xi); n2 = jnp.cross(xi - xj, xl - xj)
        n1 = n1/jnp.sqrt((n1*n1).sum(1, keepdims=True) + 1e-30); n2 = n2/jnp.sqrt((n2*n2).sum(1, keepdims=True) + 1e-30)
        th = jnp.arctan2(-(jnp.cross(n1, n2)*eh).sum(1), (n1*n2).sum(1))              # >0: surface bends toward +z (concave up)
        Eb = jnp.sum(kb*wbend*(th - th0)**2)
        Esp = 0.5*spring_w*jnp.sum(((X - x0)[bnd]**2).sum(1)) if bnd is not None else 0.0
        return Es + Eb + Esp
    return jax.jit(jax.value_and_grad(U)), jax.jit(U)

def fold(M, W, C, kappa=1.0, h=0.5, nu=0.4, Y=1.0, sub=1, ntile=(1, 1), steps=20, maxiter=400, springs=0.0, seed=0, verbose=True, kx=None, ky=None):
    """kx, ky: natural curvatures of a digit-1 stitch (1/length, > 0 bends toward +z); digit 2 gets the negatives. None = the paper's (-kappa, +kappa)."""
    kx0, ky0 = (-kappa, +kappa) if kx is None else (kx, ky)
    X0, tri, hin, sg, (nx, ny) = build(M, W, C, sub, ntile)
    A0, a0, ell = rest_data(X0, tri, hin, sg)
    bnd = None
    if springs > 0:
        gx, gy = nx*sub + 1, ny*sub + 1; idx = np.arange(len(X0)).reshape(gy, gx); bnd = np.unique(np.r_[idx[0], idx[-1], idx[:, 0], idx[:, -1]])
    vg, U = make_energy(X0, tri, hin, A0, a0, ell, h, Y, nu, springs, bnd)
    x = X0.copy(); x[:, 2] += 1e-3*np.random.default_rng(seed).standard_normal(len(x)); x = x.ravel(); log = []
    for s in np.linspace(1.0/steps, 1.0, steps):
        th0 = rest_angles(X0, hin, sg, kx0, ky0, s)
        r = minimize(lambda z: tuple(np.asarray(v, float) for v in vg(z, th0)), x, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxcor=30, ftol=1e-14, gtol=1e-9)); x = r.x
        Xc = x.reshape(-1, 3); log.append((float(s), float(r.fun), int(r.nit), float(np.ptp(Xc[:, 2]))))
        if verbose: print('  growth %.2f: U %.5g, %d its, z-range %.3f mm' % log[-1], flush=True)
    return X0, x.reshape(-1, 3), tri, (nx, ny), log