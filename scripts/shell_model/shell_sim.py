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

def plate_data(X0, tri, hin, A0):
    """Rest data of the PLATE bending term (bending='plate', added 2026-11-26). To first order the dihedral angle of an edge between triangles a and b is theta_e = n . (g_b - g_a): g = the gradient of the
    flat triangle's height (the jump is along the in-plane edge normal n, pointing from a to b, because both triangles share the edge). For a quadratic surface z = (K_xx x^2 + 2 K_xy x y + K_yy y^2)/2, g of
    a flat triangle through its three vertices is exactly linear in K (it is computed here for the three basis quadrics x^2/2, y^2/2, x y; note it is NOT the surface gradient at the centroid, but the one
    that matches the surface slope along each edge at the edge midpoint). So theta_e = G_e . (K_xx, K_yy, K_xy), and a triangle with three interior edges has three such equations:
    K_t = G_t^-1 theta, exact for every quadratic surface whatever the edge directions. (Two simpler estimators were tried and failed tests/test_shell_plate.py on 2026-11-26: the Grinspun et al. 2006
    sum theta |e| n n^T / (2A) gave 3.3 x the energy of a 45-degree twist on this grid; a centroid-gradient model gave 2.2-8.8 x.)
    Triangles with a boundary edge get weight 0 (no neighbour across it); a free edge carries no bending moment, and the strip is O(1/N) of the sheet. Returns (H hinge index per triangle edge,
    len(hin) for a boundary edge; Ginv (T, 3, 3); wt (T,) 1 for interior triangles, 0 for boundary ones)."""
    hid = {}
    for h_, (i, j, k, l, ta, tb) in enumerate(hin): hid[(min(i, j), max(i, j))] = (h_, ta, tb)
    P = X0[:, :2] - X0[:, :2].mean(0); Z = np.c_[0.5*P[:, 0]**2, 0.5*P[:, 1]**2, P[:, 0]*P[:, 1]]          # vertex heights of the three basis quadrics
    E1 = P[tri[:, 1]] - P[tri[:, 0]]; E2 = P[tri[:, 2]] - P[tri[:, 0]]; Minv = np.linalg.inv(np.stack([E1, E2], 1))                 # g . e1 = dz1, g . e2 = dz2
    g = np.einsum('tij,tjb->tib', Minv, np.stack([Z[tri[:, 1]] - Z[tri[:, 0]], Z[tri[:, 2]] - Z[tri[:, 0]]], 1))                     # (T, 2 components, 3 basis)
    ctr = P[tri].mean(1); T = len(tri); H = np.full((T, 3), len(hin), int); Ginv = np.zeros((T, 3, 3)); wt = np.zeros(T)
    for t, (p, q, r) in enumerate(tri):
        G = np.zeros((3, 3)); ok = True
        for s_, (a, b) in enumerate(((p, q), (q, r), (r, p))):
            hh = hid.get((min(a, b), max(a, b)))
            if hh is None: ok = False; continue
            H[t, s_] = hh[0]; nb = hh[2] if hh[1] == t else hh[1]
            e = P[b] - P[a]; n = np.array([e[1], -e[0]])/np.linalg.norm(e); n = n if n @ (ctr[nb] - ctr[t]) > 0 else -n          # normal pointing from t to its neighbour
            G[s_] = n @ (g[nb] - g[t])
        if ok: Ginv[t] = np.linalg.inv(G); wt[t] = 1.0
    return H, Ginv, wt

def make_energy(X0, tri, hin, A0, a0, ell, h, Y, nu, spring_w, bnd, bending='hinge', return_parts=False):
    """bending = 'hinge' (default, the energy of every run up to shell1, kept bit-identical): hinge energy kb wbend (theta - theta0)^2. MEASURED 2026-11-26 (tests/test_shell_plate.py, run on the server):
    about 2.9 x the plate energy for a cylinder, 4.9 x for a grid-aligned saddle and 10 x for a 45-degree twist, independent of the mesh size (it behaves as if nu = 0 and depends on the grid direction).
    bending = 'plate': the Kirchhoff plate energy  sum_t A_t D/2 [nu tr(dS)^2 + (1 - nu) tr(dS dS)],  dS = S_t - S0_t, S_t the discrete shape operator of plate_data and S0_t = diag(kx, ky) x the stitch sign,
    i.e. the bending term of Niu et al. 2025 Eq. 1 with the curvature taken from the dihedral angles. Then th0 is the (T, 2, 2) rest curvature tensor instead of the hinge angles."""
    a0inv = np.linalg.inv(a0); D = Y*h**3/(12*(1 - nu**2)); kb = D/2.0
    i, j, k, l, ta, tb = hin.T; wbend = 3*ell**2/(A0[ta] + A0[tb])
    c1 = Y*nu/(1 - nu**2); c2 = Y/(1 + nu); x0 = jnp.array(X0)
    if bending not in ('hinge', 'plate'): raise ValueError("bending must be 'hinge' or 'plate'")
    if bending == 'plate': H, Ginv, wt = plate_data(X0, tri, hin, A0); Aw = A0*wt
    def parts(x, th0):
        X = x.reshape(-1, 3); p = X[tri]; e1 = p[:, 1] - p[:, 0]; e2 = p[:, 2] - p[:, 0]
        a = jnp.stack([jnp.stack([(e1*e1).sum(1), (e1*e2).sum(1)], 1), jnp.stack([(e1*e2).sum(1), (e2*e2).sum(1)], 1)], 1)
        Xm = jnp.einsum('tij,tjk->tik', a0inv, a - a0); tr = Xm[:, 0, 0] + Xm[:, 1, 1]; tr2 = jnp.einsum('tij,tji->t', Xm, Xm)
        Es = 0.5*jnp.sum(A0*(h/4.0)*(c1*tr**2 + c2*tr2))
        xi, xj, xk, xl = X[i], X[j], X[k], X[l]; ev = xj - xi; eh = ev/jnp.sqrt((ev*ev).sum(1, keepdims=True) + 1e-30)
        n1 = jnp.cross(xj - xi, xk - xi); n2 = jnp.cross(xi - xj, xl - xj)
        n1 = n1/jnp.sqrt((n1*n1).sum(1, keepdims=True) + 1e-30); n2 = n2/jnp.sqrt((n2*n2).sum(1, keepdims=True) + 1e-30)
        th = jnp.arctan2(-(jnp.cross(n1, n2)*eh).sum(1), (n1*n2).sum(1))              # >0: surface bends toward +z (concave up)
        if bending == 'hinge': Eb = jnp.sum(kb*wbend*(th - th0)**2)
        else:
            kv = jnp.einsum('tij,tj->ti', Ginv, jnp.concatenate([th, jnp.zeros(1)])[H])          # (K_xx, K_yy, K_xy) per triangle
            dxx, dyy, dxy = kv[:, 0] - th0[:, 0, 0], kv[:, 1] - th0[:, 1, 1], kv[:, 2] - th0[:, 0, 1]
            Eb = jnp.sum(Aw*0.5*D*(nu*(dxx + dyy)**2 + (1 - nu)*(dxx**2 + dyy**2 + 2*dxy**2)))
        Esp = 0.5*spring_w*jnp.sum(((X - x0)[bnd]**2).sum(1)) if bnd is not None else 0.0
        return Es, Eb, Esp
    def U(x, th0):
        Es, Eb, Esp = parts(x, th0); return Es + Eb + Esp
    vg, Uj = jax.jit(jax.value_and_grad(U)), jax.jit(U)
    return (vg, Uj, jax.jit(parts)) if return_parts else (vg, Uj)          # parts = (stretch, bend, springs), for the tests

def rest_tensor(tri, sg, kx, ky, scale=1.0):
    """Rest curvature tensor S0 per triangle for bending='plate': diag(kx, ky) x scale for digit 1, the negative for digit 2 (no averaging needed at a knit/purl interface: each triangle has its own)."""
    return np.einsum('t,ab->tab', scale*np.asarray(sg, float), np.diag([kx, ky]))

def _hinge_angle(xi, xj, xk, xl):
    """Dihedral angle of one hinge, the same expression as in make_energy (for the element Hessians of make_hessian)."""
    ev = xj - xi; eh = ev/jnp.sqrt(ev @ ev + 1e-30); n1 = jnp.cross(xj - xi, xk - xi); n2 = jnp.cross(xi - xj, xl - xj)
    n1 = n1/jnp.sqrt(n1 @ n1 + 1e-30); n2 = n2/jnp.sqrt(n2 @ n2 + 1e-30); return jnp.arctan2(-(jnp.cross(n1, n2) @ eh), n1 @ n2)

def make_hessian(X0, tri, hin, A0, a0, ell, h, Y, nu, spring_w, bnd, bending='hinge'):
    """Exact sparse Hessian of the energy of make_energy (solver='newton', added 2026-11-27). The energy is a sum of element terms: membrane per triangle (3 vertices), bending per hinge (4 vertices,
    'hinge') or per interior triangle (its 3 vertices + the 3 vertices across its edges, 'plate'), springs per boundary vertex. Each element Hessian is taken by JAX (vmap of jax.hessian) and added into one
    CSR matrix whose pattern is built once. Returns hess(x, th0) -> scipy.sparse.csr_matrix (3n, 3n). Checked against the JAX Hessian-vector product of the full energy in tests/test_shell_newton.py."""
    import scipy.sparse as sp
    a0inv = np.linalg.inv(a0); D = Y*h**3/(12*(1 - nu**2)); kb = D/2.0; c1 = Y*nu/(1 - nu**2); c2 = Y/(1 + nu); n = 3*len(X0)
    i, j, k, l, ta, tb = hin.T; wbend = 3*ell**2/(A0[ta] + A0[tb])
    def mem(P, ai, a0t, At):
        e1 = P[1] - P[0]; e2 = P[2] - P[0]; a = jnp.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]]); Xm = ai @ (a - a0t)
        tr = Xm[0, 0] + Xm[1, 1]; return 0.5*At*(h/4.0)*(c1*tr**2 + c2*jnp.sum(Xm*Xm.T))
    sets = [(tri, jax.jit(jax.vmap(jax.hessian(mem))), (jnp.array(a0inv), jnp.array(a0), jnp.array(A0)), False)]
    if bending == 'hinge':
        def hb(P, w, t0): return kb*w*(_hinge_angle(P[0], P[1], P[2], P[3]) - t0)**2
        sets.append((hin[:, :4], jax.jit(jax.vmap(jax.hessian(hb))), (jnp.array(wbend),), True))
    else:
        Hh, Ginv, wt = plate_data(X0, tri, hin, A0); it = np.where(wt > 0)[0]; V6 = np.zeros((len(it), 6), int); loc = np.zeros((len(it), 3, 4), int)
        for r_, t in enumerate(it):
            vs = list(tri[t])
            for s_ in range(3):
                for v in hin[Hh[t, s_], :4]:
                    if v not in vs: vs.append(v)
            assert len(vs) == 6, 'plate element of triangle %d has %d vertices' % (t, len(vs))
            V6[r_] = vs; loc[r_] = [[vs.index(v) for v in hin[Hh[t, s_], :4]] for s_ in range(3)]
        def pb(P, lc, Gi, Aw_, t0):
            th = jnp.stack([_hinge_angle(P[lc[s_, 0]], P[lc[s_, 1]], P[lc[s_, 2]], P[lc[s_, 3]]) for s_ in range(3)]); kv = Gi @ th
            dxx, dyy, dxy = kv[0] - t0[0, 0], kv[1] - t0[1, 1], kv[2] - t0[0, 1]; return Aw_*0.5*D*(nu*(dxx + dyy)**2 + (1 - nu)*(dxx**2 + dyy**2 + 2*dxy**2))
        sets.append((V6, jax.jit(jax.vmap(jax.hessian(pb))), (jnp.array(loc), jnp.array(Ginv[it]), jnp.array((A0*wt)[it])), it))
    rows, cols = [], []
    for V, _, _, _ in sets:
        dof = (3*V[:, :, None] + np.arange(3)).reshape(len(V), -1); m = dof.shape[1]
        rows.append(np.repeat(dof, m, axis=1).ravel()); cols.append(np.tile(dof, (1, m)).ravel())
    sdiag = np.zeros(n)
    if bnd is not None and spring_w > 0: sdiag[(3*np.asarray(bnd)[:, None] + np.arange(3)).ravel()] = spring_w
    rows.append(np.arange(n)); cols.append(np.arange(n)); lin = np.concatenate(rows).astype(np.int64)*n + np.concatenate(cols)
    uniq, inv = np.unique(lin, return_inverse=True); del lin, rows, cols
    r_u = (uniq // n).astype(np.int32); c_u = (uniq % n).astype(np.int32); indptr = np.r_[0, np.cumsum(np.bincount(r_u, minlength=n))]
    def hess(x, th0):
        X = jnp.asarray(x).reshape(-1, 3); parts = []
        for V, f, args, kind in sets:
            P = X[V]
            if kind is False: Hl = f(P, *args)
            elif kind is True: Hl = f(P, *args, jnp.asarray(th0))
            else: Hl = f(P, *args, jnp.asarray(th0)[kind])
            parts.append(np.asarray(Hl, float).ravel())
        parts.append(sdiag); data = np.bincount(inv, weights=np.concatenate(parts), minlength=len(uniq))
        return sp.csr_matrix((data, c_u, indptr), shape=(n, n))
    return hess

def newton_min(x, th0, vg, U, hess, xtol=1e-4, maxit=60, lam=1e-3, stats=None, verbose=False, permc='COLAMD'):
    """One growth step by Newton's method with a Levenberg-Marquardt shift (Nocedal & Wright 2006, Numerical Optimization, 2nd ed., ch. 4 and 10.3): solve (H + lam d I) p = -g with SciPy's sparse LU
    (d = mean |diagonal of H|), accept p if the energy does not rise, then lam / 10 (more Newton-like), else lam x 10 and try again. The shift handles the rigid-body null modes and the indefinite H near a
    buckling point. CONVERGED when an accepted step at lam <= 1e-6 moves no vertex by more than xtol (mm): such a step is the Newton estimate of the remaining distance to the equilibrium, so it is a test
    on the SHAPE, not on the gradient (a small gradient did not mean converged: the L-BFGS runs of 2026-11-26 had max |grad| ~1e-3 while the sheet still moved by 20-34 mm). xtol 1e-4 mm (about 1e-4 of
    a stitch), the start lam and the factors 10 are UNSOURCED numerical choices. Returns (x, U, gradient, iterations, status: 0 converged, 1 iteration cap, 2 no descent step found, lam);
    stats (a list) receives one dict per accepted step. permc: SuperLU column ordering; COLAMD measured 2026-11-27 at ~13 s per factorisation of S000004 (41 000 unknowns), MMD_AT_PLUS_A ~170 s."""
    from scipy.sparse.linalg import splu
    import scipy.sparse as sp, time
    f, g = (np.asarray(v, float) for v in vg(x, th0)); status = 1; nit = 0; lam_min = 1e-9
    for nit in range(1, maxit + 1):
        Hs = hess(x, th0); d = float(np.abs(Hs.diagonal()).mean()); I = sp.identity(len(x), format='csr'); ok = False
        nfac = 0; t0 = time.time()
        while lam < 1e10:
            nfac += 1
            try: p = -splu((Hs + (lam*d)*I).tocsc(), permc_spec=permc).solve(g)
            except RuntimeError: lam *= 10; continue
            if not np.all(np.isfinite(p)) or g @ p >= 0: lam *= 10; continue
            step = float(np.sqrt((p.reshape(-1, 3)**2).sum(1)).max()); fn = float(U(x + p, th0)); pred = -(g @ p + 0.5*(p @ (Hs @ p)))
            if np.isfinite(fn) and (fn <= f + 1e-13*abs(f) or step < xtol): ok = True; break
            lam *= 10
        if not ok: status = 2; break
        rho = (f - fn)/pred if pred > 0 else 0.0; x = x + p; lam_used = lam
        lam = max(lam/10, lam_min) if rho > 0.75 else (lam if rho > 0.25 else lam*4)          # Levenberg-Marquardt update by actual / predicted reduction (Nocedal & Wright 2006, Alg. 4.1)
        f, g = (np.asarray(v, float) for v in vg(x, th0))
        if stats is not None: stats.append(dict(it=nit, U=f, step_mm=step, lam=lam_used, rho=rho, factorisations=nfac, seconds=round(time.time() - t0, 2), grad=float(np.abs(g).max())))
        if verbose: print('    newton %d: U %.10g, step %.3g mm, lam %.1e, rho %.3f, %d factorisations, %.1f s' % (nit, f, step, lam_used, rho, nfac, time.time() - t0), flush=True)
        if step < xtol and lam_used <= 1e-6: status = 0; break
    return x, f, g, nit, status, lam

def make_fire(vg, n, chunk=200):
    """FIRE relaxation (Bitzek, Koskinen, Gaehler, Moseler, Gumbsch 2006, Phys. Rev. Lett. 97, 170201), with the corrections of FIRE 2.0 (Guenole et al. 2020, Comput. Mater. Sci. 175, 109584):
    semi-implicit Euler, unit masses, on uphill power (P = F . v <= 0) half a step back, v = 0, dt x f_dec, alpha reset. Parameters N_min 5, f_inc 1.1, f_dec 0.5, alpha_start 0.1, f_alpha 0.99
    are the published defaults. Runs `chunk` steps per jitted call. Returns run(x, v, dt, alpha, npos, th0, dtmax) -> the same state after chunk steps."""
    def body(c, _):
        x, v, dt, a, npos, th0, dtmax = c; F = -vg(x, th0)[1]; P = F @ v
        nv = jnp.sqrt(v @ v); nF = jnp.sqrt(F @ F) + 1e-300
        up = P > 0
        v = jnp.where(up, (1 - a)*v + a*nv*F/nF, 0.0); x = jnp.where(up, x, x - 0.5*dt*c[1])
        npos = jnp.where(up, npos + 1, 0); grow = up & (npos > 5)
        dt = jnp.where(grow, jnp.minimum(dt*1.1, dtmax), jnp.where(up, dt, dt*0.5)); a = jnp.where(grow, a*0.99, jnp.where(up, a, 0.1))
        v = v + dt*F; x = x + dt*v
        return (x, v, dt, a, npos, th0, dtmax), None
    @jax.jit
    def run(x, v, dt, a, npos, th0, dtmax):
        return jax.lax.scan(body, (x, v, dt, a, npos, th0, dtmax), None, length=chunk)[0]
    return run

_FIRE_DT_SCALE = [1.0]          # 2026-11-28: a dt_max halving after a blow-up also holds for the later growth steps of the same fold() (ECC review: it was a local)
def fire_min(x, th0, vg, run, dtmax, maxit, gtol=1e-9, chunk=200):
    """One growth step by FIRE (make_fire), at most maxit steps, stopped early when max |grad| < gtol (checked every chunk). Returns (x, U, g, steps)."""
    dtmax = dtmax*_FIRE_DT_SCALE[0]
    xj = jnp.asarray(x); v = jnp.zeros_like(xj); dt = jnp.asarray(0.1*dtmax); a = jnp.asarray(0.1); npos = jnp.asarray(0); th = jnp.asarray(th0); it = 0; backs = 0
    while it < maxit:
        good = (xj, v, dt, a, npos)
        xj, v, dt, a, npos, _, _ = run(xj, v, dt, a, npos, th, jnp.asarray(dtmax)); it += chunk
        if not bool(jnp.isfinite(xj).all()):          # 2026-11-28 (ECC review): dt_max comes from the flat start and can be too large on the folded sheet: roll back the chunk,
            if backs >= 6: xj = good[0]; print('  FIRE: still non-finite after 6 halvings of dt_max: step ended at the last finite state', flush=True); break          # halve dt_max, carry on
            xj, v, dt, a, npos = good[0], jnp.zeros_like(good[0]), good[2]*0.5, jnp.asarray(0.1), jnp.asarray(0); dtmax *= 0.5; backs += 1; _FIRE_DT_SCALE[0] *= 0.5
            print('  FIRE: non-finite state, chunk rolled back, dt_max halved to %.3g' % dtmax, flush=True); continue          # (a run that never blows up is bit-identical to before)
        if float(jnp.abs(vg(xj, th)[1]).max()) < gtol: break
    f, g = (np.asarray(q, float) for q in vg(xj, th)); return np.asarray(xj, float), float(f), g, it

def stability(x, th0, hess, k=10):
    """Lowest k eigenvalues of the Hessian at the final state (shift-invert Lanczos with one sparse LU, scipy eigsh). A free sheet has 6 zero modes (rigid motion); a clearly negative eigenvalue
    means a saddle, not a minimum. Returns (eigenvalues sorted, mean |diagonal| for scale)."""
    from scipy.sparse.linalg import eigsh
    Hs = hess(x, th0); d = float(np.abs(Hs.diagonal()).mean())
    w = eigsh(Hs.tocsc(), k=k, sigma=-1e-6*d, which='LM', return_eigenvectors=False); return np.sort(w), d

def fold(M, W, C, kappa=1.0, h=0.5, nu=0.4, Y=1.0, sub=1, ntile=(1, 1), steps=20, maxiter=400, springs=0.0, seed=0, verbose=True, kx=None, ky=None, bending='hinge', snapshots=None,
         solver='lbfgs', xtol=1e-4, stats=None, warmup=200, newton_its=3, check_stability=False):
    """kx, ky: natural curvatures of a digit-1 stitch (1/length, > 0 bends toward +z); digit 2 gets the negatives. None = the paper's (-kappa, +kappa).
    bending: 'hinge' (default, bit-identical to every earlier run) or 'plate' (see make_energy). log rows: (growth s, U, iterations, z-range, largest |gradient| at the end of the step, L-BFGS status);
    the last two were added 2026-11-26 (task 32: a step that ends at the iteration cap is not converged; the gradient says how far from it).
    snapshots: None (default) or a list that receives a copy of the positions (n, 3) after every growth step: the shell's own path of equilibria at curvature fraction s (used by shell_fold.py, task 52).
    solver: 'lbfgs' (default, bit-identical to every earlier run; maxiter = L-BFGS iterations per step) or 'newton' (newton_min, added 2026-11-27; maxiter = Newton iterations per step, xtol its shape
    tolerance in mm); with 'newton' the log column 'iterations' counts Newton steps and status is 0 converged, 1 cap, 2 no descent step; stats (a list) receives (s, the per-step dicts of newton_min).
    warmup (newton only): L-BFGS iterations before Newton in every growth step (200, UNSOURCED): measured 2026-11-27 on S000004, one Newton iteration costs ~15 s (Hessian 2.4 s, sparse LU 12.9 s for
    41 000 unknowns) against ~0.06 s per L-BFGS iteration, and Newton alone from the flat (buckling) start needed many rejected, re-factorised steps.
    fire (added 2026-11-27): 'fire' = FIRE only (maxiter FIRE steps per growth step, fire_min); 'fire+newton' = FIRE then at most newton_its Newton steps (newton_min, shape test xtol): first-order
    for the long rolling path, Newton for the last digits. FIRE time step: dt_max = 1.5/sqrt(lambda_max) (lambda_max of the Hessian at full curvature by 30 power iterations with JAX Hessian-vector
    products; explicit stability needs dt < 2/sqrt(lambda_max); 1.5 is UNSOURCED). check_stability: after the last step, the lowest Hessian eigenvalues (stability), stored in stats as ('stability', w, d).
    """
    if solver not in ('lbfgs', 'newton', 'fire', 'fire+newton'): raise ValueError("solver must be 'lbfgs', 'newton', 'fire' or 'fire+newton'")
    kx0, ky0 = (-kappa, +kappa) if kx is None else (kx, ky)
    X0, tri, hin, sg, (nx, ny) = build(M, W, C, sub, ntile)
    A0, a0, ell = rest_data(X0, tri, hin, sg)
    bnd = None
    if springs > 0:
        gx, gy = nx*sub + 1, ny*sub + 1; idx = np.arange(len(X0)).reshape(gy, gx); bnd = np.unique(np.r_[idx[0], idx[-1], idx[:, 0], idx[:, -1]])
    vg, U = make_energy(X0, tri, hin, A0, a0, ell, h, Y, nu, springs, bnd, bending)
    hess = make_hessian(X0, tri, hin, A0, a0, ell, h, Y, nu, springs, bnd, bending) if (solver in ('newton', 'fire+newton') or check_stability) else None; lam = 1e-3
    x = X0.copy(); x[:, 2] += 1e-3*np.random.default_rng(seed).standard_normal(len(x)); x = x.ravel(); log = []
    if solver.startswith('fire'):
        _FIRE_DT_SCALE[0] = 1.0; frun = make_fire(vg, len(x)); th1 = rest_angles(X0, hin, sg, kx0, ky0, 1.0) if bending == 'hinge' else rest_tensor(tri, sg, kx0, ky0, 1.0)
        gfun = jax.jit(lambda z, t: vg(z, t)[1]); hvp = jax.jit(lambda z, t, w: jax.jvp(lambda q: gfun(q, t), (z,), (w,))[1])
        w = jnp.asarray(np.random.default_rng(1).standard_normal(len(x))); xj = jnp.asarray(x)
        for _ in range(30): w = hvp(xj, th1, w); lmax = float(jnp.sqrt(w @ w)); w = w/lmax
        dtmax = 1.5/np.sqrt(lmax)
    for s in np.linspace(1.0/steps, 1.0, steps):
        th0 = rest_angles(X0, hin, sg, kx0, ky0, s) if bending == 'hinge' else rest_tensor(tri, sg, kx0, ky0, s)
        if solver.startswith('fire'):
            x, fv, gv, nit = fire_min(x, th0, vg, frun, dtmax, maxiter); status = 1 if float(np.abs(gv).max()) >= 1e-9 else 0
            if solver == 'fire+newton':
                st = [] if stats is not None else None; x, fv, gv, nn, status, lam = newton_min(x, th0, vg, U, hess, xtol=xtol, maxit=newton_its, lam=1e-6, stats=st, verbose=verbose)
                if stats is not None: stats.append((float(s), st))
            Xc = x.reshape(-1, 3)
            if snapshots is not None: snapshots.append(Xc.copy())
            log.append((float(s), float(fv), int(nit), float(np.ptp(Xc[:, 2])), float(np.abs(gv).max()), int(status)))
            if verbose: print('  growth %.2f: U %.5g, %d its, z-range %.3f mm, max |grad| %.3g, status %d' % log[-1], flush=True)
            continue
        if solver == 'newton':
            if warmup > 0:          # L-BFGS first (cheap per iteration): leaves the buckling point of the flat start and removes the stiff in-plane error; Newton then finishes the soft rolling mode
                r = minimize(lambda z: tuple(np.asarray(v, float) for v in vg(z, th0)), x, jac=True, method='L-BFGS-B', options=dict(maxiter=warmup, maxcor=30, ftol=1e-14, gtol=1e-9)); x = r.x
            st = [] if stats is not None else None; x, fv, gv, nit, status, lam = newton_min(x, th0, vg, U, hess, xtol=xtol, maxit=maxiter, lam=lam, stats=st, verbose=verbose)
            if stats is not None: stats.append((float(s), st))
            Xc = x.reshape(-1, 3)
            if snapshots is not None: snapshots.append(Xc.copy())
            log.append((float(s), float(fv), int(nit), float(np.ptp(Xc[:, 2])), float(np.abs(gv).max()), int(status)))
            if verbose: print('  growth %.2f: U %.5g, %d its, z-range %.3f mm, max |grad| %.3g, status %d' % log[-1], flush=True)
            continue
        r = minimize(lambda z: tuple(np.asarray(v, float) for v in vg(z, th0)), x, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxcor=30, ftol=1e-14, gtol=1e-9)); x = r.x
        Xc = x.reshape(-1, 3)
        if snapshots is not None: snapshots.append(Xc.copy())
        log.append((float(s), float(r.fun), int(r.nit), float(np.ptp(Xc[:, 2])), float(np.abs(r.jac).max()), int(r.status)))
        if verbose: print('  growth %.2f: U %.5g, %d its, z-range %.3f mm, max |grad| %.3g, status %d' % log[-1], flush=True)
    if check_stability and stats is not None: w_, d_ = stability(x, th0, hess); stats.append(('stability', w_.tolist(), d_))
    return X0, x.reshape(-1, 3), tri, (nx, ny), log
