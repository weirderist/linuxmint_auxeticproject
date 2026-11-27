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

STRETCH_SCALE = 1.0      # multiplies EA only (axial stiffness) so stretch and bending stiffness can be chosen independently, as Sperl et al. 2022 Sec 6.2 do. 1 = solid rod.
BEND_SCALE = 1.0         # multiplies EI only. A spun yarn bends far more easily than a solid rod of its tensile modulus: measured bending/tensile stiffness ratio 0.249/79 = 0.0032 for a 16/2 acrylic spun yarn (Ding et al., arXiv 2307.12360), and Sperl et al. 2022 fit k_b over 1e-3..10 x the solid-rod value. Not measured for cotton Ne 10.
EA_FIXED = None          # measured axial stiffness (N), set by relax_job --ea or from the first point of --stretch-law. None = solid rod of modulus E times STRETCH_SCALE.
EI_FIXED = None          # measured bending stiffness (N mm^2), set by relax_job --ei. None = solid rod of modulus E times BEND_SCALE.
LAW = None               # measured stretch law (dict from relax_state.build_law) or None for Hooke. Read when the energy is traced: call set_law() after changing it.
def section(E, R):
    EA = EA_FIXED if EA_FIXED is not None else STRETCH_SCALE*E*np.pi*R**2
    EI = EI_FIXED if EI_FIXED is not None else BEND_SCALE*E*np.pi*R**4/4.0
    return EA, EI                                                                                          # EA, EI

def bend_stiffness(sim, E):
    """EA and EI of the yarn. EI is also multiplied by sim.bend_scale (1 unless the straightening factor is on: see relax_job --straight)."""
    EA, EI = section(E, sim.R); return EA, EI*getattr(sim, 'bend_scale', 1.0)

# ---------------------------------------------------------------- twist drive (2026-11-27, K: torque from the yarn twist TPI, for skewing / spirality)
# Twist itself is not a degree of freedom here (see the module note: a round yarn with free ends sheds its twist). What makes a twisted yarn skew a knit is twist LOCKED in by friction at
# the interlacings; its torque turns the plane of each loop. Modelled (our choice, UNSOURCED as a model) as a preferred TORSION tau0 of the yarn axis, acting where the axis is curved:
#   E_tor = 1/2 GJ sum_edges w (tau - tau0)^2 l,   tau = signed turn of the curvature binormal across the edge / edge length,   w = |kb_a|^2 |kb_b|^2 / (|kb_a|^2 |kb_b|^2 + TOR_EPS^4)
#   tau0 = live x 2 pi TPI / 25.4 (rad/mm) x (+1 Z twist, -1 S twist; WHICH sign gives which skew direction is NOT verified against a fabric yet)
#   GJ = TORSION_SCALE x G pi R^4 / 2 with G = E / 3 (incompressible isotropic solid; UNSOURCED for a spun yarn)
# live (0..1) is the share of the twist that still acts (steaming and relaxation history): a FOLD property, fitted, not measured. TPI = 0 or live = 0: the term is absent (results unchanged).
TORSION_SCALE = 1.0       # multiplies GJ only (as BEND_SCALE does for EI). UNSOURCED
TOR_EPS = 0.02            # curvature gate (turning angle per vertex, rad): below it the binormal, hence the torsion, is undefined and the term fades out. UNSOURCED numerical choice

def twist_args(sim, tri, E):
    """() when the twist drive is off, else (tq, GJ, tau0) for _energy: tq = pairs of consecutive bending triplets on the same yarn (one per interior edge)."""
    p = sim.params; tpi = float(getattr(p, 'tpi', 0.0) or 0.0); live = float(getattr(p, 'twist_live', 0.0) or 0.0)
    if tpi == 0.0 or live == 0.0: return ()
    tq = np.array([(j, j + 1) for j in range(len(tri) - 1) if tri[j + 1][1] == tri[j][1] + 1 and tri[j + 1][0] == tri[j][1]], int)
    if len(tq) == 0: return ()
    sign = -1.0 if str(getattr(p, 'twist_dir', 'Z')).upper() == 'S' else 1.0
    GJ = TORSION_SCALE*(E/3.0)*np.pi*sim.R**4/2.0
    return tq, GJ, sign*live*2.0*np.pi*tpi/25.4

def _torsion(X, tri, kb, tq, tau0):
    """(sum of w (tau - tau0)^2 l, per-edge tau, w): discrete torsion between consecutive curvature binormals (see the twist-drive note)."""
    ka = kb[tq[:, 0]]; kz = kb[tq[:, 1]]; t = X[tri[tq[:, 1], 1]] - X[tri[tq[:, 0], 1]]; lt = jnp.sqrt(jnp.sum(t*t, 1) + 1e-30); th = t/lt[:, None]
    na2 = jnp.sum(ka*ka, 1); nz2 = jnp.sum(kz*kz, 1); ok = na2*nz2 > 1e-24
    s = jnp.where(ok, jnp.sum(jnp.cross(ka, kz)*th, 1), 0.0); c = jnp.where(ok, jnp.sum(ka*kz, 1), 1.0)          # safe inputs: no NaN gradient where the binormal vanishes
    tau = jnp.arctan2(s, c)/lt; w = na2*nz2/(na2*nz2 + TOR_EPS**4)
    return jnp.sum(w*(tau - tau0)**2*lt), tau, w

def _energy(x, l0, seg, tri, lbar, kb0, pairs, pmask, EA, EI, kc, R2, cell, tgt, cnt, ks, tq=None, GJ=0.0, tau0=0.0):
    X = x.reshape(-1, 3)
    e = X[seg[:, 1]] - X[seg[:, 0]]; l = jnp.sqrt(jnp.sum(e*e, 1) + 1e-30)
    if LAW is None: Es = 0.5*EA*jnp.sum((l - l0)**2/l0)
    else:       # measured piecewise-linear law: force = EA x tau(strain), energy = EA l0 W(strain), W the integral of tau. A linear law (tau = strain, W = strain^2/2) is exactly the line above.
        K, T, S, W = (jnp.asarray(LAW[k]) for k in ('K', 'T', 'S', 'W')); eps = (l - l0)/l0
        k = jnp.clip(jnp.searchsorted(K, eps, side='right') - 1, 0, K.shape[0] - 1); d = eps - K[k]
        Es = EA*jnp.sum(l0*(W[k] + T[k]*d + 0.5*S[k]*d*d))
    e1 = X[tri[:, 1]] - X[tri[:, 0]]; e2 = X[tri[:, 2]] - X[tri[:, 1]]
    n1 = jnp.sqrt(jnp.sum(e1*e1, 1) + 1e-30); n2 = jnp.sqrt(jnp.sum(e2*e2, 1) + 1e-30)
    kb = 2.0*jnp.cross(e1, e2)/(n1*n2 + jnp.sum(e1*e2, 1) + 1e-12)[:, None]
    kd = kb - kb0; Eb = 0.5*EI*jnp.sum(jnp.sum(kd*kd, 1)/lbar)
    d = jnp.sqrt(jnp.sum((X[pairs[:, 0]] - X[pairs[:, 1]])**2, 1) + 1e-30)
    Ec = 0.5*kc*jnp.sum(pmask*jnp.maximum(0.0, R2 - d)**2)
    mean = jax.ops.segment_sum(X, cell, num_segments=tgt.shape[0])/cnt[:, None]          # mean position of the vertices in each stitch cell
    Esh = 0.5*ks*jnp.sum(cnt*jnp.sum((mean - tgt)**2, 1))                                  # hold on the shell fold (ks = 0 when no shell)
    if tq is None: return Es + Eb + Ec + Esh          # twist drive off: exactly the energy used before 2026-11-27
    return Es + Eb + Ec + Esh + 0.5*GJ*_torsion(X, tri, kb, tq, tau0)[0]
_vg = jax.jit(jax.value_and_grad(_energy))
MACRO = False   # relax_job --macro: give the optimiser the three soft large-scale bending modes as variables of their own (see _warp and relax)
def _warp(k3, X, c):
    """Smooth large-scale map of the fabric, parametrised by three numbers k3 = (kx, ky, t): bend into a cylinder along x (curvature kx, 1/length), then along y (ky), then add a twist z += t*(x-cx)*(y-cy).
    Identity for k3 = 0. Lengths along the mid-surface are kept; a vertex at height z above the centre plane moves with the factor (1 - z k). Written with sinc so that k = 0 is smooth for the gradient."""
    P = X
    for ax, k in ((0, k3[0]), (1, k3[1])):
        s = P[:, ax] - c[ax]; z = P[:, 2] - c[2]; a = k*s
        u = s*jnp.sinc(a/jnp.pi) - z*jnp.sin(a); w = 0.5*k*s*s*jnp.sinc(a/(2*jnp.pi))**2 + z*jnp.cos(a)
        P = jnp.stack([c[0] + u if ax == 0 else P[:, 0], c[1] + u if ax == 1 else P[:, 1], c[2] + w], axis=1)
    return jnp.stack([P[:, 0], P[:, 1], P[:, 2] + k3[2]*(P[:, 0] - c[0])*(P[:, 1] - c[1])], axis=1)
def _energy_macro(v, c, sc, *args):
    """Same energy, in the variables v = (q1, q2, q3, local positions): positions = _warp(q*sc, local). A change of variables (a bijection for every q), so the minimum is the same; it only lets the optimiser reach the soft bending modes directly."""
    return _energy(_warp(v[:3]*sc, v[3:].reshape(-1, 3), c).ravel(), *args)
_vgm = jax.jit(jax.value_and_grad(_energy_macro))
def set_law(law):
    """Install a stretch law (None = Hooke) and rebuild the compiled energy: jax.jit reads module globals only when it first traces, so changing LAW without this has no effect."""
    global LAW, _vg
    global _vgm
    LAW = law; _vg = jax.jit(jax.value_and_grad(_energy)); _vgm = jax.jit(jax.value_and_grad(_energy_macro))
    _FIRE.clear()          # the FIRE functions trace _vg: rebuild them with the new law (review 2026-11-27)

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
    EA, EI = bend_stiffness(sim, E); off, seg, tri, l0, lbar = pack(sim); pairs, pm, _ = near_pairs(sim, off)
    x = np.vstack([y.X for y in sim.yarns]).ravel(); kb0 = baseline_curvature(sim, tri); sa, s0 = shell_args(sim), shell_args(sim, False)
    ta = twist_args(sim, tri, E); tot = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, kc, 2*sim.R, *sa, *ta))
    st = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, EA, 0.0, 0.0, 0.0, *s0)); bd = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, 0.0, EI, 0.0, 0.0, *s0))
    sh = float(_energy(x, l0, seg, tri, lbar, kb0, pairs, pm, 0.0, 0.0, 0.0, 0.0, *sa))
    tw = float(0.5*ta[1]*_torsion(jnp.asarray(x).reshape(-1, 3), tri, _kb_np(sim, tri)[0], ta[0], ta[2])[0]) if ta else 0.0
    return dict(total=tot, stretch=st, bend=bd, shell=sh, torsion=tw, contact=tot - st - bd - sh - tw)

# ---------------------------------------------------------------- FIRE (2026-11-27, K approved; Bitzek et al. 2006 PRL 97 170201 with the FIRE 2.0 corrections of Guenole et al. 2020 Comput. Mater. Sci. 175 109584)
# The same minimiser as shell_sim.make_fire, applied to the yarn energy _energy. The whole loop runs inside JAX (lax.scan, jitted once per array shape), so it runs on a GPU without a host
# round trip per iteration; SciPy L-BFGS-B runs on the CPU and needs one host call per iteration. Used when SOLVER == 'fire' (relax_job --solver fire); L-BFGS stays the default and is unchanged.
SOLVER = 'lbfgs'          # 'lbfgs' (default) or 'fire'
FIRE_CHUNK = 200          # steps per jitted call (UNSOURCED numerical choice)
FIRE_MAXIT = 20000         # step cap per relaxation round (UNSOURCED)
FIRE_REBUILD = 1000        # FIRE steps between rebuilds of the contact pair list (UNSOURCED; review 2026-11-27: one list for a whole 20000-step solve let yarns pass through each other)
FIRE_GTOL_REL = 1e-6      # converged when max |grad| < FIRE_GTOL_REL x EA (dimensionless strain 1e-6; UNSOURCED)
_FIRE = {}

def _fire_chunk_impl(x, v, dt, a, npos, dtmax, *args):
    def body(c, _):
        x, v, dt, a, npos, dtmax = c
        E, g = _vg(x, *args); F = -g; P = F @ v
        nv = jnp.sqrt(v @ v); nF = jnp.sqrt(F @ F) + 1e-300
        up = P > 0
        v = jnp.where(up, (1 - a)*v + a*nv*F/nF, 0.0); x = jnp.where(up, x, x - 0.5*dt*c[1])          # uphill: half step back and stop (FIRE)
        npos = jnp.where(up, npos + 1, 0); grow = up & (npos > 5)
        dt = jnp.where(grow, jnp.minimum(dt*1.1, dtmax), jnp.where(up, dt, dt*0.5)); a = jnp.where(grow, a*0.99, jnp.where(up, a, 0.1))
        v = v + dt*F; x = x + dt*v
        return (x, v, dt, a, npos, dtmax), E
    (x, v, dt, a, npos, dtmax), Es = jax.lax.scan(body, (x, v, dt, a, npos, dtmax), None, length=FIRE_CHUNK)
    return x, v, dt, a, npos, Es[-1]

def _fire_fns():
    if 'run' not in _FIRE:
        _FIRE['run'] = jax.jit(_fire_chunk_impl)
        _FIRE['gfun'] = jax.jit(lambda x, *a: _vg(x, *a)[1])
        _FIRE['hvp'] = jax.jit(lambda x, w, *a: jax.jvp(lambda q: _FIRE['gfun'](q, *a), (x,), (w,))[1])
    return _FIRE

def fire_min(x0, args, maxit=20000, gtol=None, seed=1):
    """Minimise the yarn energy from x0 with FIRE. Returns (x, E, max|grad|, steps, converged). The time step limit dt_max = 1.5/sqrt(lambda_max), lambda_max of the Hessian by 30 power
    iterations at x0 (explicit stability needs dt < 2/sqrt(lambda_max); 1.5 is UNSOURCED). Steps are counted in multiples of FIRE_CHUNK."""
    F = _fire_fns(); args = tuple(jnp.asarray(a) for a in args); x = jnp.asarray(x0, float)
    w = jnp.asarray(np.random.default_rng(seed).standard_normal(x.shape[0]))
    for _ in range(30):
        w = F['hvp'](x, w, *args); lmax = float(jnp.sqrt(w @ w)); w = w/lmax
    dtmax = 1.5/np.sqrt(max(lmax, 1e-30)); EA = float(np.asarray(args[7])) if len(args) > 7 else 1.0
    if gtol is None: gtol = FIRE_GTOL_REL*max(EA, 1e-30)
    v = jnp.zeros_like(x); dt = jnp.asarray(0.1*dtmax); a = jnp.asarray(0.1); npos = jnp.asarray(0); dmax = jnp.asarray(dtmax); steps = 0; conv = False
    while steps < maxit:
        x, v, dt, a, npos, E = F['run'](x, v, dt, a, npos, dmax, *args); steps += FIRE_CHUNK
        g = np.asarray(F['gfun'](x, *args), float)
        if np.abs(g).max() < gtol: conv = True; break
    g = np.asarray(F['gfun'](x, *args), float)
    return np.asarray(x, float), float(_vg(x, *args)[0]), float(np.abs(g).max()), steps, conv

def relax(sim, E, kc, maxiter=300, rounds=4, verbose=False):
    """Minimise positions at fixed rest lengths. Contact pairs are rebuilt between rounds (positions move)."""
    EA, EI = bend_stiffness(sim, E)
    for r in range(rounds):
        off, seg, tri, l0, lbar = pack(sim); pairs, pm, npair = near_pairs(sim, off); kb0 = baseline_curvature(sim, tri); sa = shell_args(sim)
        ta = twist_args(sim, tri, E); x0 = np.vstack([y.X for y in sim.yarns]).ravel(); args = (l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, kc, 2*sim.R, *sa, *ta)
        if MACRO:       # variables: three scaled bending modes (unit = the edge of the sheet moves by about one yarn radius) + the positions, re-referenced to the current shape every round
            P = x0.reshape(-1, 3); c = 0.5*(P.max(0) + P.min(0)); L = np.maximum(P.max(0) - P.min(0), 10*sim.R); sc = np.array([8*sim.R/L[0]**2, 8*sim.R/L[1]**2, 4*sim.R/(L[0]*L[1])])
            f = lambda v: tuple(np.asarray(t, float) for t in _vgm(v, c, sc, *args)); res = minimize(f, np.r_[np.zeros(3), x0], jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxcor=30, ftol=1e-13, gtol=1e-9))
            k3 = res.x[:3]*sc; X = np.asarray(_warp(k3, res.x[3:].reshape(-1, 3), c), float); sim.macro_k = getattr(sim, 'macro_k', np.zeros(3)) + k3
        else:
            if SOLVER == 'fire':          # FIRE needs thousands of steps (the L-BFGS maxiter does not apply); the contact list is rebuilt every FIRE_REBUILD steps
                xf, done, conv, gtol = x0, 0, False, FIRE_GTOL_REL*max(EA, 1e-30)
                while done < FIRE_MAXIT:
                    xf, Ef, gmax, steps, conv = fire_min(xf, args, maxit=min(FIRE_REBUILD, FIRE_MAXIT - done), gtol=gtol); done += steps
                    for y, o in zip(sim.yarns, off): y.X = xf.reshape(-1, 3)[o:o+len(y.X)].copy()
                    pairs, pm, npair = near_pairs(sim, off); args = (l0, seg, tri, lbar, kb0, pairs, pm, EA, EI, kc, 2*sim.R, *sa, *ta)
                    gmax = float(np.abs(np.asarray(_fire_fns()['gfun'](jnp.asarray(xf), *(jnp.asarray(a) for a in args)))).max()); Ef = float(_vg(xf, *args)[0])
                    if gmax < gtol: conv = True; break          # converged WITH the rebuilt contact list
                    conv = False
                res = type('R', (), dict(x=xf, fun=Ef, nit=done, message=('FIRE converged' if conv else 'FIRE step cap'), gmax=gmax, conv=conv))(); X = xf.reshape(-1, 3)
            else:
                f = lambda x: tuple(np.asarray(v, float) for v in _vg(x, *args))
                res = minimize(f, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxcor=30, ftol=1e-13, gtol=1e-9)); X = res.x.reshape(-1, 3)
        for y, o in zip(sim.yarns, off): y.X = X[o:o+len(y.X)].copy()
        if verbose: print('   relax round %d: %d pairs, E %.6g, %d its, %s' % (r+1, npair, res.fun, res.nit, res.message if isinstance(res.message, str) else res.message.decode()) + ('' if not MACRO else '   macro bend kx %+.4g ky %+.4g twist %+.4g (added this round)' % tuple(k3)), flush=True)
        if (getattr(res, 'conv', False) if (SOLVER == 'fire' and not MACRO) else res.nit < maxiter): break          # FIRE: stop when converged (review 2026-11-27); L-BFGS: as before

def min_axis_distance(sim, skip=6):
    P = np.vstack([y.X for y in sim.yarns]); yid = np.concatenate([[k]*len(y.X) for k, y in enumerate(sim.yarns)]); ix = np.concatenate([np.arange(len(y.X)) for y in sim.yarns])
    pr = cKDTree(P).query_pairs(2*sim.R*1.5, output_type='ndarray')
    pr = pr[(yid[pr[:, 0]] != yid[pr[:, 1]]) | (np.abs(ix[pr[:, 0]] - ix[pr[:, 1]]) > skip)]
    return float(np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1).min()) if len(pr) else float('inf')