#!/usr/bin/env python3
"""relax_state.py - data structures and state machine for the baseline -> dynamic relaxation transition.
Reads a build_knit.py result folder (fab_axis.csv, meta.json). Writes state.json (atomic) beside it. Solvers are injected, not built in.

Sourced models this layout is meant for:
  * rod bending + twist energy: Bergou, Wardetzky, Robinson, Audoly, Grinspun (2008) "Discrete Elastic Rods", ACM Trans. Graphics 27(3).
  * frictional slip over a curved contact: capstan (Euler-Eytelwein) relation, T_high / T_low <= exp(mu * phi); standard mechanics.
UNSOURCED, supplied by the user, no defaults: mu, EA, EI, GJ, contact stiffness. release() refuses to start while any is None.
Slip is carried by REST LENGTH transfer between adjacent segments at a contact node, not by renumbering vertices; vertex count and
connectivity never change, so the solver mesh stays valid.
"""
import enum, json, os, time
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

class S(enum.Enum):
    LOADED = 'loaded'            # baseline read and discretised, constraints still on
    ARMED = 'armed'              # reference configuration and material parameters set, waiting for the trigger
    RELEASED = 'released'        # trigger received, rigid locks dropped, contact set built
    RELAXING = 'relaxing'        # outer loop: positions at fixed rest lengths, then slip, repeat
    SETTLED = 'settled'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
ALLOWED = {S.LOADED: {S.ARMED, S.CANCELLED}, S.ARMED: {S.RELEASED, S.CANCELLED}, S.RELEASED: {S.RELAXING, S.FAILED, S.CANCELLED},
           S.RELAXING: {S.RELAXING, S.SETTLED, S.FAILED, S.CANCELLED}, S.SETTLED: set(), S.FAILED: {S.ARMED}, S.CANCELLED: set()}

@dataclass
class Params:
    """All physical inputs. None = not supplied; nothing is assumed."""
    mu: float = None; EA: float = None; EI: float = None; GJ: float = None; k_contact: float = None
    rest_length_ratio: float = 1.0     # rest length = ratio * baseline length. 1.0 = baseline is length-relaxed. Modelling choice, not sourced.
    rest_curvature: float = 0.0        # stress-free yarn is straight. Modelling choice (heat-setting excluded by the project).
    slip_step_max: float = 0.2         # max fraction of a segment's rest length moved per outer iteration (numerical control)
    outer_max: int = 200; tol_len: float = 1e-6; tol_energy: float = 1e-8
    def missing(self): return [k for k in ('mu', 'EA', 'EI', 'GJ', 'k_contact') if getattr(self, k) is None]

@dataclass
class Yarn:
    course: int
    X: np.ndarray                      # (n,3) vertex positions
    l0: np.ndarray                     # (n-1,) rest length per segment (the migrating quantity)
    theta: np.ndarray                  # (n-1,) material-frame twist angle per segment (DER degree of freedom), 0 at start
    pinned: np.ndarray                 # (n,) bool, True while rigid locks are on

@dataclass
class Contact:
    ya: int; ia: int; yb: int; ib: int; dist: float      # vertex ia of yarn ya touches vertex ib of yarn yb
    phi_a: float = 0.0; phi_b: float = 0.0               # wrap angle (rad) of each yarn at the contact

@dataclass
class Sim:
    folder: str; R: float; yarns: list; params: Params
    state: S = S.LOADED; contacts: list = field(default_factory=list); history: list = field(default_factory=list)
    energy_log: list = field(default_factory=list)

    # ---- state machine
    def to(self, new, note=''):
        if new not in ALLOWED[self.state]: raise RuntimeError('illegal transition %s -> %s' % (self.state.value, new.value))
        self.state = new; self.history.append(dict(state=new.value, t=time.strftime('%Y-%m-%dT%H:%M:%S'), note=note)); self.save()
    def save(self):
        d = dict(state=self.state.value, history=self.history, params=self.params.__dict__, R=self.R, n_yarns=len(self.yarns),
                 n_vertices=int(sum(len(y.X) for y in self.yarns)), n_contacts=len(self.contacts), energy_log=self.energy_log[-50:])
        tmp = os.path.join(self.folder, 'state.json.tmp'); json.dump(d, open(tmp, 'w'), indent=1); os.replace(tmp, os.path.join(self.folder, 'state.json'))

def resample(P, ds):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.linspace(0, s[-1], max(3, int(round(s[-1]/ds)) + 1))
    return np.c_[[np.interp(u, s, P[:, k]) for k in range(3)]].T

def load(folder, params, ds_over_R=1.0):
    meta = json.load(open(os.path.join(folder, 'meta.json'))); R = meta['R']
    A = np.loadtxt(os.path.join(folder, 'fab_axis.csv'), delimiter=',', skiprows=1)
    yarns = []
    for c in np.unique(A[:, 0]).astype(int):
        X = resample(A[A[:, 0] == c][:, 1:], ds_over_R*R)
        yarns.append(Yarn(c, X, np.linalg.norm(np.diff(X, axis=0), axis=1)*params.rest_length_ratio, np.zeros(len(X)-1), np.ones(len(X), bool)))
    sim = Sim(folder, R, yarns, params); sim.history.append(dict(state='loaded', t=time.strftime('%Y-%m-%dT%H:%M:%S'), note=meta.get('id', ''))); sim.save()
    return sim

# ---- contacts and wrap angle
def wrap_angle(X, i, k=3):
    a, b = max(0, i-k), min(len(X)-1, i+k)
    t1, t2 = X[i]-X[a], X[b]-X[i]
    n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
    return 0.0 if n1 < 1e-12 or n2 < 1e-12 else float(np.arccos(np.clip(t1 @ t2/(n1*n2), -1, 1)))

def build_contacts(sim, reach=1.15, skip_arc=3.0):
    """Vertex pairs of different yarns, or far apart on one yarn, closer than reach*2R. Local minima of distance only (one contact per crossing)."""
    P = np.vstack([y.X for y in sim.yarns]); yid = np.concatenate([[k]*len(y.X) for k, y in enumerate(sim.yarns)])
    idx = np.concatenate([np.arange(len(y.X)) for y in sim.yarns]); arc = np.concatenate([np.r_[0, np.cumsum(y.l0)] for y in sim.yarns])
    pairs = cKDTree(P).query_pairs(reach*2*sim.R, output_type='ndarray'); best = {}
    for p, q in pairs:
        if yid[p] == yid[q] and abs(arc[p]-arc[q]) < skip_arc*2*sim.R: continue
        d = float(np.linalg.norm(P[p]-P[q])); key = (yid[p], yid[q], idx[p]//4, idx[q]//4)     # coarse cell: keep the closest pair per crossing
        if key not in best or d < best[key][4]: best[key] = (int(yid[p]), int(idx[p]), int(yid[q]), int(idx[q]), d)
    return [Contact(*v, wrap_angle(sim.yarns[v[0]].X, v[1]), wrap_angle(sim.yarns[v[2]].X, v[3])) for v in best.values()]

# ---- slip (capstan criterion, rest-length transfer)
def tension(l, l0, EA): return EA*(l/l0 - 1.0)            # linear axial law. Unsourced assumption: Hookean, EA user-supplied

def slip_at_node(la, l0a, lb, l0b, EA, mu, phi, step_max):
    """Segments a|b meet at a contact node. Returns delta moved from b's rest length to a's (negative = other way); 0 if the contact holds."""
    Ta, Tb = tension(la, l0a, EA), tension(lb, l0b, EA)
    hi_is_a = Ta >= Tb; Thi, Tlo = (Ta, Tb) if hi_is_a else (Tb, Ta)
    ratio = np.exp(mu*phi)
    if Thi <= max(Tlo, 0.0)*ratio + 1e-12 or Thi <= 0: return 0.0            # holds: T_hi <= T_lo * exp(mu phi)
    src_l0, dst_l0 = (l0b, l0a) if hi_is_a else (l0a, l0b)                    # material flows toward the HIGH-tension side
    cap = step_max*src_l0
    def gap(dl):                                                              # T_hi(dl) - ratio*max(T_lo(dl),0)
        a0, b0 = (l0a + dl, l0b - dl) if hi_is_a else (l0a - dl, l0b + dl)
        Ta2, Tb2 = tension(la, a0, EA), tension(lb, b0, EA); Thi2, Tlo2 = (Ta2, Tb2) if hi_is_a else (Tb2, Ta2)
        return Thi2 - ratio*max(Tlo2, 0.0)
    dl = cap if gap(cap) > 0 else brentq(gap, 0.0, cap)
    return dl if hi_is_a else -dl

def migrate(sim):
    """One slip pass over all contacts. Returns max |rest-length change|. Contacts are interior vertices of a yarn: segment ia-1 | ia."""
    mx = 0.0; p = sim.params
    for c in sim.contacts:
        for yi, i, phi in ((c.ya, c.ia, c.phi_a), (c.yb, c.ib, c.phi_b)):
            y = sim.yarns[yi]
            if i < 1 or i > len(y.l0)-1: continue
            la, lb = np.linalg.norm(y.X[i]-y.X[i-1]), np.linalg.norm(y.X[i+1]-y.X[i])
            d = slip_at_node(la, y.l0[i-1], lb, y.l0[i], p.EA, p.mu, phi, p.slip_step_max)
            if d: y.l0[i-1] += d; y.l0[i] -= d; mx = max(mx, abs(d))
    return mx

# ---- the trigger
def release(sim):
    sim.to(S.ARMED) if sim.state == S.LOADED else None
    miss = sim.params.missing()
    if miss: raise ValueError('parameters not supplied (no defaults are assumed): %s' % ', '.join(miss))
    for y in sim.yarns: y.pinned[:] = False                                   # rigid locks dropped
    sim.contacts = build_contacts(sim); sim.to(S.RELEASED, '%d contacts' % len(sim.contacts))

def run(sim, relax_fn, energy_fn):
    """relax_fn(sim) -> None minimises positions at fixed l0 (e.g. L-BFGS-B on the DER + contact energy); energy_fn(sim) -> float."""
    p = sim.params; sim.to(S.RELAXING, 'outer loop start'); E_prev = None
    for k in range(p.outer_max):
        relax_fn(sim); dmax = migrate(sim); E = energy_fn(sim); sim.energy_log.append(dict(iter=k, E=E, max_dl0=dmax))
        if not np.isfinite(E): return sim.to(S.FAILED, 'non-finite energy')
        if dmax < p.tol_len and E_prev is not None and abs(E - E_prev) < p.tol_energy*max(1.0, abs(E)): return sim.to(S.SETTLED, 'iter %d' % k)
        E_prev = E; sim.contacts = build_contacts(sim); sim.to(S.RELAXING, 'iter %d E %.6g dl0 %.3g' % (k, E, dmax))
    sim.to(S.FAILED, 'outer_max reached without settling')
