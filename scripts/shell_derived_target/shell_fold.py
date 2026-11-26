#!/usr/bin/env python3
"""shell_fold.py - the SHELL-DERIVED TARGET FABRIC by a PROGRESSIVE FOLD (K's plan of 23-24 Nov 2026, task 47: results first, replicating the results with parameters next).
This replaces the one-jump placement of shell_target.py + relax_job.py. Nothing outside --out is written; the shell run is only read. relax_job.py, relax_solver.py, relax_state.py, loop_grade.py,
shell_sim.py and the frozen copy of shell1 are NOT modified: this driver imports them (relax_job's numpy helpers shell_field, shell_targets, rotate_kb0, yarn_inputs; shell_target's strain rule).

Method
  1. start     the yarn mesh of the shell run's own pattern is built flat (shell_target.build, same CPI, WPI, tiles, Ne); rest lengths = the normal start state of relax_job (feeder-tension ratio, optional
               loop_grade.apply_start_grade); the stress-free curvature kb0 of the flat state is kept; the usual out-of-plane perturbation is added once.
  2. stages    N stages, fold fraction t = s/N, s = 1..N. At stage s every stitch cell has a TARGET fraction (t, or its frozen value). The shell at that fraction is the shell's OWN equilibrium at curvature fraction t
               (task 52, 2026-11-26: the growth steps of the same shell solve, recorded once and cached as shell_path.npz; between two steps the sheets and F are interpolated; --linear-path restores the
               old straight line X_t = flat + t (X - flat), which invented compression where the sheet turns). Per stage: (a) rest lengths are recomputed from the strain of the shell's OWN triangles at that fraction, F_t of the path with t of the cell (shell_target.segment_stretch, course_factors:
               tension sites give yarn away, compression sites collect it, sign -1 as in shell_target; the factors are applied to the START rest lengths, so every course keeps its total at every stage);
               (b) the per-cell targets of relax_job (cell mean of the yarn vertices held toward the sheet, yarn offset along the sheet normal) are rebuilt on X_t, and kb0 is rotated into the local frame of X_t;
               (c) a short L-BFGS relaxation (relax_solver.relax, contact ON, strong shell hold) is run; (d) blocked cells are frozen (below).
  3. freezing  After the relaxation of a stage, a cell is BLOCKED when the closest distance between yarn axes in the cell is below blocked_frac x 2R (the yarn diameter; UNSOURCED fraction). A blocked cell
               freezes its target at the last fraction at which it was clear (nothing is reverted; the next relaxations pull it back toward that target). Frozen cells stay frozen; the others continue to t = 1.
               Where a frozen cell shares a shell grid vertex with a free neighbour, the vertex takes the smaller fraction (UNSOURCED, conservative).
  4. final     the normal relaxation: outer rounds of L-BFGS-B (relax_job's loop: rs.run, capstan slip between rounds), 40 x 200 by default, with the usual weak hold (0.05 x EA/R), targets at the final
               fractions (frozen cells stay at their frozen fraction), rest lengths recomputed once for the final fractions. No creep (it would erase the redistribution; as in shell_target).
No shell is discarded; a self-intersecting shell (shell_sim has no contact term) is used exactly as it is. The yarn is folded as far as contact allows; the shell is a partial inspiration, not a constraint.

Outputs in --out (besides the relax_job-style files meta.json, fab_axis.csv, shell.npz, state.json, relaxed_axis.csv, relaxed_l0.csv, relaxed_tube.obj)
  fold_fraction.csv        per stitch cell: cell_x, cell_y, fraction (fold reached), frozen, frozen_at_stage, dmin_final_over_2R, blocked_at_end   <- the self-intersection detection result, per cell
  fold_summary.json        schema shell_fold_summary_v1: share of cells frozen, minimum / mean fraction, share blocked at the end, stage records, settings, UNSOURCED list. `--aggregate DIR...` prints one CSV
                           line per run folder (for the end-of-campaign pass over all runs).
  fold_cell_metrics.csv    per cell, before (flat start) versus after: rest length of the cell, course spacing, wale spacing, the shell target spacings, the shell strain along courses and wales, and the
                           per-relation sign flags (1 agree, 0 disagree, empty = below the threshold).   fold_check.json: correlations and sign-agreement shares of those relations.
  fold_stage_log.csv       per stage: fold fraction, cells frozen, energy, shell deviation, closest axis distance, seconds.   fold_checkpoint.npz: resume point (after every stage and every outer round).

Required physical behaviour, and what this driver can and cannot show (K, 24 Nov 2026)
  Yarn is one polyline per course, so yarn flows only ALONG a course: the redistribution can set the loop length (c) only. The course spacing (a) (distance between neighbouring courses) and the wale
  spacing (b) (between neighbouring wales) are not set by it; they follow from the shell hold on the cell means and from the loop geometry in the relaxation. They are MEASURED here per stitch cell
  from the cell-mean positions of the yarn vertices (course spacing = distance between the means of cells (j, i) and (j +- 1, i), wale spacing between (j, i) and (j, i +- 1)), and compared with
  R1 loop length change against the shell strain along the course minus the course mean (rule: sign x ...; sign -1 = tension sites smaller loops),
  R2 spacing change against loop length change (real knitting: more yarn in an area, looser area; expected + for both spacings),
  R3 spacing change against the shell strain (geometry: stretch along the wales lengthens the course spacing, stretch along the courses the wale spacing; expected +).
  CAUTION (not forced, not hidden): with sign -1, R1 expects tension sites to hold LESS yarn while R3 expects them to be farther apart; together they make R2 negative, i.e. R1, R2 and R3 cannot all
  agree. With sign +1 all three can. Also, the cell means are held toward the shell, so R3 is largely imposed by the hold and is not independent evidence. If R2 or R3 fail on a real run, the driver
  reports it (fold_check.json); a proposed change is to scale the target cell spacing by the local loop-length factor (target spacing x f^k) or to use sign +1. Neither is applied here.

UNSOURCED (every value below is a guess or a numerical choice, not from the literature; also recorded in fold_summary.json)
  stages N = 10; stage shell hold 0.5 x EA/R (strong); final shell hold 0.05 x EA/R (the 13 Nov 2026 success setting recorded in CLAUDE.md); stage relaxation 100 iterations x 1 round;
  blocked_frac = 0.8 of the yarn diameter (axis distance, cells of one stitch); fold path = the shell's own growth equilibria, linear interpolation between two of them; smaller fraction at shared shell vertices; normals of the interpolated sheet;
  segment-to-cell assignment by the flat midpoint; spacing from cell means; sign agreement thresholds 1e-3; gain and sign of the strain rule (as in shell_target); ref-scale 0 (shell_target's default;
  relax_job advises 1 with a shell: open parameter); kb0 rotated into the frame of the sheet at every stage (an adaptation of Sperl, Narain, Wojtan 2021, as in relax_job); no revert on freezing.

Usage
  python3 shell_fold.py --run campaigns/shell_runs/shell1/runs/S000001 --out folds/S000001 [--stages 10] [--blocked-frac 0.8] [--sign -1] [--gain 1] [--ref-scale 0] [--stages-only]
  Resume after an interruption: the same command with --resume.  Aggregate: python3 shell_fold.py --aggregate folds/S000001 folds/S000002 ...
"""
import argparse, csv, json, math, os, shutil, sys, time, warnings
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_S = os.path.dirname(HERE)
if os.path.basename(_S) == 'scripts': sys.path.extend(os.path.join(_S, _d) for _d in sorted(os.listdir(_S)) if _d not in ('legacy', '__pycache__') and os.path.isdir(os.path.join(_S, _d)))          # the other folders of scripts/ (not in a flat frozen copy)
import shell_target as st

SCHEMA = 'shell_fold_summary_v1'
EPS = 1e-3                                        # UNSOURCED: a change or strain below this is not counted in a sign-agreement share
DEFAULT_STAGES, STAGE_SHELL_W, FINAL_SHELL_W, STAGE_ITER, STAGE_ROUNDS, BLOCKED_FRAC = 10, 0.5, 0.05, 100, 1, 0.8
RELAX_DEFAULTS = dict(E=500.0, mu=0.49, k_contact=40.0, perturb=0.20, curl=0.5, tension=17.0, outer=40, maxiter=200)       # as shell_target.DEFAULTS, no creep
UNSOURCED = ['stages N = 10', 'stage shell hold 0.5 x EA/R', 'final shell hold 0.05 x EA/R (13 Nov 2026 success setting)', 'stage relaxation 100 iterations x 1 round', 'blocked fraction 0.8 of 2R (axis distance)',
             'fold path = shell growth equilibria (curvature fraction = fold fraction), interpolated linearly between neighbouring growth steps', 'smaller fraction at a shell vertex shared with a frozen cell', 'normals of the interpolated sheet', 'segment-to-cell by flat midpoint',
             'spacings from cell means', 'sign-agreement threshold 1e-3', 'gain and sign of the strain rule (shell_target)', 'ref-scale 0 (open)', 'kb0 rotated into the sheet frame at every stage',
             'frozen cells are not reverted']


# ------------------------------------------------------------------ lazily imported solver-side modules (jax is imported only here; the numpy tests insert stand-ins)
def _jd(o):
    return o.item() if hasattr(o, 'item') else str(o)

def _rj():
    import relax_job as rj; return rj

def _sv():
    import relax_solver as sv; return sv


# ------------------------------------------------------------------ shell at a fraction, cells (numpy only)
def flat_grid(sh):
    gy, gx = np.asarray(sh['X']).shape[:2]; ii, jj = np.meshgrid(np.arange(gx), np.arange(gy))
    return np.stack([ii*sh['W']/sh['sub'], jj*sh['C']/sh['sub'], np.zeros((gy, gx))], -1)

def fold_path(sh):
    """(fractions, sheets) of the fold path. With sh['path'] (set by shell_path, task 52, 2026-11-26): the shell's OWN equilibria after every growth step, fraction k/N, from the flat sheet to the shell.
    Without it: the two-point path flat -> shell, i.e. the straight line X_t = flat + t (X - flat) used before 2026-11-26. That straight line keeps no length where the sheet turns: at t = 0.5 of a
    180-degree roll the course stretch was 0.02 (review of 2026-11-26), so the middle stages saw a false compression and froze the crests; it is kept only for the numpy tests and --linear-path."""
    p = sh.get('path')
    if p is not None: return np.asarray(p[0], float), np.asarray(p[1], float)
    return np.array([0.0, 1.0]), np.stack([flat_grid(sh), np.asarray(sh['X'], float)])

def _bracket(ts, t):
    """Index k and weight w with t = (1 - w) ts[k] + w ts[k+1] (clipped to the path)."""
    t = np.asarray(t, float); k = np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 2); return k, np.clip((t - ts[k])/(ts[k + 1] - ts[k]), 0.0, 1.0)

def shell_at(sh, t_grid):
    """The sheet at fold fraction t (per shell grid vertex), interpolated between the two neighbouring sheets of the fold path (fold_path). Same keys as sh."""
    ts, G = fold_path(sh); t = np.asarray(t_grid, float); k, w = _bracket(ts, t); jj, ii = np.indices(t.shape)
    return dict(sh, X=(1 - w)[..., None]*G[k, jj, ii] + w[..., None]*G[k + 1, jj, ii])

def shell_path(run_dir, sh, out):
    """The shell's own fold path for task 52: the same shell solve as the run (its settings.json; shell_sim.fold with the expressions of shell_run.py), recording the sheet after every growth step.
    Cached in <out>/shell_path.npz (written atomically; never in the campaign's run folder). The last sheet must equal the run's shell.npz: with the same code and settings the solve is deterministic;
    a difference (another code version) is reported and the last sheet is replaced by the run's own, so the path always ends at the shell that was analysed. Needs JAX (one shell solve, minutes)."""
    cp = os.path.join(out, 'shell_path.npz')
    if os.path.exists(cp):
        z = np.load(cp); return dict(fractions=z['t'], sheets=z['X'], end_diff=float(z['end_diff']), cached=True)
    import shell_sim as ss
    rj = _rj(); s = json.load(open(os.path.join(run_dir, 'settings.json'))); W, C = 25.4/s['wpi'], 25.4/s['cpi']
    kappa, h = rj.SHELL_KAPPA_C/C, rj.SHELL_H_C*C*s.get('h_scale', 1.0); kx, ky = s.get('kappa_scale', 1.0)*(-rj.SHELL_KAPPA_C/C), s.get('kappa_scale', 1.0)*(rj.SHELL_KAPPA_C/C)
    snaps = []; t0 = time.time(); print('fold path: recomputing the shell growth (%d steps) to record its equilibria' % s['steps'], flush=True)
    X0, X, tri, (nx, ny), log = ss.fold(s['matrix'], W, C, kappa=kappa, h=h, nu=s.get('nu', 0.4), sub=s['sub'], ntile=(s['tiles'], s['tiles']), steps=s['steps'], maxiter=s['iter'], springs=s.get('springs', 0.0),
                                        seed=s.get('seed', 0), verbose=False, kx=kx, ky=ky, bending=s.get('bending', 'hinge'), snapshots=snaps)
    gy, gx = np.asarray(sh['X']).shape[:2]; sheets = np.stack([flat_grid(sh)] + [np.asarray(x).reshape(gy, gx, 3) for x in snaps]); ts = np.r_[0.0, [r[0] for r in log]]
    end_diff = float(np.abs(sheets[-1] - np.asarray(sh['X'], float)).max())
    if end_diff > 1e-9: print("fold path: the recomputed shell differs from the run by up to %.3g mm (another code version?): the last sheet is replaced by the run's own" % end_diff, flush=True)
    sheets[-1] = np.asarray(sh['X'], float)
    with open(cp + '.tmp', 'wb') as fh: np.savez(fh, t=ts, X=sheets, end_diff=end_diff); fh.flush(); os.fsync(fh.fileno())
    os.replace(cp + '.tmp', cp); print('fold path: %d sheets in %.0f s, end difference %.3g mm' % (len(ts), time.time() - t0, end_diff), flush=True)
    return dict(fractions=ts, sheets=sheets, end_diff=end_diff, cached=False)

def cell_layout(P, sh):
    """(cell index per vertex, ids of the used stitch cells (cj nx + ci), their number): the same cells as relax_job.shell_cells, with the ids kept."""
    nx, ny = sh['nx'], sh['ny']; ci = np.clip((P[:, 0]//sh['W']).astype(int), 0, nx - 1); cj = np.clip((P[:, 1]//sh['C']).astype(int), 0, ny - 1)
    uniq, cell = np.unique(cj*nx + ci, return_inverse=True); return cell.ravel(), uniq.astype(int), len(uniq)

def to_grid(vals, uniq, nx, ny, fill=np.nan):
    g = np.full(ny*nx, fill, float); g[uniq] = vals; return g.reshape(ny, nx)

def grid_fraction(frac_cells, uniq, sh, t_default):
    """Fold fraction at every vertex of the shell grid from the per-cell fractions: a vertex shared by several stitch cells takes the smallest of them; cells without yarn vertices use t_default."""
    nx, ny, sub = sh['nx'], sh['ny'], sh['sub']; tc = to_grid(frac_cells, uniq, nx, ny, fill=t_default)
    i = np.arange(nx*sub + 1); j = np.arange(ny*sub + 1)
    ia, ib = np.clip(i//sub, 0, nx - 1), np.clip((i - 1)//sub, 0, nx - 1); ja, jb = np.clip(j//sub, 0, ny - 1), np.clip((j - 1)//sub, 0, ny - 1)
    stack = np.stack([tc[jb][:, ib], tc[jb][:, ia], tc[ja][:, ib], tc[ja][:, ia]]); return stack.min(0)


# ------------------------------------------------------------------ rest lengths at a fraction (numpy only, via shell_target)
def deformation_at(sh, tc):
    """Deformation gradient of the shell's OWN triangles at fold fraction t, with t constant on every stitch cell (tc, (ny, nx)): (Fx, Fy) as shell_target.deformation, interpolated per triangle between
    the two neighbouring sheets of the fold path (fold_path; with the two-point path this is F_t = I + t (F - I), the formula used before 2026-11-26). Taken from the path sheets, NOT from the
    interpolated sheet of the targets: where a frozen cell (small t) touches a free cell (large t) that sheet has a step of one grid spacing and a false strain of the order of 100 % (found by the
    synthetic test); this F has no such step, and is exactly the strain of the path wherever t is uniform."""
    ts, G = fold_path(sh); key = (id(sh.get('path')), G.shape)
    if sh.get('_Fpath_key') != key: sh['_Fpath'] = [st.deformation(dict(sh, X=g)) for g in G]; sh['_Fpath_key'] = key          # cached per path
    FP = sh['_Fpath']; sub = sh['sub']; tg = np.kron(np.asarray(tc, float), np.ones((sub, sub))); k, w = _bracket(ts, tg)
    Fx = np.stack([f[0] for f in FP]); Fy = np.stack([f[1] for f in FP]); jj, ii = np.indices(tg.shape); w = w[None, :, :, None]
    pick = lambda A, kk: np.moveaxis(A[kk, :, jj, ii], 2, 0)          # (2, gy-1, gx-1, 3) at path index kk of every cell
    return (1 - w)*pick(Fx, k) + w*pick(Fx, k + 1), (1 - w)*pick(Fy, k) + w*pick(Fy, k + 1)

def stage_factors(flatX, l0_start, sh, Fx, Fy, sign, gain):
    """Per course: lambda of every flat segment (shell_target.segment_stretch with the deformation Fx, Fy of the shell triangle holding the segment midpoint), factor f = 1 + sign gain (e - e_bar)
    (shell_target.course_factors, conserves sum(l0)). Raises ValueError below the floor."""
    fs, lams, ebar = [], [], []
    for X, l0 in zip(flatX, l0_start):
        lam = st.segment_stretch(X, sh, Fx, Fy); f, eb = st.course_factors(l0, lam, sign, gain); fs.append(f); lams.append(lam); ebar.append(eb)
    if min(float(f.min()) for f in fs) < st.FACTOR_FLOOR: raise ValueError('a rest-length factor of %.3f is below the floor %.2f: lower --gain' % (min(float(f.min()) for f in fs), st.FACTOR_FLOOR))
    return fs, lams, ebar


# ------------------------------------------------------------------ blocked test and freezing (numpy only)
def cell_min_distance(P, yid, idx, cell, nc, R, skip=6, reach=1.5):
    """Closest distance between vertices of different yarns (or more than `skip` vertices apart on one yarn) for every cell, over the pairs with at least one vertex in it; inf where none within reach x 2R.
    Same pair rule as relax_solver.min_axis_distance."""
    d = np.full(nc, np.inf); pr = cKDTree(P).query_pairs(reach*2*R, output_type='ndarray')
    if len(pr) == 0: return d
    pr = pr[(yid[pr[:, 0]] != yid[pr[:, 1]]) | (np.abs(idx[pr[:, 0]] - idx[pr[:, 1]]) > skip)]
    if len(pr) == 0: return d
    dist = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1); np.minimum.at(d, cell[pr[:, 0]], dist); np.minimum.at(d, cell[pr[:, 1]], dist); return d

@dataclass
class FoldState:
    frac: np.ndarray            # fraction the target of every cell currently asks for
    frozen: np.ndarray          # bool
    last_clear: np.ndarray      # last fold fraction at which the cell was clear (0 = flat)
    frozen_stage: np.ndarray    # stage at which the cell froze (0 = never)
    stage_done: int = 0

    @classmethod
    def new(cls, nc): return cls(np.zeros(nc), np.zeros(nc, bool), np.zeros(nc), np.zeros(nc, int), 0)
    def targets(self, t): return np.where(self.frozen, self.frac, t)

def update_fractions(state, stage, t, dmin, thresh):
    """After the relaxation of stage `stage` (fraction t): blocked free cells (closest axis distance < thresh) freeze at their last clear fraction; the free cells that are clear advance. Returns the new blocked mask."""
    blocked = (dmin < thresh) & ~state.frozen
    state.frac[blocked] = state.last_clear[blocked]; state.frozen |= blocked; state.frozen_stage[blocked] = stage
    free = ~state.frozen; state.last_clear[free] = t; state.frac[free] = t; state.stage_done = stage
    return blocked


# ------------------------------------------------------------------ the folding context
class Ctx:
    """Holds the flat start (positions, start rest lengths, flat kb0), the shell and the cell layout; applies one fold state (rest lengths, hold targets, kb0 frame) to the Sim."""
    def __init__(self, sim, sh, sign=-1.0, gain=1.0, frame=True, kb0_flat=None):
        self.sim, self.sh, self.sign, self.gain, self.frame = sim, sh, float(sign), float(gain), bool(frame)
        self.flatX = [y.X.copy() for y in sim.yarns]; self.l0_start = [y.l0.copy() for y in sim.yarns]; self.P0 = np.vstack(self.flatX)
        self.yid = np.concatenate([[k]*len(y.X) for k, y in enumerate(sim.yarns)]); self.idx = np.concatenate([np.arange(len(y.X)) for y in sim.yarns])
        self.cell, self.uniq, self.nc = cell_layout(self.P0, sh); self.cnt = np.bincount(self.cell, minlength=self.nc).astype(float)
        self.kb0_flat = None if kb0_flat is None else np.array(kb0_flat, float); self.sh_t = None; self.tgt = None; self.factors = None; self.Fx = self.Fy = None

    def apply(self, frac_cells, t_default, ks, set_rest=True, set_kb0=True):
        rj = _rj(); sh_t = shell_at(self.sh, grid_fraction(frac_cells, self.uniq, self.sh, t_default))
        self.Fx, self.Fy = deformation_at(self.sh, to_grid(frac_cells, self.uniq, self.sh['nx'], self.sh['ny'], fill=t_default))
        if set_rest:
            fs, lams, ebar = stage_factors(self.flatX, self.l0_start, self.sh, self.Fx, self.Fy, self.sign, self.gain)
            for y, l0, f in zip(self.sim.yarns, self.l0_start, fs): y.l0 = l0*f
            self.factors = (fs, lams, ebar)
        U, N = rj.shell_field(sh_t); T = rj.shell_targets(sh_t, U, N, self.P0)
        tgt = np.c_[[np.bincount(self.cell, weights=T[:, k], minlength=self.nc) for k in range(3)]].T/self.cnt[:, None]
        self.sim.shell = dict(cell=self.cell, tgt=tgt, cnt=self.cnt, ks=ks)
        if set_kb0 and self.frame and self.kb0_flat is not None: self.sim.kb0 = self.kb0_flat.copy(); rj.rotate_kb0(self.sim, sh_t, self.P0)
        self.sh_t, self.tgt = sh_t, tgt

    def min_distance(self):
        return cell_min_distance(np.vstack([y.X for y in self.sim.yarns]), self.yid, self.idx, self.cell, self.nc, self.sim.R)

    def conservation_error(self):
        """Largest relative change of the total rest length of one course against the start (0 = every course keeps its yarn)."""
        return max(abs(float(y.l0.sum()) - float(l.sum()))/float(l.sum()) for y, l in zip(self.sim.yarns, self.l0_start))

def run_stages(ctx, state, n_stages, ks_stage, relax_fn, thresh, dmin_fn=None, on_stage=None):
    """Stages state.stage_done + 1 .. n_stages (a resumed run continues). relax_fn(sim) relaxes; dmin_fn() (default ctx.min_distance) gives the closest axis distance per cell. Returns the stage records."""
    recs = []
    for s in range(state.stage_done + 1, n_stages + 1):
        t = s/n_stages; t0 = time.time(); state.frac[~state.frozen] = t
        ctx.apply(state.targets(t), t, ks_stage)
        relax_fn(ctx.sim); d = (dmin_fn or ctx.min_distance)(); new = update_fractions(state, s, t, d, thresh)
        rec = dict(stage=s, t=t, frozen=int(state.frozen.sum()), newly_frozen=int(new.sum()), dmin_min=float(d.min()) if len(d) else None, conservation_err=ctx.conservation_error(), seconds=round(time.time() - t0, 2))
        recs.append(rec)
        if on_stage: on_stage(s, rec)
    return recs


# ------------------------------------------------------------------ measurement per stitch cell (numpy only)
def _nanmean(*a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning); return np.nanmean(np.stack(a), axis=0)

def spacings(M):
    """Course spacing and wale spacing per cell from the grid of cell-mean positions M (ny, nx, 3): the mean distance to the existing neighbours along y (courses) and along x (wales)."""
    ny, nx = M.shape[:2]; dc = np.linalg.norm(M[1:] - M[:-1], axis=2); dw = np.linalg.norm(M[:, 1:] - M[:, :-1], axis=2); nan_r = np.full((1, nx), np.nan); nan_c = np.full((ny, 1), np.nan)
    return _nanmean(np.vstack([dc, nan_r]), np.vstack([nan_r, dc])), _nanmean(np.hstack([dw, nan_c]), np.hstack([nan_c, dw]))

def measure_state(flatX, X, l0, sh, cell, uniq):
    """Rest length per cell (sum of l0 of the segments whose FLAT midpoint lies in the cell), cell-mean positions, and the course and wale spacings from them."""
    nx, ny = sh['nx'], sh['ny']; rest = np.zeros(ny*nx)
    for fx, l in zip(flatX, l0):
        m = 0.5*(fx[:-1] + fx[1:]); ci = np.clip((m[:, 0]//sh['W']).astype(int), 0, nx - 1); cj = np.clip((m[:, 1]//sh['C']).astype(int), 0, ny - 1); np.add.at(rest, cj*nx + ci, l)
    rest = rest.reshape(ny, nx); rest[rest == 0] = np.nan
    P = np.vstack(X); cnt = np.bincount(cell, minlength=len(uniq)).astype(float); Mf = np.full((ny*nx, 3), np.nan)
    for k in range(3): Mf[uniq, k] = np.bincount(cell, weights=P[:, k], minlength=len(uniq))/cnt
    M = Mf.reshape(ny, nx, 3); cs, ws = spacings(M); return dict(rest=rest, M=M, course=cs, wale=ws)

def shell_strain_cells(sh, Fx=None, Fy=None):
    """Membrane strain of the shell along the courses (|dX/dx| - 1) and along the wales (|dX/dy| - 1), mean over the grid triangles of every stitch cell; (ny, nx) each. Fx, Fy: the deformation
    at the reached fold fractions (deformation_at); default the complete shell."""
    if Fx is None: Fx, Fy = st.deformation(sh)
    nx, ny, sub = sh['nx'], sh['ny'], sh['sub']
    f = lambda F: (np.linalg.norm(F, axis=-1).mean(0) - 1.0).reshape(ny, sub, nx, sub).mean((1, 3)); return f(Fx), f(Fy)

def agree(x, y, expected, eps=EPS):
    """Relation of y to x: Pearson correlation, the share of cells (|x|, |y| > eps) whose product of signs has the expected sign, their number, and a flag grid (1 agree, 0 disagree, nan below eps)."""
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > eps) & (np.abs(y) > eps)
    flag = np.full(x.shape, np.nan); flag[ok] = (np.sign(x[ok])*np.sign(y[ok])*expected > 0).astype(float)
    both = np.isfinite(x) & np.isfinite(y)
    r = float(np.corrcoef(x[both], y[both])[0, 1]) if both.sum() > 2 and x[both].std() > 0 and y[both].std() > 0 else None
    return dict(corr=r, share_agree=(float(np.nanmean(flag)) if ok.any() else None), n=int(ok.sum()), expected_sign=float(expected)), flag

def direction_check(before, after, target, e_course, e_wale, sign=-1.0, eps=EPS):
    """R1, R2, R3 (module text) for a before / after pair of measure_state results and the shell target spacings; returns (summary, per-cell arrays)."""
    dl = after['rest']/before['rest'] - 1.0; dcs = after['course']/before['course'] - 1.0; dws = after['wale']/before['wale'] - 1.0
    w = np.where(np.isfinite(before['rest']), before['rest'], 0.0); ev = np.where(np.isfinite(e_course), e_course, 0.0); row = (w*ev).sum(1, keepdims=True)/np.maximum(w.sum(1, keepdims=True), 1e-30); ecc = e_course - row
    out, flags = {}, {}
    rel = (('R1_loop_vs_course_strain_minus_course_mean', 'agree_R1', ecc, dl, float(np.sign(sign) or -1.0)), ('R2_course_spacing_vs_loop_length', 'agree_R2_course', dl, dcs, 1.0),
           ('R2_wale_spacing_vs_loop_length', 'agree_R2_wale', dl, dws, 1.0), ('R3_course_spacing_vs_wale_strain', 'agree_R3_course', e_wale, dcs, 1.0), ('R3_wale_spacing_vs_course_strain', 'agree_R3_wale', e_course, dws, 1.0))
    for name, short, x, y, ex in rel: out[name], flags[short] = agree(x, y, ex, eps)
    f1, f2, f3 = flags['agree_R1'], flags['agree_R2_course'], flags['agree_R2_wale']; allok = np.isfinite(f1) & np.isfinite(f2) & np.isfinite(f3)
    out['R1_R2_all_agree_share'] = float(((f1 == 1) & (f2 == 1) & (f3 == 1))[allok].mean()) if allok.any() else None          # cells where the loop rule AND both spacings follow the loop length
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out['spacing_vs_shell_target_rms'] = dict(course=float(np.sqrt(np.nanmean((after['course']/target['course'] - 1.0)**2))), wale=float(np.sqrt(np.nanmean((after['wale']/target['wale'] - 1.0)**2))))
        out['loop_length_change'] = dict(min=float(np.nanmin(dl)), max=float(np.nanmax(dl)), mean=float(np.nanmean(dl)))
    cells = dict(rest_before=before['rest'], rest_after=after['rest'], d_rest=dl, course_before=before['course'], course_after=after['course'], course_target=target['course'], d_course=dcs,
                 wale_before=before['wale'], wale_after=after['wale'], wale_target=target['wale'], d_wale=dws, e_course=e_course, e_wale=e_wale, e_course_minus_course_mean=ecc)
    cells.update(flags)
    return out, cells

def write_cells_csv(path, cells, nx, ny):
    keys = list(cells); rows = []
    for j in range(ny):
        for i in range(nx): rows.append([i, j] + [('' if not np.isfinite(cells[k][j, i]) else '%.8g' % cells[k][j, i]) for k in keys])
    with open(path, 'w', newline='') as fh: w = csv.writer(fh); w.writerow(['cell_x', 'cell_y'] + keys); w.writerows(rows)


# ------------------------------------------------------------------ writing the results (numpy only)
def write_fraction_csv(path, state, uniq, nx, dmin_final, thresh):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['cell_x', 'cell_y', 'fraction', 'frozen', 'frozen_at_stage', 'dmin_final_over_2R', 'blocked_at_end'])
        for k, u in enumerate(uniq): w.writerow([int(u % nx), int(u//nx), '%.6f' % state.frac[k], int(state.frozen[k]), int(state.frozen_stage[k]), '' if not np.isfinite(dmin_final[k]) else '%.6f' % dmin_final[k], int(dmin_final[k] < thresh)])

def summary_dict(state, dmin_final, thresh, stages, R, extra):
    """dmin_final: closest axis distance per cell ALREADY divided by 2R (as passed by run_fold and compared with thresh = blocked_frac). R is kept in the signature only; until 2026-11-26 the minimum was divided by 2R a second time."""
    n = len(state.frac); fin = np.isfinite(dmin_final)
    return dict(schema=SCHEMA, n_cells=int(n), n_frozen=int(state.frozen.sum()), share_frozen=float(state.frozen.mean()) if n else None, min_fraction=float(state.frac.min()) if n else None, mean_fraction=float(state.frac.mean()) if n else None,
                share_blocked_at_end=float((dmin_final[fin] < thresh).mean()) if fin.any() else None, dmin_final_over_2R_min=float(dmin_final[fin].min()) if fin.any() else None, stages=stages, unsourced=UNSOURCED, **extra)

def aggregate(folders, out=sys.stdout):
    """One CSV line per run folder with a fold_summary.json (the end-of-campaign self-intersection pass)."""
    cols = ['run', 'n_cells', 'n_frozen', 'share_frozen', 'min_fraction', 'mean_fraction', 'share_blocked_at_end', 'dmin_final_over_2R_min', 'stages_run', 'state']; w = csv.writer(out); w.writerow(cols)
    for f in folders:
        p = os.path.join(f, 'fold_summary.json')
        if not os.path.isfile(p): w.writerow([os.path.basename(os.path.normpath(f))] + ['missing']*(len(cols) - 1)); continue
        d = json.load(open(p)); w.writerow([os.path.basename(os.path.normpath(f)), d.get('n_cells'), d.get('n_frozen'), d.get('share_frozen'), d.get('min_fraction'), d.get('mean_fraction'), d.get('share_blocked_at_end'), d.get('dmin_final_over_2R_min'), len(d.get('stages', [])), d.get('state')])


# ------------------------------------------------------------------ the driver (needs the solver: jax)
def parse(argv):
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0]); A = a.add_argument
    A('--run'); A('--out'); A('--ne', type=float, default=12.0); A('--sign', type=float, choices=(-1.0, 1.0), default=-1.0); A('--gain', type=float, default=1.0, help='UNSOURCED')
    A('--stages', type=int, default=DEFAULT_STAGES, help='UNSOURCED'); A('--blocked-frac', type=float, default=BLOCKED_FRAC, help='blocked when the closest axis distance in a cell < this x 2R (UNSOURCED)')
    A('--stage-iter', type=int, default=STAGE_ITER, help='UNSOURCED'); A('--stage-rounds', type=int, default=STAGE_ROUNDS); A('--stage-shell-w', type=float, default=STAGE_SHELL_W, help='hold during the stages, x EA/R (UNSOURCED)')
    A('--shell-w', type=float, default=FINAL_SHELL_W, help='hold in the final relaxation, x EA/R'); A('--ref-scale', type=float, default=0.0); A('--no-shell-frame', action='store_true')
    for k, v in RELAX_DEFAULTS.items(): A('--' + k.replace('_', '-'), type=type(v), default=v)
    A('--seed', type=int, default=0); A('--sides', type=int, default=8); A('--length-ratio', type=float, default=1.0); A('--stretch-scale', type=float, default=1.0)
    A('--bend-scale', type=float, default=1.0); A('--ea', type=float, default=0.0); A('--ei', type=float, default=0.0); A('--stretch-law', default='')
    A('--loop-grade', type=float, default=0.0); A('--loop-grade-target', choices=['boundary', 'tm', 'both'], default='both'); A('--loop-grade-min-strength', type=float, default=0.2)
    A('--stages-only', action='store_true', help='stop after the stages (no final 40 x 200 relaxation)'); A('--build-only', action='store_true'); A('--resume', action='store_true')
    A('--linear-path', action='store_true', help='use the old straight-line fold path flat -> shell (before 2026-11-26) instead of the shell growth equilibria (task 52); only for comparison')
    A('--aggregate', nargs='+', metavar='DIR'); return a.parse_args(argv)

def save_checkpoint(path, sim, ctx, state, phase, k):
    d = {'X%d' % i: y.X for i, y in enumerate(sim.yarns)}; d.update({'l%d' % i: y.l0 for i, y in enumerate(sim.yarns)})
    tmp = path + '.tmp.npz'; np.savez(tmp, phase=phase, k=k, kb0=sim.kb0, elog=json.dumps(sim.energy_log), frac=state.frac, frozen=state.frozen, last_clear=state.last_clear, frozen_stage=state.frozen_stage, stage_done=state.stage_done, **d)
    os.replace(tmp, path)

def run_fold(g, out, sh, meta):
    """Everything after the build: setup as relax_job.main, the stages, the final relaxation, the files."""
    import argparse as ap, relax_state as rs
    rj, sv = _rj(), _sv(); t_all = time.time(); folder = out; ck_path = os.path.join(out, 'fold_checkpoint.npz')
    ns = ap.Namespace(stretch_scale=g.stretch_scale, bend_scale=g.bend_scale, ea=g.ea, ei=g.ei, stretch_law=g.stretch_law)
    try: EA, EI = rj.yarn_inputs(ns, meta['R'])
    except ValueError as err: sys.exit(str(err))
    if g.ref_scale == 0: print('note: --ref-scale 0 (straight-yarn reference), as shell_target.py; relax_job advises --ref-scale 1 with a shell (open parameter, task 47)', flush=True)
    ten_strain = rs.strain_at(0.01*g.tension/EA); ratio = g.length_ratio/(1.0 + ten_strain)
    p = rs.Params(mu=g.mu, EA=EA, EI=EI, GJ=0.0, k_contact=g.k_contact, outer_max=g.outer, rest_length_ratio=ratio, curl=g.curl, ref_scale=g.ref_scale)
    sim = rs.load(folder, p)
    if g.loop_grade:
        import loop_grade as lg
        try: lg.apply_start_grade(sim, folder, meta, g.loop_grade, g.loop_grade_target, g.loop_grade_min_strength, with_shell=True)
        except ValueError as e: sys.exit('--loop-grade: %s' % e)
    rs.release(sim); e0 = sv.energy_parts(sim, g.k_contact, g.E)                                   # also fixes kb0 of the flat state (relax_solver.baseline_curvature)
    ctx = Ctx(sim, sh, g.sign, g.gain, not g.no_shell_frame, kb0_flat=sim.kb0); R = sim.R; thresh = g.blocked_frac*2*R; ks_stage = g.stage_shell_w*(EA/g.stretch_scale)/R; ks_final = g.shell_w*(EA/g.stretch_scale)/R
    d0 = ctx.min_distance()
    if np.isfinite(d0).any() and float(d0[np.isfinite(d0)].min()) < thresh: sys.exit('the flat start already has axes closer than the blocked threshold (%.3f x 2R < %.3f x 2R): lower --blocked-frac' % (d0[np.isfinite(d0)].min()/(2*R), g.blocked_frac))
    before = measure_state(ctx.flatX, ctx.flatX, ctx.l0_start, sh, ctx.cell, ctx.uniq)
    state = FoldState.new(ctx.nc); phase, k0 = 'stage', 0
    if g.resume:
        if not os.path.exists(ck_path): sys.exit('--resume: no fold_checkpoint.npz in %s' % out)
        ck = np.load(ck_path, allow_pickle=False); phase, k0 = str(ck['phase']), int(ck['k'])
        state = FoldState(ck['frac'].copy(), ck['frozen'].copy(), ck['last_clear'].copy(), ck['frozen_stage'].copy(), int(ck['stage_done']))
        for i, y in enumerate(sim.yarns): y.X[:] = ck['X%d' % i]; y.l0[:] = ck['l%d' % i]
        sim.energy_log = json.loads(str(ck['elog'])); print('resume: phase %s, k %d, %d stages done' % (phase, k0, state.stage_done), flush=True)
        if phase == 'done': sys.exit('%s is already finished' % out)
    else:
        rng = np.random.default_rng(g.seed)
        for y in sim.yarns: y.X[:, 2] += g.perturb*R*rng.standard_normal(len(y.X))
    stage_csv = os.path.join(out, 'fold_stage_log.csv'); relax_stage = lambda s: sv.relax(s, g.E, g.k_contact, maxiter=g.stage_iter, rounds=g.stage_rounds, verbose=True)
    def on_stage(s, rec):
        e = sv.energy_parts(sim, g.k_contact, g.E); dv = sv.shell_deviation(sim); rec.update(E=e['total'], E_shell=e['shell'], shell_rms_over_R=(dv['rms']/R if dv else None))
        new = not os.path.exists(stage_csv)
        with open(stage_csv, 'a', newline='') as fh:
            w = csv.writer(fh); new and w.writerow(['stage', 't', 'frozen', 'newly_frozen', 'dmin_min_over_2R', 'conservation_err', 'E', 'E_shell', 'shell_rms_over_R', 'seconds'])
            w.writerow([s, rec['t'], rec['frozen'], rec['newly_frozen'], '' if rec['dmin_min'] is None else rec['dmin_min']/(2*R), rec['conservation_err'], rec['E'], rec['E_shell'], rec['shell_rms_over_R'], rec['seconds']])
        print('stage %d/%d (fold %.2f): %d cells frozen (+%d), closest axes %.3f x 2R, E %.6g, shell rms %.3f R, %.0f s' % (s, g.stages, rec['t'], rec['frozen'], rec['newly_frozen'], (rec['dmin_min'] or 0)/(2*R), rec['E'], rec['shell_rms_over_R'] or 0, rec['seconds']), flush=True)
        save_checkpoint(ck_path, sim, ctx, state, 'stage', s)
    stages = []
    if phase == 'stage': stages = run_stages(ctx, state, g.stages, ks_stage, relax_stage, thresh, on_stage=on_stage)
    # ---- final relaxation
    ctx.apply(state.targets(1.0), 1.0, ks_final, set_rest=(phase == 'stage'), set_kb0=(phase == 'stage'))
    if phase == 'final': sim.kb0 = np.load(ck_path)['kb0'].copy()
    status = 'stages only'
    if not g.stages_only:
        save_checkpoint(ck_path, sim, ctx, state, 'final', k0 if phase == 'final' else 0)
        def on_round(k, E, dmax):
            print('outer round %d: E %.6g, largest rest-length change %.3g' % (k + 1, E, dmax), flush=True); save_checkpoint(ck_path, sim, ctx, state, 'final', k + 1)
        relax_final = lambda s: sv.relax(s, g.E, g.k_contact, maxiter=g.maxiter, verbose=True); energy = lambda s: sv.energy_parts(s, g.k_contact, g.E)['total']
        rs.run(sim, relax_final, energy, on_round, k0=(k0 if phase == 'final' else 0)); status = sim.state.value
    dmin_f = ctx.min_distance(); e1 = sv.energy_parts(sim, g.k_contact, g.E); dv = sv.shell_deviation(sim)
    after = measure_state(ctx.flatX, [y.X for y in sim.yarns], [y.l0 for y in sim.yarns], sh, ctx.cell, ctx.uniq)
    nx, ny = sh['nx'], sh['ny']; tM = np.full((ny*nx, 3), np.nan); tM[ctx.uniq] = ctx.tgt; ts, tw = spacings(tM.reshape(ny, nx, 3)); ec, ew = shell_strain_cells(sh, ctx.Fx, ctx.Fy)
    chk, cells = direction_check(before, after, dict(course=ts, wale=tw), ec, ew, g.sign)
    write_cells_csv(os.path.join(out, 'fold_cell_metrics.csv'), cells, nx, ny); json.dump(dict(schema=SCHEMA, sign=g.sign, gain=g.gain, eps=EPS, relations=chk, state=status), open(os.path.join(out, 'fold_check.json'), 'w'), indent=1, default=_jd)
    write_fraction_csv(os.path.join(out, 'fold_fraction.csv'), state, ctx.uniq, nx, dmin_f/(2*R), g.blocked_frac)
    extra = dict(state=status, settings=dict(stages=g.stages, blocked_frac=g.blocked_frac, stage_iter=g.stage_iter, stage_rounds=g.stage_rounds, stage_shell_w=g.stage_shell_w, shell_w=g.shell_w, sign=g.sign, gain=g.gain, ref_scale=g.ref_scale,
                 shell_frame=not g.no_shell_frame, E=g.E, mu=g.mu, k_contact=g.k_contact, outer=g.outer, maxiter=g.maxiter, perturb=g.perturb, curl=g.curl, tension_cN=g.tension, loop_grade=g.loop_grade),
                 end=dict(e1, min_dist_over_2R=float(sv.min_axis_distance(sim)/(2*R))), shell_rms_over_R=(dv['rms']/R if dv else None), seconds=round(time.time() - t_all, 1), start=dict(e0))
    s = summary_dict(state, dmin_f/(2*R), g.blocked_frac, stages, R, extra); json.dump(s, open(os.path.join(out, 'fold_summary.json'), 'w'), indent=1, default=_jd)
    np.savetxt(os.path.join(out, 'relaxed_axis.csv'), np.vstack([np.c_[np.full(len(y.X), y.course), y.X] for y in sim.yarns]), delimiter=',', header='course,x,y,z', comments='', fmt=['%d', '%.6f', '%.6f', '%.6f'])
    np.savetxt(os.path.join(out, 'relaxed_l0.csv'), np.vstack([np.c_[np.full(len(y.l0), y.course), np.arange(len(y.l0)), y.l0] for y in sim.yarns]), delimiter=',', header='course,seg,l0', comments='', fmt=['%d', '%d', '%.10g'])
    from build_knit import write_obj; write_obj(os.path.join(out, 'relaxed_tube.obj'), [y.X for y in sim.yarns], R, g.sides, 'progressive fold of %s' % os.path.basename(os.path.abspath(out)))
    sd = json.load(open(os.path.join(out, 'state.json'))); sd['summary'] = dict(start=e0, end=e1, method='shell_fold', fold_summary='fold_summary.json'); json.dump(sd, open(os.path.join(out, 'state.json.tmp'), 'w'), indent=1); os.replace(os.path.join(out, 'state.json.tmp'), os.path.join(out, 'state.json'))
    if not g.stages_only: save_checkpoint(ck_path, sim, ctx, state, 'done', g.outer)
    print('fold done (%s): %d of %d cells frozen, minimum fraction %.2f, %d cells still blocked at the end; checks in fold_check.json' % (status, state.frozen.sum(), ctx.nc, state.frac.min(), int((dmin_f < thresh).sum())), flush=True)
    sys.exit(0 if status in ('settled', 'stages only') else 2)

def main():
    if '--aggregate' not in sys.argv: st.refuse_if_busy(sys.argv)          # one solver job at a time (the aggregate pass only reads files)
    g = parse(sys.argv[1:])
    if g.aggregate: return aggregate(g.aggregate)
    if not g.run or not g.out: sys.exit('--run and --out are required (or --aggregate DIR ...)')
    if not 0.0 < g.blocked_frac <= 1.0: sys.exit('--blocked-frac must be in (0, 1]')
    if g.stages < 1: sys.exit('--stages must be >= 1')
    s = json.load(open(os.path.join(g.run, 'settings.json'))); shp = os.path.join(g.run, 'shell.npz')
    if not os.path.isfile(os.path.join(g.run, 'metrics.json')) or not os.path.isfile(shp): sys.exit('%s is not a finished shell run (metrics.json and shell.npz needed)' % g.run)
    marker = os.path.join(g.out, 'fold_source.json')
    if os.path.exists(marker):
        if not g.resume and not g.build_only: sys.exit('%s already holds a fold; never overwritten (use --resume to continue it)' % g.out)
    else:
        if os.path.exists(g.out) and os.listdir(g.out): sys.exit('%s exists and is not empty; never overwritten' % g.out)
        if g.resume: sys.exit('--resume: %s holds no fold to resume' % g.out)
        st.build(g.run, g.out, s, g.ne)
    sh = st.load_shell(shp); meta = json.load(open(os.path.join(g.out, 'meta.json'))); st.check_match(meta, sh)
    if not g.build_only and not g.linear_path:          # task 52: the shell's own equilibria as the fold path (one extra shell solve, cached in the output folder)
        pth = shell_path(g.run, sh, g.out); sh['path'] = (pth['fractions'], pth['sheets'])
    if not os.path.exists(os.path.join(g.out, 'shell.npz')): shutil.copy2(shp, os.path.join(g.out, 'shell.npz'))
    json.dump(dict(source_run=os.path.abspath(g.run), settings=s, args={k: v for k, v in vars(g).items() if k not in ('aggregate',)}, t=time.strftime('%Y-%m-%dT%H:%M:%S')), open(marker, 'w'), indent=1, default=str)
    if g.build_only: print('build only: %s ready (mesh and shell match)' % g.out); return
    run_fold(g, g.out, sh, meta)

if __name__ == '__main__':
    main()
