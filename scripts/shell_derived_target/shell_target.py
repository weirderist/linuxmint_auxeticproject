#!/usr/bin/env python3
"""shell_target.py - the SHELL-DERIVED TARGET FABRIC of one finished pure-shell run (K's plan of 23 Nov 2026: results first, replicating the results with parameters next).

Pipeline (one shell run in, one ordinary result folder out; nothing outside --out is written, the shell run is only read):
  1. build   the yarn mesh of the run's own pattern with build_knit.py (same matrix, tiles, CPI, WPI; Ne from --ne), so the mesh and the shell sheet share one flat frame.
  2. fold    the campaign's OWN shell.npz is used (copied into --out); the fold is never recomputed, so the target belongs to exactly that shell run.
  3. lengths every yarn segment gets a new rest length from the membrane strain of the shell along that segment (definitions below), conserving the total yarn of every course.
  4. relax   relax_job.py, unchanged, places the yarn on the fold, holds the stitch cells toward it and runs the outer loop (L-BFGS-B per round, capstan slip between rounds).
Step 3 is put into relax_job.py by replacing loop_grade.apply_start_grade for this process only (relax_job calls it at exactly the right moment: after loading the flat as-built yarn, before
the yarn is moved onto the fold). Step 2 replaces relax_job.run_shell the same way. No existing file is edited. In the result, params.loop_grade is only the switch that makes relax_job call
step 3; the record saved under 'loop_grade' has method 'shell_strain' and holds what was really done.

Definitions (step 3)
  flat frame      the sheet before folding: x along the courses (stitch width W = 25.4/WPI mm), y along the wales (course spacing C = 25.4/CPI mm); build_knit.py and shell_sim.py use the same frame.
  deformation     F = [dX/dx, dX/dy] (3 x 2) of the folded sheet X(x, y) (shell.npz), constant on each triangle of the shell's own mesh (the triangles its membrane energy is computed on).
                  Central differences on the grid were rejected: on folds of radius about one grid step they shorten the chords and show about 10 % false compression.
  segment stretch for a yarn segment d = (dx, dy, dz) of the flat as-built yarn: lambda = |(F (dx, dy), dz)| / |d|, F of the shell triangle holding the segment midpoint. A pure fold (isometric
                  bending) gives lambda = 1; lambda > 1 where the sheet is stretched (tension, red on the strain map), < 1 where it is compressed (blue). Only the membrane strain of the
                  shell enters; the change of length of yarn lying off the mid-surface when the sheet bends (factor 1 - z kappa) is NOT included.
  rule            e = lambda - 1; in every course (one yarn), e_bar = rest-length-weighted mean of e; factor f = 1 + sign x gain x (e - e_bar); l0 <- l0 x f.
                  sum(l0 f) = sum(l0) exactly, so every course keeps its yarn (yarn is inextensible and slip moves rest length only along one yarn).
  sign = -1       (default, K's description, 23 Nov 2026): tension sites give yarn away (shorter rest length, smaller loops), compression sites collect it (longer, bigger loops). Held on the
                  stretched shell geometry, a shortened segment is under tension, so the yarn tension follows the shell's tension sites.
  sign = +1       the opposite: rest length follows the shell metric, so the yarn is closer to stress-free on the folded sheet.
UNSOURCED: gain (default 1), the sign choice, F taken constant per shell triangle at the segment midpoint, the per-course weighted mean, FACTOR_FLOOR (numerical guard, as loop_grade.py).

Relaxation defaults (overridable after --): the 13 Nov 2026 success settings recorded in CLAUDE.md (E 3700 then; the default is now E 500 MPa, derived from Xiong et al. 2026 (webui/main.py CONFIG, 2026-11-26): give -- --E 3700 to reproduce 13 Nov exactly; mu 0.49, k-contact 40, perturb 0.20, curl 0.5, tension 17 cN, shell hold 0.05 x EA/R)
WITHOUT creep (creep does not conserve rest length and would erase step 3), with outer 40 and maxiter 200 (K, 24 Nov 2026). ref-scale is left at relax_job's default, as in the 13 Nov run.

Usage:
  python3 shell_target.py --run campaigns/shell_runs/shell1/runs/S000001 --out targets/S000001 [--ne 12] [--sign -1] [--gain 1] [--build-only] [-- extra relax_job options]
  Resume after an interruption: the same command with  -- --resume  (the build and the shell copy are skipped when meta.json and shell.npz are already in --out).
"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_S = os.path.dirname(HERE)
if os.path.basename(_S) == 'scripts': sys.path.extend(os.path.join(_S, _d) for _d in sorted(os.listdir(_S)) if _d not in ('legacy', '__pycache__') and os.path.isdir(os.path.join(_S, _d)))          # the other folders of scripts/ (not in a flat frozen copy)

FACTOR_FLOOR = 0.2          # numerical guard, as loop_grade.FACTOR_FLOOR
import yarn_defaults as YD          # 2026-11-28: the standard settings (was: --k-contact 40 ABSOLUTE = 0.19 EA/R, --ref-scale 0, curl 0.5, tension 17, perturb 0.2: the yarns passed through each other)
_S = YD.STANDARD
DEFAULTS = ['--E', repr(_S['E']), '--mu', repr(_S['mu']), '--perturb', repr(_S['perturb']), '--curl', repr(_S['curl']), '--tension', repr(_S['tension_cN']), '--ref-scale', repr(_S['ref_scale']),
            '--shell-w', '0.05', '--outer', str(_S['outer']), '--maxiter', str(_S['maxiter']), '--stall-tol', repr(_S['stall_tol'])]          # --k-contact is added from R at run time (kc_ratio x EA/R)
UNSOURCED = ['gain', 'sign', 'F constant per shell triangle, taken at the segment midpoint', 'rest-length-weighted course mean', 'factor floor 0.2 (numerical guard)']


# ------------------------------------------------------------------ step 3: rest lengths from the shell strain (numpy only; tested without the solver)
def deformation(sh):
    """Piecewise-constant F of the shell's own triangle mesh (the triangulation of shell_sim.build, whose membrane energy is per triangle): per grid cell and per triangle,
    dX/dx and dX/dy from the triangle's edges. Returns (Fx, Fy), each (2, gy-1, gx-1, 3): index 0 = the triangle containing the cell's first corner, 1 = the other."""
    X = np.asarray(sh['X'], float); hx, hy = sh['W']/sh['sub'], sh['C']/sh['sub']
    A, B, Cc, D = X[:-1, :-1], X[:-1, 1:], X[1:, 1:], X[1:, :-1]                        # corners (i, j), (i+1, j), (i+1, j+1), (i, j+1) of every cell
    ny_, nx_ = A.shape[:2]; even = ((np.arange(nx_)[None, :] + np.arange(ny_)[:, None]) % 2 == 0)[..., None]
    # even cell: triangles (a, b, c) below the diagonal a-c and (a, c, d) above it; odd cell: (a, b, d) below the diagonal b-d and (b, c, d) above it (shell_sim.build)
    Fx0 = (B - A)/hx; Fy0 = np.where(even, (Cc - B)/hy, (D - A)/hy)
    Fx1 = (Cc - D)/hx; Fy1 = np.where(even, (D - A)/hy, (Cc - B)/hy)
    return np.stack([Fx0, Fx1]), np.stack([Fy0, Fy1])

def _locate(x, y, sh, nx_, ny_):
    """(cell row j, cell column i, triangle 0/1) of flat points; points outside the sheet use the nearest cell."""
    hx, hy = sh['W']/sh['sub'], sh['C']/sh['sub']; u = np.clip(x/hx, 0, nx_ - 1e-9); v = np.clip(y/hy, 0, ny_ - 1e-9)
    i = u.astype(int); j = v.astype(int); fu = u - i; fv = v - j; even = (i + j) % 2 == 0
    return j, i, np.where(even, (fv > fu), (fu + fv > 1)).astype(int)

def segment_stretch(P, sh, Fx=None, Fy=None):
    """lambda per segment of the flat polyline P (n, 3): folded length / flat length (definition in the module text), F of the shell triangle that holds the segment midpoint."""
    if Fx is None: Fx, Fy = deformation(sh)
    d = np.diff(P, axis=0); m = 0.5*(P[:-1] + P[1:]); j, i, t = _locate(m[:, 0], m[:, 1], sh, Fx.shape[2], Fx.shape[1])
    v = Fx[t, j, i]*d[:, 0:1] + Fy[t, j, i]*d[:, 1:2]
    return np.sqrt((v*v).sum(1) + d[:, 2]**2)/np.maximum(np.linalg.norm(d, axis=1), 1e-30)

def course_factors(l0, lam, sign, gain):
    """Rest-length factors of one course: f = 1 + sign gain (e - e_bar), e = lam - 1, e_bar weighted by l0. Conserves sum(l0)."""
    e = lam - 1.0; eb = float((l0*e).sum()/l0.sum()); return 1.0 + sign*gain*(e - eb), eb

def apply_shell_strain(sim, sh, sign=-1.0, gain=1.0):
    """Change every yarn's l0 in place (call on the FLAT as-built yarn); returns the record."""
    Fx, Fy = deformation(sh); fs, lams, ebar, before, after = [], [], [], 0.0, 0.0
    for y in sim.yarns:
        lam = segment_stretch(y.X, sh, Fx, Fy); f, eb = course_factors(y.l0, lam, sign, gain)
        fs.append(f); lams.append(lam); ebar.append(eb)
    fa = np.concatenate(fs)
    if fa.min() < FACTOR_FLOOR: raise ValueError('a rest-length factor of %.3f is below the floor %.2f: lower --gain' % (fa.min(), FACTOR_FLOOR))
    for y, f in zip(sim.yarns, fs): before += float(y.l0.sum()); y.l0 = y.l0*f; after += float(y.l0.sum())
    la = np.concatenate(lams)
    return dict(method='shell_strain', sign=float(sign), gain=float(gain), sections_count=len(sim.yarns), factor_min=float(fa.min()), factor_max=float(fa.max()),
                total_l0_before=before, total_l0_after=after, stretch_min=float(la.min()), stretch_max=float(la.max()), stretch_mean=float(la.mean()),
                course_mean_strain=[float(v) for v in ebar], conservation='per course (one yarn): sum of rest lengths unchanged',
                note='params.loop_grade in this run is only the switch that called this rule; see scripts/shell_derived_target/shell_target.py', unsourced=UNSOURCED), fs


# ------------------------------------------------------------------ steps 1, 2 and 4
def load_shell(path):
    z = np.load(path); return dict(X=z['X'], W=float(z['W']), C=float(z['C']), sub=int(z['sub']), nx=int(z['nx']), ny=int(z['ny']))

def build(run, out, s, ne):
    mat = '/'.join(' '.join(str(int(v)) for v in row) for row in s['matrix'])
    cmd = [sys.executable, os.path.join(os.path.dirname(HERE), 'yarn_model', 'build_knit.py'), '--matrix', mat, '--tiles', str(int(s['tiles'])), '--ne', str(ne), '--cpi', str(s['cpi']), '--wpi', str(s['wpi']),
           '--out', out, '--no-tube', '--name', 'shell target of %s' % os.path.basename(os.path.normpath(run)), '--id', 'target-%s' % os.path.basename(os.path.normpath(run))]
    print('build: ' + ' '.join(cmd), flush=True); subprocess.run(cmd, check=True)

def check_match(meta, sh):
    """The mesh and the shell must share W, C and the stitch counts."""
    bad = []
    if abs(meta['W'] - sh['W']) > 1e-6*sh['W'] or abs(meta['C'] - sh['C']) > 1e-6*sh['C']: bad.append('W, C: mesh %.6g, %.6g; shell %.6g, %.6g' % (meta['W'], meta['C'], sh['W'], sh['C']))
    if meta['stitches_per_course'] != sh['nx'] or meta['courses'] != sh['ny']: bad.append('stitches: mesh %d x %d, shell %d x %d' % (meta['stitches_per_course'], meta['courses'], sh['nx'], sh['ny']))
    if bad: sys.exit('mesh and shell do not match: ' + '; '.join(bad))

def refuse_if_busy(argv):
    """One solver job at a time on the server (2026-11-26). The campaign runners cannot see this script (it runs relax_job in-process, and the frozen shell1 runner only knows relax_job.py, tile_sim.py and
    shell_run.py), so it checks them itself before starting. --ignore-busy skips the check (K's decision)."""
    if '--ignore-busy' in argv: argv.remove('--ignore-busy'); return
    why = ''
    try:
        import campaign as cm; why = cm.busy_reason()
        if not why: import shell_campaign as sc; why = sc.busy_message()
    except Exception: return          # the check is a courtesy; a missing module must not stop a run on a machine without campaigns
    if why: sys.exit('another solver job is running, so this one does not start (one job at a time): %s\nWait for it, pause the campaign, or add --ignore-busy.' % why)

def main():
    refuse_if_busy(sys.argv)
    argv = sys.argv[1:]; extra = argv[argv.index('--') + 1:] if '--' in argv else []; argv = argv[:argv.index('--')] if '--' in argv else argv
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0]); a.add_argument('--run', required=True); a.add_argument('--out', required=True)
    a.add_argument('--ne', type=float, default=12.0, help='yarn count (the shell run has none; K fixed Ne 12)'); a.add_argument('--sign', type=float, choices=(-1.0, 1.0), default=-1.0)
    a.add_argument('--gain', type=float, default=1.0, help='UNSOURCED'); a.add_argument('--build-only', action='store_true')
    g = a.parse_args(argv)
    s = json.load(open(os.path.join(g.run, 'settings.json'))); shp = os.path.join(g.run, 'shell.npz')
    if not os.path.isfile(os.path.join(g.run, 'metrics.json')) or not os.path.isfile(shp): sys.exit('%s is not a finished shell run (metrics.json and shell.npz needed)' % g.run)
    resume = '--resume' in extra
    if os.path.exists(os.path.join(g.out, 'meta.json')):
        if not resume and not g.build_only: sys.exit('%s already holds a result; never overwritten (use -- --resume to continue it)' % g.out)
    else:
        if os.path.exists(g.out) and os.listdir(g.out): sys.exit('%s exists and is not empty; never overwritten' % g.out)
        build(g.run, g.out, s, g.ne)
    sh = load_shell(shp); meta = json.load(open(os.path.join(g.out, 'meta.json'))); check_match(meta, sh)
    dst = os.path.join(g.out, 'shell.npz')
    if not os.path.exists(dst): shutil.copy2(shp, dst)
    json.dump(dict(source_run=os.path.abspath(g.run), settings=s, ne=g.ne, sign=g.sign, gain=g.gain, extra=extra, t=time.strftime('%Y-%m-%dT%H:%M:%S')), open(os.path.join(g.out, 'target_source.json'), 'w'), indent=1)
    if g.build_only: print('build only: %s ready (mesh and shell match)' % g.out); return

    import relax_job as rj, loop_grade as lg
    def run_shell(_g, _meta, _kxy):
        print('shell: the campaign fold %s is used (not recomputed)' % shp, flush=True); return load_shell(dst)
    def grade(sim, folder, meta_, *_a, **_k):
        rec, fs = apply_shell_strain(sim, sh, g.sign, g.gain)
        np.savetxt(os.path.join(folder, 'shell_l0_factor.csv'), np.vstack([np.c_[np.full(len(f), y.course), np.arange(len(f)), f] for y, f in zip(sim.yarns, fs)]), delimiter=',', header='course,seg,factor', comments='', fmt=['%d', '%d', '%.8f'])
        print('shell strain rule (sign %+g, gain %g, UNSOURCED): segment stretch %.4f to %.4f, rest-length factor %.4f to %.4f, every course conserved' % (g.sign, g.gain, rec['stretch_min'], rec['stretch_max'], rec['factor_min'], rec['factor_max']), flush=True)
        return rec
    rj.run_shell = run_shell; lg.apply_start_grade = grade
    kc = YD.k_contact(_S['E'], json.load(open(os.path.join(g.out, 'meta.json')))['R'])          # kc_ratio x EA/R of THIS build (never an absolute number)
    sys.argv = ['relax_job.py', '--folder', g.out, '--shell', '--loop-grade', '0.01', '--kappa-scale', str(s.get('kappa_scale', 1.0)), '--k-contact', repr(kc)] + DEFAULTS + extra    # later options win in argparse
    rj.main()

if __name__ == '__main__':
    main()
