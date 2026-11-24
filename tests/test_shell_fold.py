"""Tests of scripts/shell_fold.py (progressive fold): numpy and scipy only, NO JAX and no solver (a stand-in relax_solver module is put in sys.modules so that relax_job's numpy helpers import).
The relaxation is replaced by a stub that moves every stitch cell exactly onto its hold target, and the blocking by an injected distance function; so these tests check the LOGIC (freezing, blocked
test, conservation, signs, measurement) and say nothing about the physics of a real run.
Run from the repository root:  python tests/test_shell_fold.py"""
import os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.modules.setdefault('relax_solver', types.ModuleType('relax_solver'))          # stand-in: relax_job does `import relax_solver as sv` at import time (jax); none of its functions is called here
import numpy as np
import relax_state as rs, shell_fold as sf, shell_target as st
ok = True
def check(name, cond, info=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info)

# ---------------------------------------------------------------- synthetic shell and yarn
W, C, SUB, NX, NY, R = 1.6, 1.2, 2, 8, 6, 0.13
gx, gy = NX*SUB + 1, NY*SUB + 1; ii, jj = np.meshgrid(np.arange(gx), np.arange(gy)); x = ii*W/SUB; y = jj*C/SUB; L = NX*W
xs = np.where(x < L/2, 1.06*x, 1.06*L/2 + 0.94*(x - L/2))              # tension (+6 %) in the left half, compression (-6 %) in the right half, along the courses
ys = y*(1 + 0.04*np.cos(np.pi*x/L))                                   # and +4 % (left) to -4 % (right) along the wales, smoothly (a small shear comes with it)
r = 5.0; X = np.stack([r*np.sin(xs/r), ys, r - r*np.cos(xs/r)], -1)       # then bent onto a cylinder (isometric: no extra strain)
SH = dict(X=X, W=W, C=C, sub=SUB, nx=NX, ny=NY)

def make_sim(npts=80):
    yarns = []
    for j in range(NY):
        xx = np.linspace(0.05, L - 0.05, npts); P = np.c_[xx, np.full(npts, (j + 0.5)*C), 0.03*np.sin(3*xx)]
        yarns.append(rs.Yarn(j, P, np.linalg.norm(np.diff(P, axis=0), axis=1), np.zeros(npts - 1), np.ones(npts, bool), None))
    return rs.Sim('.', R, yarns, rs.Params(mu=0.5, EA=1.0, EI=1.0, GJ=0.0, k_contact=1.0))

# ---------------------------------------------------------------- 1 shell at a fraction, cells, vertex fractions
check('shell_at: t = 0 is the flat sheet, t = 1 the shell', np.allclose(sf.shell_at(SH, np.zeros((gy, gx)))['X'], sf.flat_grid(SH)) and np.allclose(sf.shell_at(SH, np.ones((gy, gx)))['X'], X))
sim = make_sim(); ctx = sf.Ctx(sim, SH, -1.0, 1.0, frame=False)
import relax_job as rj
cell_rj, nc_rj = rj.shell_cells(ctx.P0, SH); check('cell layout equals relax_job.shell_cells', np.array_equal(cell_rj, ctx.cell) and nc_rj == ctx.nc == NX*NY, (ctx.nc, NX*NY))
fr = np.ones(ctx.nc); k_f = 2*NX + 3; fr[k_f] = 0.4                                        # one cell frozen at 0.4, all others free at 1
tg = sf.grid_fraction(fr, ctx.uniq, SH, 1.0); ci, cj = k_f % NX, k_f//NX
inner = tg[cj*SUB + 1, ci*SUB + 1]; edge = tg[cj*SUB, ci*SUB]; far = tg[0, 0]
check('grid fraction: inside the frozen cell 0.4, its shared corner vertices 0.4 (smaller wins), a far vertex 1', abs(inner - 0.4) < 1e-12 and abs(edge - 0.4) < 1e-12 and far == 1.0, (inner, edge, far))
check('grid fraction: a vertex of the neighbour that is NOT shared with the frozen cell stays 1', tg[cj*SUB + 1, (ci + 1)*SUB + 1] == 1.0)

# ---------------------------------------------------------------- 2 blocked test (closest axis distance per cell)
xa = np.linspace(0.1, 12.0, 120); A = np.c_[xa, np.full(120, 0.5), np.zeros(120)]; B = A.copy(); B[:, 1] = np.where(xa < 3.0, 0.5 + 0.10, 0.5 + 1.0)       # yarn B 0.10 from A for x < 3, 1.0 away beyond
P = np.vstack([A, B]); yid = np.r_[np.zeros(120, int), np.ones(120, int)]; idx = np.r_[np.arange(120), np.arange(120)]
cell, uniq, nc = sf.cell_layout(P, SH); d = sf.cell_min_distance(P, yid, idx, cell, nc, R)
cx = (uniq % NX); left = d[cx == 0]; right = d[cx >= 4]
check('blocked test: closest distance 0.10 in the cells with the close pair, none (inf) in cells whose pairs are beyond reach', abs(left.min() - 0.10) < 1e-9 and np.isinf(right).all(), (left.min(), np.isinf(right).all()))
st0 = sf.FoldState.new(nc); new = sf.update_fractions(st0, 1, 0.1, d, 0.8*2*R)
check('blocked at stage 1 (threshold 0.8 x 2R = %.3f): cells with the close pair freeze at fraction 0 (flat); the others advance to 0.1' % (0.8*2*R), st0.frozen[cx == 0].any() and (st0.frac[st0.frozen] == 0).all() and np.allclose(st0.frac[~st0.frozen], 0.1), (int(st0.frozen.sum()), int(new.sum())))

# ---------------------------------------------------------------- 3 freezing logic over ten stages
N = 10; s3 = sf.FoldState.new(3); thr = 0.2
def dm(stage):       # cell 0 blocked from stage 5 on (its last clear fraction is 0.4); cell 1 never; cell 2 blocked from stage 8 on, clear again at stage 9 (must stay frozen)
    d = np.array([1.0, 1.0, 1.0]); d[0] = 0.05 if stage >= 5 else 1.0; d[2] = 0.05 if stage == 8 else 1.0; return d
for s in range(1, N + 1):
    t = s/N; s3.frac[~s3.frozen] = t; sf.update_fractions(s3, s, t, dm(s), thr)
check('freezing: the blocked cell stops at its last clear fraction 0.4 (stage 5), the second at 0.7 (stage 8), the unblocked neighbour reaches 1.0', np.allclose(s3.frac, [0.4, 1.0, 0.7]) and list(s3.frozen_stage) == [5, 0, 8], (s3.frac, s3.frozen_stage))
check('a frozen cell stays frozen even when it is clear again later, and an already frozen cell is not frozen twice', s3.frozen.tolist() == [True, False, True] and s3.frozen_stage[2] == 8)

# ---------------------------------------------------------------- 4 end to end with a stub relaxation: conservation per course at every stage, sign of the loop change, blocked cell
def stub_relax(sim_):          # moves every cell rigidly so that its mean is exactly the hold target (the L-BFGS relaxation with an infinitely stiff hold)
    sh_ = sim_.shell; P = np.vstack([y.X for y in sim_.yarns]); m = np.c_[[np.bincount(sh_['cell'], weights=P[:, k], minlength=len(sh_['cnt'])) for k in range(3)]].T/sh_['cnt'][:, None]
    o = 0
    for y in sim_.yarns:
        n = len(y.X); y.X += (sh_['tgt'] - m)[sh_['cell'][o:o + n]]; o += n
def run(sign, blocked_cell=None, stages=10):
    sim_ = make_sim(); c = sf.Ctx(sim_, SH, sign, 1.0, frame=False); state = sf.FoldState.new(c.nc); calls = [0]
    def dmin():
        calls[0] += 1; d_ = np.full(c.nc, 1.0)
        if blocked_cell is not None and calls[0] >= 5: d_[blocked_cell] = 0.01
        return d_
    recs = sf.run_stages(c, state, stages, 0.5, stub_relax, 0.8*2*R, dmin_fn=dmin); c.apply(state.targets(1.0), 1.0, 0.05); return sim_, c, state, recs
sim4, c4, s4, rec4 = run(-1.0)
check('per-course conservation of the total rest length at every one of the 10 stages (largest relative error %.1e)' % max(r_['conservation_err'] for r_ in rec4), all(r_['conservation_err'] < 1e-12 for r_ in rec4) and len(rec4) == 10)
check('the rest lengths really changed (otherwise the conservation test is empty)', max(float(np.abs(y.l0 - l).max()) for y, l in zip(sim4.yarns, c4.l0_start)) > 1e-3)
b4 = sf.measure_state(c4.flatX, c4.flatX, c4.l0_start, SH, c4.cell, c4.uniq); a4 = sf.measure_state(c4.flatX, [y.X for y in sim4.yarns], [y.l0 for y in sim4.yarns], SH, c4.cell, c4.uniq)
dl = a4['rest']/b4['rest'] - 1; ec, ew = sf.shell_strain_cells(SH, c4.Fx, c4.Fy)
check('shell strain along the courses is +6 % left and -6 % right (the synthetic shell)', abs(ec[:, :3].mean() - 0.06) < 5e-3 and abs(ec[:, 5:].mean() + 0.06) < 5e-3, (ec[:, :3].mean(), ec[:, 5:].mean()))
check('sign -1: tension cells (left) have SMALLER rest length, compression cells (right) LARGER (mean change %.3f left, %.3f right)' % (np.nanmean(dl[:, :3]), np.nanmean(dl[:, 5:])), np.nanmean(dl[:, :3]) < -0.03 and np.nanmean(dl[:, 5:]) > 0.03)
sim4p, c4p, _, _ = run(+1.0); dlp = sf.measure_state(c4p.flatX, [y.X for y in sim4p.yarns], [y.l0 for y in sim4p.yarns], SH, c4p.cell, c4p.uniq)['rest']/b4['rest'] - 1
check('sign +1 is the mirror image (tension cells larger, compression cells smaller)', np.nanmean(dlp[:, :3]) > 0.03 and np.nanmean(dlp[:, 5:]) < -0.03)
mean_cells = np.zeros((NY, NX, 3)); P4 = np.vstack([y.X for y in sim4.yarns]); tg4 = np.full((NY*NX, 3), np.nan); tg4[c4.uniq] = c4.tgt
check('stub relaxation puts the cell means on the hold targets', np.allclose(a4['M'].reshape(-1, 3)[c4.uniq], c4.tgt, atol=1e-9))
check('every cell reached the full fold when nothing is blocked', (s4.frac == 1.0).all() and not s4.frozen.any())
bc = 2*NX + 3; sim5, c5, s5, rec5 = run(-1.0, blocked_cell=bc)          # every stitch cell holds yarn, so the cell id equals its index
kc = int(np.where(c5.uniq == bc)[0][0]); nb = int(np.where(c5.uniq == bc + 1)[0][0])
check('end to end: the blocked cell froze at its last clear fraction 0.4, its neighbour continued to 1.0, conservation holds at every stage', abs(s5.frac[kc] - 0.4) < 1e-12 and s5.frozen[kc] and s5.frac[nb] == 1.0 and not s5.frozen[nb] and all(r_['conservation_err'] < 1e-12 for r_ in rec5), (s5.frac[kc], s5.frac[nb]))
tz = np.linalg.norm(c5.tgt[kc] - c5.P0[c5.cell == kc].mean(0)); fl = c5.P0[c5.cell == kc].mean(0)
check('the frozen cell target lies between the flat position and its fully folded position', 0 < np.linalg.norm(c5.tgt[kc] - fl) < np.linalg.norm(c4.tgt[kc] - fl) + 1e-9, (np.linalg.norm(c5.tgt[kc] - fl), np.linalg.norm(c4.tgt[kc] - fl)))

# ---------------------------------------------------------------- 5 the measurement on known data
M = np.zeros((4, 5, 3)); M[..., 0] = np.arange(5)[None, :]*1.5; M[..., 1] = np.arange(4)[:, None]*1.1; cs, ws = sf.spacings(M)
check('spacings of a regular grid: course spacing 1.1 and wale spacing 1.5 in every cell, edges included', np.allclose(cs, 1.1) and np.allclose(ws, 1.5))
Mh = M.copy(); Mh[0, 0] = np.nan; cs2, ws2 = sf.spacings(Mh); check('a missing cell gives nan for itself and uses the other neighbour for the cells next to it', np.isnan(cs2[0, 0]) and abs(cs2[1, 0] - 1.1) < 1e-12 and abs(ws2[0, 1] - 1.5) < 1e-12)
b = dict(rest=np.full((4, 6), 1.0), course=np.full((4, 6), 1.0), wale=np.full((4, 6), 1.0))
ecs = np.tile(np.array([0.05, -0.05]), (4, 3)); dls = -0.1*np.sign(ecs)                                   # row mean 0; loops smaller where the strain is positive
a = dict(rest=1 + dls, course=1 + 0.5*dls, wale=1 + 0.5*dls); tgt_sp = dict(course=a['course'], wale=a['wale'])
out, cells = sf.direction_check(b, a, tgt_sp, ecs, -dls, sign=-1.0)
R1 = out['R1_loop_vs_course_strain_minus_course_mean']; R2 = out['R2_course_spacing_vs_loop_length']; R3 = out['R3_course_spacing_vs_wale_strain']
check('R1 on a constructed case with the sign -1 rule: correlation -1, every cell agrees', R1['corr'] < -0.999 and R1['share_agree'] == 1.0 and R1['n'] == 24, R1)
check('R2: spacings move with the loop length: correlation +1, all agree', R2['corr'] > 0.999 and R2['share_agree'] == 1.0)
check('R3 fails when the strain is set opposite to the spacing change: share 0, the flag grid says 0 in every cell', R3['share_agree'] == 0.0 and np.nanmax(cells['agree_R3_course']) == 0.0, R3)
a2 = dict(a, rest=a['rest'].copy()); a2['rest'][0, 0] = 1 - dls[0, 0]; a2['rest'][2, 3] = 1 - dls[2, 3]                  # two cells with the wrong sign of the loop change
out2, cells2 = sf.direction_check(b, a2, tgt_sp, ecs, -dls, sign=-1.0)
check('two wrong cells are counted and flagged: share 22/24, flag 0 at exactly those cells', abs(out2['R1_loop_vs_course_strain_minus_course_mean']['share_agree'] - 22/24) < 1e-12 and cells2['agree_R1'][0, 0] == 0 and cells2['agree_R1'][2, 3] == 0 and np.nansum(cells2['agree_R1'] == 0) == 2)
small = dict(b, rest=np.full((4, 6), 1.0)); out3, _ = sf.direction_check(b, small, tgt_sp, ecs, -dls, sign=-1.0)
check('changes below the threshold are not counted (n = 0, share None)', out3['R1_loop_vs_course_strain_minus_course_mean']['n'] == 0 and out3['R1_loop_vs_course_strain_minus_course_mean']['share_agree'] is None)

# ---------------------------------------------------------------- 6 (a), (b), (c) on the synthetic case with the stub relaxation (reported, not asserted: the stub imposes the hold exactly)
tM = np.full((NY*NX, 3), np.nan); tM[c4.uniq] = c4.tgt; ts, tw = sf.spacings(tM.reshape(NY, NX, 3)); chk, cells4 = sf.direction_check(b4, a4, dict(course=ts, wale=tw), ec, ew, -1.0)
print('\nSYNTHETIC CASE (stub relaxation, sign -1, 10 stages): relations between the three quantities and the shell strain')
for k_, v_ in chk.items():
    if isinstance(v_, dict) and 'corr' in v_: print('  %-46s corr %s, share agree %s (n %d, expected sign %+d)' % (k_, None if v_['corr'] is None else '%+.3f' % v_['corr'], None if v_['share_agree'] is None else '%.3f' % v_['share_agree'], v_['n'], v_['expected_sign']))
print('  cells where R1 and both R2 agree:', chk['R1_R2_all_agree_share'], '; spacing against the shell target spacing, rms relative:', chk['spacing_vs_shell_target_rms'], '; loop length change:', chk['loop_length_change'])
check('(c) loop length follows the rule in the synthetic case (R1 share 1)', chk['R1_loop_vs_course_strain_minus_course_mean']['share_agree'] > 0.95)
cc = lambda d_, k_: d_[k_]['corr']
check('(a) and (b): the spacings follow the SHELL strain (R3 > 0), as the hold imposes, but go AGAINST the loop length with sign -1 (R2 < 0): the three cannot all agree (R1, R2, R3 shares of cells: 1, 0)', all(cc(chk, k_) is not None for k_ in ('R2_course_spacing_vs_loop_length', 'R2_wale_spacing_vs_loop_length', 'R3_course_spacing_vs_wale_strain', 'R3_wale_spacing_vs_course_strain')) and cc(chk, 'R2_course_spacing_vs_loop_length') < 0 and cc(chk, 'R2_wale_spacing_vs_loop_length') < 0 and cc(chk, 'R3_course_spacing_vs_wale_strain') > 0 and cc(chk, 'R3_wale_spacing_vs_course_strain') > 0 and chk['R1_R2_all_agree_share'] == 0.0, {k_: cc(chk, k_) for k_ in chk if isinstance(chk[k_], dict) and 'corr' in chk[k_]})
chkp, _ = sf.direction_check(b4, sf.measure_state(c4p.flatX, [y.X for y in sim4p.yarns], [y.l0 for y in sim4p.yarns], SH, c4p.cell, c4p.uniq), dict(course=ts, wale=tw), ec, ew, +1.0)
check('with sign +1 all three relations agree (R1, both R2, both R3 correlations > 0; share of cells where R1 and both R2 agree: %s)' % chkp['R1_R2_all_agree_share'], all(cc(chkp, k_) > 0 for k_ in ('R1_loop_vs_course_strain_minus_course_mean', 'R2_course_spacing_vs_loop_length', 'R2_wale_spacing_vs_loop_length', 'R3_course_spacing_vs_wale_strain', 'R3_wale_spacing_vs_course_strain')) and chkp['R1_R2_all_agree_share'] > 0.9)

# ---------------------------------------------------------------- 7 files and aggregation
import tempfile, json, io, csv
with tempfile.TemporaryDirectory() as td:
    dmin_f = np.full(c5.nc, 1.0); dmin_f[kc] = 0.01
    sf.write_fraction_csv(os.path.join(td, 'fold_fraction.csv'), s5, c5.uniq, NX, dmin_f, 0.8)
    rows = list(csv.DictReader(open(os.path.join(td, 'fold_fraction.csv')))); rr = [r_ for r_ in rows if r_['frozen'] == '1']
    check('fold_fraction.csv: one row per cell, the frozen cell with fraction 0.4, its stage and blocked_at_end', len(rows) == NX*NY and len(rr) == 1 and abs(float(rr[0]['fraction']) - 0.4) < 1e-6 and rr[0]['frozen_at_stage'] == '5' and rr[0]['blocked_at_end'] == '1', rr)
    sm = sf.summary_dict(s5, dmin_f, 0.8, rec5, R, dict(state='stages only')); json.dump(sm, open(os.path.join(td, 'fold_summary.json'), 'w'), default=sf._jd)
    buf = io.StringIO(); sf.aggregate([td], buf); line = list(csv.DictReader(io.StringIO(buf.getvalue())))[0]
    check('summary and aggregation: share frozen 1/48, minimum fraction 0.4, schema, unsourced list present', sm['schema'] == sf.SCHEMA and abs(float(line['share_frozen']) - 1/48) < 1e-9 and abs(float(line['min_fraction']) - 0.4) < 1e-9 and int(line['n_frozen']) == 1 and len(sm['unsourced']) > 5, line)
    sf.write_cells_csv(os.path.join(td, 'c.csv'), cells4, NX, NY); check('fold_cell_metrics.csv has one row per cell and all columns', len(list(csv.DictReader(open(os.path.join(td, 'c.csv'))))) == NX*NY)

print('ALL OK' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
