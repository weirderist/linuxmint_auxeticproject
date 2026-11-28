#!/usr/bin/env python3
"""Tests of the pattern features (shell_features.py), the rule job (shell_rules.py) and the proposer (shell_proposer.py) on SYNTHETIC finished runs: no solver, numpy only, temporary folders.
Run from the repository root: python tests/test_shell_learning.py"""
import math, json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='shelllearn_')
os.environ.update(SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs
import shell_campaign as sc, shell_features as sf, shell_rules as rl, shell_proposer as pr, shell_policy as sp, campaign as cm
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)

# ---- features
cols = [[1, 2, 1, 2]] * 4                       # alternating columns: every wale is pure
f = sf.features(cols, 1, 22.0, 16.0)
check('pure wales: tm_wale_share 1, courses mixed: tm_course_share 0, axis_direction +1, no dilution', f['tm_wale_share'] == 1.0 and f['tm_course_share'] == 0.0 and f['axis_direction'] == 1.0 and f['dilution_wale'] == 0.0 and f['knit_fraction'] == 0.5)
ft = sf.features([list(r) for r in zip(*cols)], 1, 22.0, 16.0)
check('the transposed pattern (pure courses) gives axis_direction -1', ft['tm_course_share'] == 1.0 and ft['tm_wale_share'] == 0.0 and ft['axis_direction'] == -1.0)
blocks = [[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 1, 1], [2, 2, 1, 1]]
check('2 x 2 block pattern: 4 sections of 4 stitches (25 per 100 stitches ... 25.0), boundary density, ln ratio', sf.sections(np.array(blocks)) == [4, 4, 4, 4] and abs(sf.features(blocks, 1, 22.0, 16.0)['sections_per_100'] - 25.0) < 1e-9 and abs(sf.features(blocks, 1, 22.0, 16.0)['log_ratio'] - np.log(22 / 16)) < 1e-12)
check('tiling the 2 x 2 blocks by 2 merges nothing wrongly: 4 x 4 blocks of the tiled sheet, 16 regions / ... diagonals stay separate', len(sf.sections(sf.tile(blocks, 2))) >= 8)
P1 = sc.matrix_from_text(open(os.path.join(ROOT, 'campaigns', 'pattern1_matrix.txt')).read()); fp = sf.features(P1, 3, 22.0, 16.0)
check('Pattern 1 (tiles 3): knit fraction 0.5 and exactly 4 of 16 wales per tile are strong-majority axes (wales 6, 7, 14, 15 as found independently in tests/NOTES_loop_grade.md), no course axes', abs(fp['knit_fraction'] - 0.5) < 1e-9 and abs(fp['tm_wale_share'] - 0.25) < 1e-9 and fp['tm_course_share'] == 0.0 and fp['axis_direction'] == 0.25, str({k: round(v, 3) for k, v in fp.items()}))
check('targets are scale-free and None when a descriptor is missing', sf.targets(dict(thickness=10.0, flat_x=100.0, flat_y=100.0))['thickness_rel'] == 0.1 and sf.targets(dict())['thickness_rel'] is None)

# ---- a synthetic seeded campaign with 130 finished runs whose targets depend on the features
pol = sp.load(os.path.join(ROOT, 'campaigns', 'shell_policy.json'))
r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'campaigns', 'campaign.py'), 'shell-plan', 'syn', '--policy', os.path.join(ROOT, 'campaigns', 'shell_policy.json'), '--filler', '60'], capture_output=True, text=True, env=dict(os.environ))
c = sc.Camp('syn'); rng = np.random.default_rng(1); items = [x for x in c.queue_items() if not x.get('control')][:130]
def metrics_for(s):
    ff = sf.features(s['matrix'], s['tiles'], s['cpi'], s['wpi']); flat_x, flat_y = 76.2, 69.3
    th = (0.10 + 0.20 * ff['tm_wale_share'] - 0.15 * ff['dilution_wale'] + 0.05 * ff['log_ratio'] + 0.01 * rng.standard_normal()) * np.sqrt(flat_x * flat_y)
    return dict(thickness=float(th), flat_x=flat_x, flat_y=flat_y, shrink_x=float(-0.5 + 0.1 * ff['knit_fraction'] + 0.01 * rng.standard_normal()), shrink_y=float(-0.1 - 0.2 * ff['boundary_density'] + 0.01 * rng.standard_normal()),
                fold_turn_x_rad=float(0.5 + 0.3 * ff['axis_direction'] + 0.02 * rng.standard_normal()), fold_turn_y_rad=0.01, wave_amp_median_mm=float(1 + 2 * ff['sections_per_100'] / 25))
for x in items[:130]: sc.append_line(c.index, dict(id=x['id'], status='done', t=cm.now(), metrics=metrics_for(x['settings']), wall_s=100.0))
sc.append_line(c.index, dict(id=[y['id'] for y in c.queue_items() if y.get('control')][0], status='done', t=cm.now(), metrics=metrics_for(items[0]['settings']), wall_s=1.0))
rows = [r_ for r_ in sf.dataset(c) if not r_['control']]
check('the synthetic campaign has 130 finished non-control runs with features and targets', r.returncode == 0 and len(rows) == 130 and all(v is not None for v in rows[0]['targets'].values()), (r.stdout + r.stderr)[-150:])

# ---- rule job
few = rl.analyse_target(rows[:60], 'thickness_rel'); check('fewer than 100 runs: no rule is reported (status waiting)', few['status'] == 'waiting' and 'lasso' not in few)
res = rl.analyse_target(rows, 'thickness_rel'); la = res['lasso']; top = [x['feature'] for x in la['coefficients']][:3]
check('lasso finds the generating features of thickness_rel (tm_wale_share, dilution_wale, log_ratio) with a high cross-validated R^2', res['status'] == 'ok' and 'tm_wale_share' in top and la['cv_r2'] > 0.6, 'top %s, cv R2 %.3f, selected %d' % (top, la['cv_r2'], la['n_selected']))
check('every coefficient has a bootstrap stability and the rule carries the not-confirmed-on-real-fabric label', all(0 <= x['stability'] <= 1 for x in la['coefficients']) and res['label'] == rl.LABEL and la['support'] == 130)
check('the tree cross-check gives readable rules and its own cross-validated R^2 (CV by seed family since task 53a: here the 130 runs are ~10 seed families, so the tree must predict unseen families: it is reported with its honest R^2, not required to be good; it was > 0.2 with row folds, which leaked siblings)', res['tree']['rules'] and all(isinstance(t, str) for t in res['tree']['rules']) and math.isfinite(res['tree']['cv_r2']), 'cv R2 %.3f (measured -0.23 on 2026-11-28: the tree does not carry over to unseen families; the lasso does, R2 0.96); %s' % (res['tree']['cv_r2'], res['tree']['rules'][0]))
fam = [sf.family(r_['source'], r_['run']) for r_ in rows]; fo = rl.folds(len(rows), 5, 0, fam)
check('task 53a: the CV folds never split a seed family (every family in one fold) and every fold has runs', all(len(f) for f in fo) and all(len({fi for fi, f in enumerate(fo) if j in set(f.tolist())}) == 1 for j in range(len(rows))) and all(len({fam[j] for j in f} & {fam[j] for f2 in fo if f2 is not f for j in f2}) == 0 for f in fo), '%d families' % len(set(fam)))
check('task 53a: family() = seed:<id> for seeds and their edits / variants / fillers, the run id for a random pattern', sf.family('edit:seed03', 'S1') == 'seed:seed03' and sf.family('seed:seed03', 'S2') == 'seed:seed03' and sf.family('variant:seed03', 'S3') == 'seed:seed03' and sf.family('random:blocks', 'S4') == 'S4' and sf.family('proposer:gp', 'S5') == 'S5')
check('task 53a: with no families (every run its own) the folds are exactly the old row folds', all((a_ == b_).all() for a_, b_ in zip(rl.folds(50, 5, 0, [str(i) for i in range(50)]), rl.folds(50, 5, 0))))
step = [dict(r_, targets=dict(r_['targets'], shrink_x=0.3 if r_['features']['axis_direction'] > 0 else -0.3)) for r_ in rows]
trr = rl.analyse_target(step, 'shrink_x')['tree']; check('a step function of axis_direction is found by the tree (first split on axis_direction)', 'axis_direction' in trr['rules'][0], trr['rules'][0])
out = rl.run(c); rj = json.load(open(os.path.join(c.d, 'analysis', 'rules.json')))
check('rl.run writes analysis/rules.json and a numbered copy, excludes controls (130 runs), keeps the label', out['n_finished'] == 130 and rj['label'] == rl.LABEL and os.path.exists(os.path.join(c.d, 'analysis', 'rules_000130.json')) and all(t_['status'] == 'ok' for t_ in rj['targets']))
check('rules do not change the campaign: queue and index untouched by the job', len(c.queue_items()) == 11 + 2 + 91 + 60)

# ---- Gaussian process and proposer
rngp = np.random.default_rng(2); X = rngp.uniform(-1.5, 1.5, (60, 3)); y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.02 * rngp.standard_normal(60); m = pr.gp_fit(X, y); Xs = rngp.uniform(-1.5, 1.5, (200, 3)); mean, sd = pr.gp_predict(m, Xs)
ys = np.sin(Xs[:, 0]) + 0.5 * Xs[:, 1] ** 2
check('GP recovers a smooth function of 3 inputs from 60 points (RMSE < 0.15)', np.sqrt(np.mean((mean - ys) ** 2)) < 0.15, 'rmse %.3f, ls %.1f noise %g' % (np.sqrt(np.mean((mean - ys) ** 2)), m['ls'], m['noise']))
check('predictive std is small at training points and large far away', pr.gp_predict(m, X[:5])[1].max() < pr.gp_predict(m, np.array([[9.0, 9.0, 9.0]]))[1].min())
Xtr = rngp.standard_normal((20, 3)); Xc = rngp.standard_normal((100, 3)); ch = pr.select_batch(Xtr, Xc, 5)
dmin = min(np.linalg.norm(Xc[a] - Xc[b]) for i, a in enumerate(ch) for b in ch[i + 1:])
check('batch selection picks 5 distinct candidates that are not clones of each other', len(set(ch)) == 5 and dmin > 0.3, 'min distance %.2f' % dmin)
ev = pr.evaluate(rows, repeats=6, seed=0, budgets=(20, 40, 60, 80)); wins = sum(1 for b in ev.values() if b['gp_rmse'] <= b['random_rmse'] * 1.05)
check('fixed test against random sampling: uncertainty sampling is not worse than random at 3 of 4 budgets', wins >= 3, str({k: (round(v['random_rmse'], 4), round(v['gp_rmse'], 4)) for k, v in ev.items()}))
# The checks below exercise the GP selection itself on 130 synthetic runs. The proposer gate (800 finished original runs, K 23 Nov 2026) would keep it closed here, so it is lowered to 100 for
# this section only; the gate and the hard cap are tested in tests/test_shell_proposer_gate.py.
pr.PROPOSER_MIN_ORIGINALS_FINISHED = 100
c2 = sc.Camp('syn2'); sc.Camp.create('syn2', c.spec_()); k2 = sc.Camp('syn2')
rk2 = pr.propose(k2, 3); check('proposer proposes nothing below 100 results (empty campaign: the gate is closed)', rk2['proposed'] == [] and rk2['status'] != 'proposed', rk2['status'])
rp = pr.propose(c, 3, dry=True); check('dry run proposes nothing into the queue', rp['status'] == 'dry run' and len(c.queue_items()) == 11 + 2 + 91 + 60)
before = len(c.queue_items()); rp = pr.propose(c, 3); new = c.queue_items()[before:]
check('proposer enqueues 3 runs at tier 5 (ahead of the filler) through the policy-enforcing enqueue', rp['status'] == 'proposed' and len(new) == 3 and all(x['tier'] == 5 and x['source'] == 'proposer:gp' for x in new))
check('every proposed run passes the approved policy and ESSENTIAL settings are untouched', all(sp.violations(sc.normalise(x['settings'], 12000), pol) == [] and all(x['settings'][k] == v for k, v in pol['essential'].items()) for x in new))
check('with 2 or more proposals pending the proposer adds no more', pr.propose(c, 3)['status'] == 'enough proposals queued')
tiers_now = [x['tier'] for x in c.pending()]
check('pending order is by tier: the 3 proposals (tier 5) come before every filler run (tier 9)', tiers_now == sorted(tiers_now) and tiers_now.count(5) == 3 and tiers_now.index(5) < tiers_now.index(9), str(sorted(set(tiers_now))))
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
