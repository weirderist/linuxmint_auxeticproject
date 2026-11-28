#!/usr/bin/env python3
"""Tests of scripts/shell_learning/shell_proposer_eval.py (proposer against random) and scripts/shell_learning/shell_rules_sr.py (open-ended rule search) on SYNTHETIC finished runs: no solver, numpy only, temporary folders.
Run from the repository root: python tests/test_shell_eval_sr.py   (a few minutes: the rule search runs its permutation test)"""
import json, os, subprocess, sys, tempfile, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='shelleval_')
os.environ.update(SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs
import shell_campaign as sc, shell_features as sf, shell_proposer as pr, shell_proposer_eval as pe, shell_rules_sr as sr, campaign as cm
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info), flush=True); ok or bad.append(name)
t0 = time.time()

# ---- feature generator on known patterns
P1 = sc.matrix_from_text(open(os.path.join(ROOT, 'campaigns', 'pattern1_matrix.txt')).read())
f1 = sr.pattern_features(dict(matrix=P1, tiles=1, cpi=22.0, wpi=16.0))
check('Pattern 1 (tiles 1): strong-majority wales come in groups of 2 (two-wale TM axes) and successive groups alternate in type', f1['wale_tm_group_mean'] == 2.0 and f1['wale_tm_group_alt'] == 1.0,
      'group mean %s, alternation %s' % (f1['wale_tm_group_mean'], f1['wale_tm_group_alt']))
cb = [[1 + (i + j) % 2 for j in range(8)] for i in range(8)]       # 8 x 8 checkerboard: 7 x 7 = 49 windows of 2 x 2, 25 start with digit 1, 24 with digit 2
fc = sr.pattern_features(dict(matrix=cb, tiles=1, cpi=22.0, wpi=16.0))
check('checkerboard: only the two alternating 2x2 motifs, runs of length 1, every diagonal a strong-majority (diagonal TM) axis, no wale or course axes',
      abs(fc['m2x2_12/21'] - 25 / 49) < 1e-12 and abs(fc['m2x2_21/12'] - 24 / 49) < 1e-12 and abs(fc['m2x2_12/21'] + fc['m2x2_21/12'] - 1) < 1e-12 and fc['run_wale_mean'] == 1.0 and fc['run_course_mean'] == 1.0 and fc['tm_diag_share'] == 1.0 and fc['tm_wale_share'] == 0.0)
check('motif frequencies of each window type sum to 1 (16 + 8 + 8 motifs)', all(abs(sum(v for k, v in f1.items() if k.startswith(p)) - 1) < 1e-12 for p in ('m2x2_', 'm3x1_', 'm1x3_'))
      and sum(k.startswith('m2x2_') for k in f1) == 16 and sum(k.startswith('m3x1_') for k in f1) == 8)
st = [[1] * 6] * 3 + [[2] * 6] * 3; fs = sr.pattern_features(dict(matrix=st, tiles=1, cpi=22.0, wpi=16.0))
check('two horizontal stripes (3 courses each): sections 6 wide and 3 high, wale runs of 3, course runs of 6, all stitches in edge-touching sections',
      fs['sec_width_mean'] == 6 and fs['sec_height_mean'] == 3 and fs['run_wale_mean'] == 3 and fs['run_course_mean'] == 6 and fs['sec_edge_stitch_frac'] == 1.0)

# ---- a synthetic seeded campaign with 130 finished runs (same construction as tests/test_shell_learning.py)
r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'campaigns', 'campaign.py'), 'shell-plan', 'syn', '--policy', os.path.join(ROOT, 'campaigns', 'shell_policy.json'), '--filler', '60'], capture_output=True, text=True, env=dict(os.environ))
c = sc.Camp('syn'); rng = np.random.default_rng(1); items = [x for x in c.queue_items() if not x.get('control')][:130]
def metrics_for(s):
    ff = sf.features(s['matrix'], s['tiles'], s['cpi'], s['wpi']); fx, fy = 76.2, 69.3
    th = (0.10 + 0.20 * ff['tm_wale_share'] - 0.15 * ff['dilution_wale'] + 0.05 * ff['log_ratio'] + 0.01 * rng.standard_normal()) * np.sqrt(fx * fy)
    return dict(thickness=float(th), flat_x=fx, flat_y=fy, shrink_x=float(-0.5 + 0.1 * ff['knit_fraction'] + 0.01 * rng.standard_normal()), shrink_y=float(-0.1 - 0.2 * ff['boundary_density'] + 0.01 * rng.standard_normal()),
                fold_turn_x_rad=float(0.5 + 0.3 * ff['axis_direction'] + 0.02 * rng.standard_normal()), fold_turn_y_rad=0.01, wave_amp_median_mm=float(1 + 2 * ff['sections_per_100'] / 25))
for x in items: sc.append_line(c.index, dict(id=x['id'], status='done', t=cm.now(), metrics=metrics_for(x['settings']), wall_s=100.0))
rows = [r_ for r_ in sf.dataset(c) if not r_['control']]
check('synthetic campaign: 130 finished non-control runs', r.returncode == 0 and len(rows) == 130, (r.stdout + r.stderr)[-150:])
nq = len(c.queue_items()); ni = len(sc.read_lines(c.index))

# ---- proposer evaluation
check('proposer eval waits below 100 runs', pe.evaluate_rows(rows[:80])['status'] == 'waiting')
ev = pe.evaluate_rows(rows, repeats=10)
check('proposer eval: ok, budgets below the pool size, mean and SE per budget for both ways and every target, one verdict of the three', ev['status'] == 'ok' and ev['verdict_short'] in pe.VERDICTS and ev['budgets'] and max(ev['budgets']) < ev['n_pool']
      and all(set(b['proposer']) == set(pr.PROP_TARGETS) and all('se' in v for v in b['random'].values()) for b in ev['per_budget']), ev.get('verdict', ev.get('note')))
print('      (synthetic data, information only) %s' % ev['verdict'])
check('held-out split depends only on the run id: same runs held out with fewer runs, none of them in the pool', {r_['run'] for r_ in rows if pe.is_test(r_['run'], 0.25)} >= {r_['run'] for r_ in rows[:100] if pe.is_test(r_['run'], 0.25)} and ev['n_test'] + ev['n_pool'] == 130)
rr = pe.evaluate_rows(rows, repeats=10, rule_a=pe.random_rule)
check('control: random against random is judged "no different from random"', rr['verdict_short'] == 'no different from random', rr['verdict'])
def clump_rule(Xsel, Xrest, k, rng):                      # deliberately bad: always the candidates the model is MOST sure about (clones of what it has)
    out = []; X = Xsel.copy(); av = list(range(len(Xrest)))
    for _ in range(k):
        s = pr.posterior_std(X, Xrest[av], 1.4, 1e-2); j = av[int(np.argmin(s))]; out.append(j); av.remove(j); X = np.vstack([X, Xrest[j][None]])
    return out
bw = pe.evaluate_rows(rows, repeats=10, rule_a=clump_rule)
check('control: a deliberately bad rule (picks near-duplicates) is judged "worse than random"', bw['verdict_short'] == 'worse than random', bw['verdict'])
res, path = pe.run(c, repeats=5)
check('pe.run writes analysis/proposer_eval.json in the campaign folder', os.path.isfile(path) and json.load(open(path))['verdict_short'] in pe.VERDICTS and path.endswith(os.path.join('analysis', 'proposer_eval.json')))

# ---- rule search
FAST = dict(boots=50, perms=10)
X0, names0, notes = sr.feature_table(rows)
check('feature table: more than 60 base features, all finite; constant settings are still listed here and dropped per target', X0.shape == (130, len(names0)) and len(names0) > 60 and np.all(np.isfinite(X0)), '%d features' % len(names0))
def with_target(rows_, fn, name='thickness_rel'):
    return [dict(r_, targets=dict(r_['targets'], **{name: fn(r_)})) for r_ in rows_]
rngn = np.random.default_rng(7)
# planted interaction of two features that VARY in this synthetic campaign (tm_wale_share sd 0.27, log_ratio sd 0.24). An earlier planted rule 0.5 knit_fraction x tiles was not identifiable here:
# 108 of 130 runs have tiles 4 (support 22) and knit_fraction has sd 0.046, so the product is almost linear; the guards correctly refused it (support, beats_linear).
# the planted-rule checks test the search machinery on INDEPENDENT runs (each its own family, as the 600 random runs of shell2); with ~10 seed families only, the held-out set is whole unseen
# families (task 53a) and the search did not recover the planted term on 2026-11-28 (a real limit of so few families, recorded in CLAUDE.md task 53)
indep = [dict(r_, source='random:test') for r_ in rows]
planted = with_target(indep, lambda r_: 0.5 * r_['features']['tm_wale_share'] * r_['features']['log_ratio'] + 0.005 * rngn.standard_normal())
t = sr.search_target(planted, 'thickness_rel', X0, names0, FAST); rule = t.get('rule') or t.get('candidate') or {}
pt = [x for x in rule.get('terms', []) if 'tm_wale_share' in x['term'] and 'log_ratio' in x['term'] and '*' in x['term']]
check('planted rule y = 0.5 tm_wale_share x log_ratio (+ noise) is REPORTED with the product term, coefficient within 10 % of 0.5, and beats the linear rules on the held-out runs',
      t['status'] == 'rule' and len(pt) == 1 and abs(pt[0]['coefficient'] - 0.5) < 0.05 and rule['heldout_mse'] < t['heldout_mse_linear_rules'],
      '%s | %s | failed %s' % (t['status'], rule.get('formula'), t.get('failed_guards')))
check('the planted product is flagged as a NEW term (not only existing linear-rule features)', rule.get('only_existing_features') is False and any('*' in x for x in rule.get('new_terms', [])), str(rule.get('new_terms')))
kt = sr.search_target(with_target(rows, lambda r_: 0.5 * r_['features']['knit_fraction'] * r_['features']['tiles'] + 0.005 * rngn.standard_normal()), 'thickness_rel', X0, names0, FAST)
check('a planted product of two features that hardly vary (knit_fraction x tiles, tiles 4 in 108 of 130 runs) is NOT reported as a new rule (guards)', kt['status'] == 'no rule', str(kt['failed_guards']))
check('reported rule has stability >= 0.75 per term, supporting runs, a permutation FDR estimate and the label', t['status'] == 'rule' and all(x['stability'] >= 0.75 and x['supporting_runs'] >= 20 for x in rule['terms'])
      and t['permutation']['fdr_estimate'] is not None and t['label'] == sr.LABEL, str(t['permutation']))
noise_status = []
for k in range(3):
    rz = np.random.default_rng(100 + k); tn = sr.search_target(with_target(rows, lambda r_: float(rz.standard_normal())), 'thickness_rel', X0, names0, dict(FAST, seed=k)); noise_status.append((tn['status'], tn['failed_guards'], tn['permutation']['mean_stable_terms_by_chance']))
check('pure noise targets (3 draws): no rule is reported', all(s == 'no rule' for s, _, _ in noise_status), str(noise_status))
lin = sr.search_target(rows, 'thickness_rel', X0, names0, FAST)
check('a target that IS linear in the existing features (the synthetic thickness_rel) gives no new rule: it does not beat the linear rules', lin['status'] == 'no rule' and 'beats_linear' in lin['failed_guards'], '%s %s' % (lin['status'], lin['failed_guards']))

# settings and the plug-in
tz = [dict(r_, settings=dict(r_['settings'], tightness_ratio=float([0.707, 1.0, 1.225][i % 3]))) for i, r_ in enumerate(rows)]
Xt, nt, _ = sr.feature_table(tz)
check('a new numeric setting (tightness_ratio) carried by the runs is picked up as set_tightness_ratio; absent from all runs it does not exist', 'set_tightness_ratio' in nt and 'set_tightness_ratio' not in names0)
tp = [dict(r_, settings=dict(r_['settings'], tightness_ratio=1.1)) if i < 50 else r_ for i, r_ in enumerate(rows)]; _, npart, notep = sr.feature_table(tp)
check('present in only some runs and without a default: dropped and listed', 'set_tightness_ratio' not in npart and any('set_tightness_ratio' in d for d in notep['dropped_features']))
ttr = with_target(tz, lambda r_: 0.3 * r_['settings']['tightness_ratio'] ** 2 * r_['features']['knit_fraction'] + 0.002 * rngn.standard_normal())
tt = sr.search_target(ttr, 'thickness_rel', Xt, nt, dict(FAST, perms=30)); rl_ = tt.get('rule') or tt.get('candidate') or {}          # perms 30 = the default; 10 make the FDR estimate too noisy near its 0.25 limit
check('a planted effect of the new setting is found (a reported rule with a set_tightness_ratio term)', tt['status'] == 'rule' and any('set_tightness_ratio' in x['term'] for x in rl_.get('terms', [])), '%s | %s' % (tt['status'], rl_.get('formula')))
uf = os.path.join(TMP, 'user_feats.py')
open(uf, 'w').write('import numpy as np\nSETTING_DEFAULTS = {"tightness_ratio": 1.0}\ndef feat_purl_rows(T, ctx):\n    return float((T == 2).all(axis=1).mean()) + 0.001 * ctx["tiles"]\n'
                    'def feat_broken(T, ctx):\n    raise RuntimeError("deliberate")\n')
Xu, nu_, noteu = sr.feature_table(tp, uf)
check('plug-in: feat_purl_rows is picked up as user_purl_rows, a failing feature is dropped and listed, SETTING_DEFAULTS fills a partly present setting',
      'user_purl_rows' in nu_ and 'user_broken' not in nu_ and any('user_broken' in d for d in noteu['dropped_features']) and 'set_tightness_ratio' in nu_, str(noteu)[:200])

# whole job
out = sr.run_rows(rows, ('shrink_x', 'wave_amp_rel'), FAST)
check('run_rows: every target has a status and the label; waiting below 100 runs', all(x['status'] in ('rule', 'no rule') and x['label'] == sr.LABEL for x in out['targets']) and sr.run_rows(rows[:60], ('shrink_x',), FAST)['targets'][0]['status'] == 'waiting')
check('nothing written to the campaign: queue and index unchanged', len(c.queue_items()) == nq and len(sc.read_lines(c.index)) == ni)
print('\n(%.0f s)  %s' % (time.time() - t0, 'ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
