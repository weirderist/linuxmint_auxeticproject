"""test_shell_policy_v2.py - the shell2 policy (campaigns/shell2_policy.json) and its plan generator scripts/campaigns/shell_policy.py build_plan_v2 (2026-11-27). numpy not needed.
Run from the repository root:  python tests/test_shell_policy_v2.py
Checks: plan size and mix; balanced prefixes (every 100 runs have the target mix, R and density cover their ranges); every entry passes normalise and the policy; the policy refuses a hinge / L-BFGS
entry and a matrix side over 30; normalise rejects an unknown solver and keeps shell1 settings (no new keys) unchanged; the same seed gives the same plan."""
import collections, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'campaigns'))
import shell_policy as sp, shell_campaign as sc

fails = []
def check(msg, ok, extra=''):
    print(('PASS  ' if ok else 'FAIL  ') + msg + (('   ' + str(extra)) if extra != '' else '')); ok or fails.append(msg)

pol = sp.load(os.path.join(ROOT, 'campaigns', 'shell2_policy.json')); sd = os.path.join(ROOT, 'campaigns', 'shell_seeds')
seeds = [dict(id=x['id'], matrix=sc.matrix_from_text(open(os.path.join(sd, x['file'])).read())) for x in json.load(open(os.path.join(sd, 'seeds.json')))['seeds']]
plan = sp.build_plan_v2(pol, seeds, 20261127, 1000); gen = [e for e in plan if e['tier'] == 2]
cat = lambda e: e['source'].split(':')[1] if e['source'].startswith('random') else 'edits'
check('plan = seeds + 1000 generated runs', len(gen) == 1000 and sum(e['tier'] == 0 for e in plan) == len(seeds), (len(plan), len(seeds)))
c = collections.Counter(cat(e) for e in gen); check('mix 400 blocks / 200 pixels / 400 edits', c == {'blocks': 400, 'pixels': 200, 'edits': 400}, dict(c))
ok = True; lo, hi = pol['ranges']['ratio_cpi_over_wpi']; dlo, dhi = pol['ranges']['density_range']
for k in range(0, 1000, 100):
    blk = gen[k:k + 100]; cc = collections.Counter(cat(e) for e in blk); R = [e['settings']['cpi'] / e['settings']['wpi'] for e in blk]; S = [e['settings']['cpi'] * e['settings']['wpi'] for e in blk]
    ok &= cc == {'blocks': 40, 'pixels': 20, 'edits': 40} and min(R) < lo * 1.05 and max(R) > hi / 1.05 and min(S) < dlo + 0.05 * (dhi - dlo) and max(S) > dhi - 0.05 * (dhi - dlo)
check('every block of 100 runs has the target mix and covers R and CPI x WPI', ok)
bad = [e['label'] for e in plan if sp.violations(sc.normalise(e['settings'], pol['ranges']['max_stitches']), pol)]
check('every entry passes normalise and the policy', not bad, bad[:3])
check('seed edits change the size of some matrices', any((len(e['settings']['matrix']), len(e['settings']['matrix'][0])) != (len(s['matrix']), len(s['matrix'][0])) for e in gen if cat(e) == 'edits' for s in seeds if e['source'] == 'edit:' + s['id']))
check('random sides within 6-30, stitches within the cap', all(6 <= len(e['settings']['matrix']) <= 30 and 6 <= len(e['settings']['matrix'][0]) <= 30 and sp.n_stitches(e['settings']) <= 3600 for e in gen))
s0 = dict(gen[0]['settings']); check('policy refuses bending hinge', any('bending' in v for v in sp.violations(dict(s0, bending='hinge'), pol)))
check('policy refuses solver lbfgs', any('solver' in v for v in sp.violations(dict(s0, solver='lbfgs'), pol)))
check('policy refuses a 31-row matrix', any('matrix' in v for v in sp.violations(dict(s0, matrix=[[1, 2] * 3] * 31, tiles=2), pol)))
try: sc.normalise(dict(s0, solver='adam')); check('normalise rejects an unknown solver', False)
except ValueError: check('normalise rejects an unknown solver', True)
s1 = dict(matrix=[[1, 2], [2, 1]], tiles=3, cpi=22, wpi=16); n1 = sc.normalise(s1)
check('shell1-type settings get no new keys (hash unchanged)', not any(k in n1 for k in ('bending', 'solver', 'check_stability')))
check('same seed, same plan', [e['settings'] for e in sp.build_plan_v2(pol, seeds, 20261127, 1000)] == [e['settings'] for e in plan])
print('ALL OK' if not fails else 'FAILED: %d' % len(fails)); sys.exit(1 if fails else 0)
