#!/usr/bin/env python3
"""Tests of the pure-shell policy (scripts/campaigns/shell_policy.py, campaigns/shell_policy.json) and its enforcement in scripts/campaigns/shell_campaign.py, WITHOUT the solver (shell_run.py stub).
Temporary folders only. Run from the repository root: python tests/test_shell_policy.py"""
import json, math, os, random, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='shellpolicy_')
os.environ.update(SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), SHELL_RUN_STUB='1', CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs
import shell_campaign as sc, shell_policy as sp, campaign as cm
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)
def raises(fn, exc=ValueError):
    try: fn()
    except exc: return True
    return False

pol = sp.load(os.path.join(ROOT, 'campaigns', 'shell_policy.json')); sj = json.load(open(os.path.join(ROOT, 'campaigns', 'shell_seeds', 'seeds.json')))
base = os.path.join(ROOT, 'campaigns', 'shell_seeds')
seeds = [dict(id=x['id'], matrix=sc.matrix_from_text(open(os.path.join(base, x['file'])).read())) for x in sj['seeds']]
P1 = next(s for s in seeds if s['id'] == 'pattern1')['matrix']
check('policy file: ESSENTIAL = the verified 13 Nov values, tiles FREE 3 to 6, ratio 0.7 to 1.6 (K, 2026-11-21)', pol['essential'] == dict(kappa_scale=1.0, h_scale=1.0, nu=0.4, springs=0.0, sub=2, steps=15, iter=400, seed=0) and pol['ranges']['tiles'] == [3, 6] and pol['ranges']['ratio_cpi_over_wpi'] == [0.7, 1.6] and 'tiles' in pol['free'])
check('the 10 seeds are the 9 PNG matrices and Pattern 1 (20 x 16)', len(seeds) == 10 and (len(P1), len(P1[0])) == (20, 16) and all(len(s['matrix']) == x['rows'] for s, x in zip(seeds, sj['seeds'])))
check('pattern1 seed equals campaigns/pattern1_matrix.txt', P1 == sc.matrix_from_text(open(os.path.join(ROOT, 'campaigns', 'pattern1_matrix.txt')).read()))

def S(**kw):
    d = dict(sp._setting(pol, P1, 22.0, 16.0, 3)); d.update(kw); return sc.normalise(d, 10**6)
check('K\'s fabric at the success settings violates nothing', sp.violations(S(), pol) == [])
for k, v in (('kappa_scale', 1.1), ('h_scale', 0.9), ('nu', 0.3), ('springs', 1e-4), ('sub', 3), ('steps', 20), ('iter', 800), ('seed', 1), ('ne', 10.0)):
    check('ESSENTIAL %s = %s is flagged' % (k, v), any(k in x for x in sp.violations(S(**{k: v}), pol)))
check('ratio 2.0 flagged, ratio 0.7 and 1.6 allowed', sp.violations(S(cpi=math.sqrt(352 * 2.0), wpi=math.sqrt(352 / 2.0)), pol) != [] and sp.violations(S(cpi=math.sqrt(352 * 0.7), wpi=math.sqrt(352 / 0.7)), pol) == [] and sp.violations(S(cpi=math.sqrt(352 * 1.6), wpi=math.sqrt(352 / 1.6)), pol) == [])
check('density other than 352 flagged', sp.violations(S(cpi=30, wpi=16), pol) != [])
check('tiles 2 and 7 flagged, 3 and 6 allowed (6 only if the stitch cap holds)', sp.violations(S(tiles=2), pol) != [] and sp.violations(S(tiles=7), pol) != [] and sp.violations(S(tiles=3), pol) == [])
check('stitch cap: Pattern 1 at tiles 6 (11,520 stitches) is allowed, a 24 x 28 seed at tiles 4 (10,752) is allowed, at tiles 5 (16,800) refused', sp.violations(S(tiles=6), pol) == [] and sp.violations(sc.normalise(sp._setting(pol, seeds[6]['matrix'], 22.0, 16.0, 4), 12000), pol) == [] and sp.violations(sc.normalise(sp._setting(pol, seeds[6]['matrix'], 22.0, 16.0, 5), 30000), pol) != [])
ctl = dict(reason='cap sensitivity', keys=['iter'])
check('a labelled control may change iter only', sp.violations(S(iter=800), pol, ctl) == [] and sp.violations(S(seed=1), pol, ctl) != [] and sp.violations(S(iter=800), pol, dict(reason='', keys=['iter'])) != [] and sp.violations(S(kappa_scale=2.0), pol, dict(reason='x', keys=['kappa_scale'])) != [])

# ---- edits
rng = random.Random(1); ok = True; fr = []
for i, sd in enumerate(seeds * 20):
    M = sp.edit_matrix(sd['matrix'], rng, pol)
    if M is None: ok = False; continue
    f0 = sp.knit_fraction(sd['matrix']); f = sp.knit_fraction(M); fr.append(abs(f - f0))
    ok = ok and M != sd['matrix'] and len(M) == len(sd['matrix']) and all(len(r) == len(sd['matrix'][0]) for r in M) and {v for r in M for v in r} == {1, 2} and abs(f - f0) <= 0.10 + 1e-9
check('200 edits of the 10 seeds: size unchanged, differs from the seed, both loop types, knit fraction within 10 points', ok, 'max change %.3f' % max(fr))
d = [sum(a != b for ra, rb in zip(sd['matrix'], sp.edit_matrix(sd['matrix'], random.Random(i), pol)) for a, b in zip(ra, rb)) for i, sd in enumerate(seeds * 5)]
check('edits are slight: at most 3 operations of at most 3 x 3 cells or 2-cell boundary shifts (changed cells per edit <= 3 x 9 and typically small)', max(d) <= 3 * 9 * 3 and min(d) >= 1, 'changed cells min %d median %d max %d' % (min(d), sorted(d)[len(d) // 2], max(d)))
check('edit is deterministic for a seed', sp.edit_matrix(P1, random.Random(5), pol) == sp.edit_matrix(P1, random.Random(5), pol))

# ---- plan
plan = sp.build_plan(pol, seeds, 20261121, 91, 300); plan2 = sp.build_plan(pol, seeds, 20261121, 91, 300); plan3 = sp.build_plan(pol, seeds, 7, 91, 300)
tiers = {}
for e in plan: tiers[e['tier']] = tiers.get(e['tier'], 0) + 1
check('plan tiers: 11 seed runs, 2 controls, 91 variants, 300 filler', tiers == {0: 11, 1: 2, 2: 91, 9: 300}, str(tiers))
check('plan is deterministic for a seed and differs for another', json.dumps(plan, sort_keys=True) == json.dumps(plan2, sort_keys=True) and json.dumps(plan, sort_keys=True) != json.dumps(plan3, sort_keys=True))
check('every plan entry passes the policy (controls with their label); controls are Pattern 1 tiles 3 at iter 800 and 200', all(sp.violations(sc.normalise(e['settings'], 12000), pol, e['control']) == [] for e in plan) and sorted(e['settings']['iter'] for e in plan if e['control']) == [200, 800])
check('all entries unique by settings', len({sp._sha(e['settings']) for e in plan}) == len(plan))
Rs = [e['settings']['cpi'] / e['settings']['wpi'] for e in plan if e['tier'] == 2]
check('variant ratios cover the approved range (stratified, WPI above CPI included)', min(Rs) < 0.73 and max(Rs) > 1.55 and sum(r < 1 for r in Rs) >= 15, 'R %.3f to %.3f, %d below 1' % (min(Rs), max(Rs), sum(r < 1 for r in Rs)))
check('variants at constant stitch density 352 (CPI x WPI) and tiles 4, filler tiles 3 to 6 within the cap', all(abs(e['settings']['cpi'] * e['settings']['wpi'] / 352 - 1) < 0.005 for e in plan) and {e['settings']['tiles'] for e in plan if e['tier'] == 2} == {4} and {e['settings']['tiles'] for e in plan if e['tier'] == 9} <= {3, 4, 5, 6} and all(sp.n_stitches(e['settings']) <= 12000 for e in plan))
check('variants spread over all 10 seeds', len({e['source'] for e in plan if e['tier'] == 2}) == 10)

# ---- enforcement in the campaign
c = sc.Camp.create('pol', dict(policy=pol)); good = sp._setting(pol, P1, 22.0, 16.0, 3)
check('enqueue accepts the success settings', c.enqueue(good, source='t') == 'S000001')
for label, st in (('kappa_scale 1.1', dict(good, kappa_scale=1.1)), ('iter 800 without a control label', dict(good, iter=800)), ('ratio 2.0', dict(good, cpi=math.sqrt(704), wpi=math.sqrt(176))), ('tiles 7', dict(good, tiles=7)), ('ne 10', dict(good, ne=10.0))):
    check('enqueue refuses ' + label, raises(lambda st=st: c.enqueue(st, source='t')))
check('enqueue accepts a labelled control (iter 800)', c.enqueue(dict(good, iter=800), source='control', tier=1, control=ctl) == 'S000002')
check('a control that changes the seed is refused', raises(lambda: c.enqueue(dict(good, seed=3), source='control', control=ctl)))
check('at most 2 control runs per campaign', c.enqueue(dict(good, iter=200), source='control', tier=1, control=ctl) == 'S000003' and raises(lambda: c.enqueue(dict(good, iter=100), source='control', tier=1, control=ctl)))
c.enqueue(sp._setting(pol, P1, 22.0, 16.0, 4), source='filler', tier=9); c.enqueue(sp._setting(pol, P1, 22.0, 16.0, 5), source='seed', tier=0)
check('the queue runs lower tiers first (seed 0, controls 1, default 5, filler 9), then by id', [x['source'] for x in c.pending()] == ['seed', 'control', 'control', 't', 'filler'], str([x['source'] for x in c.pending()]))
# a hand-edited queue line that breaks the policy is rejected by the runner, the others still run
sc.append_line(c.queue, dict(id='S000099', t=cm.now(), source='hand', sha1='x', tier=0, settings=sc.normalise(dict(good, kappa_scale=1.5), 12000)))
c.set_control('run'); sc.serve(c, poll_s=0.2, stop_when_empty=True, lock=False, run_poll=0.1)
last = c.index_last()
check('runner rejects a hand-edited entry that breaks the policy (status rejected, reason recorded) and runs the others', last['S000099']['status'] == 'rejected' and 'ESSENTIAL' in last['S000099']['reason'] and sum(1 for x in last.values() if x['status'] == 'done') == 5, str({k: v['status'] for k, v in last.items()}))
check('the policy and the plan are copied into the campaign spec; policy.py is part of the frozen code', c.spec_()['policy']['essential']['iter'] == 400 and 'shell_policy.py' in c.manifest()['files'])

# ---- shell-plan --policy through the command line (real seeds, small filler)
import subprocess
r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'campaigns', 'campaign.py'), 'shell-plan', 'seeded', '--policy', os.path.join(ROOT, 'campaigns', 'shell_policy.json'), '--filler', '20'], capture_output=True, text=True, env=dict(os.environ))
sd = sc.Camp('seeded')
check('shell-plan --policy builds the seeded queue and is paused', r.returncode == 0 and sd.exists() and len(sd.queue_items()) == 11 + 2 + 91 + 20 and sd.get_control() == 'pause', (r.stdout + r.stderr)[-200:])
check('the seeded queue starts with the seeds at tiers 0, then controls, variants, filler', [x['tier'] for x in sd.pending()][:13] == [0] * 11 + [1] * 2)
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
