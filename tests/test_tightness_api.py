#!/usr/bin/env python3
"""API tests of the display-only tightness values (S = CPI x WPI, TF_rel = sqrt(tex x S)) in webui/campaign_api.py, with FastAPI's TestClient, WITHOUT a server or the solver (shell_run.py in stub mode).
Temporary folders only. Checks the values against hand-computed numbers, the 'constant in this campaign' note, and that reading writes nothing into any run folder.
Run from the repository root: python tests/test_tightness_api.py"""
import hashlib, json, math, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='tightapi_')
os.environ.update(SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), CAMPAIGN_DIR=os.path.join(TMP, 'campaigns'), SHELL_RUN_STUB='1', CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'))
os.makedirs(os.environ['CAMPAIGN_DIR'])
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs; sys.path.insert(0, os.path.join(ROOT, 'webui'))
from fastapi import FastAPI
from fastapi.testclient import TestClient
import campaign_api as api, shell_campaign as sc, shell_policy as sp
app = FastAPI(); api.setup(app, lambda: False); cl = TestClient(app)
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)
HAND = 131.610537                     # sqrt(590.5 / 12 x 22 x 16) = sqrt(17321.333) (tests/test_shell_tightness.py)
def tree_hash(d):
    h = hashlib.sha256()
    for r, ds, fs in sorted(os.walk(d)):
        ds.sort()
        for f in sorted(fs): p = os.path.join(r, f); h.update(p.encode()); h.update(open(p, 'rb').read())
    return h.hexdigest()

# ---- shell campaign with the approved policy (CPI x WPI fixed at 352): two finished stub runs at R 1.375 and R 1.6
pol = sp.load(os.path.join(ROOT, 'campaigns', 'shell_policy.json')); P1 = open(os.path.join(ROOT, 'campaigns', 'pattern1_matrix.txt')).read()
c = sc.Camp.create('pol', dict(policy=pol)); cpi2, wpi2 = sp.cpi_wpi(1.6, 352.0)
ids = [c.enqueue(dict(pol['essential'], matrix=P1, tiles=3, cpi=22.0, wpi=16.0, ne=12.0)), c.enqueue(dict(pol['essential'], matrix=P1, tiles=3, cpi=cpi2, wpi=wpi2, ne=12.0))]
c.set_control('run'); sc.serve(c, poll_s=0.2, stop_when_empty=True, lock=False, run_poll=0.1)
before = tree_hash(c.runs); d = cl.get('/api/shell/pol').json(); rows = {r['run']: r for r in d['rows']}
check('two runs finished in stub mode', d['counts']['done'] == 2, str(d['counts']))
check('run list: S = 352 and TF_rel = 131.6105 for CPI 22, WPI 16 (hand value)', rows[ids[0]]['stitch_density'] == 352.0 and abs(rows[ids[0]]['tf_rel'] - HAND) < 1e-5, str((rows[ids[0]]['stitch_density'], rows[ids[0]]['tf_rel'])))
TFC, TFW = 7.014866 * 22.0, 7.014866 * 16.0          # sqrt(tex) x CPI and x WPI, hand values 154.327 and 112.238
check('run list: TF_c_rel 154.327 and TF_w_rel 112.238 for CPI 22, WPI 16 (hand values)', abs(rows[ids[0]]['tf_c_rel'] - TFC) < 1e-3 and abs(rows[ids[0]]['tf_w_rel'] - TFW) < 1e-3)
check('the run at R 1.6 (same S) differs in both: TF_c_rel larger, TF_w_rel smaller, while TF_rel stays within 0.25 %', rows[ids[1]]['tf_c_rel'] > rows[ids[0]]['tf_c_rel'] * 1.05 and rows[ids[1]]['tf_w_rel'] < rows[ids[0]]['tf_w_rel'] * 0.95 and abs(rows[ids[1]]['tf_rel'] / rows[ids[0]]['tf_rel'] - 1) < 0.0025, str((rows[ids[1]]['tf_c_rel'], rows[ids[1]]['tf_w_rel'])))
check('R 1.6 at the same density: S within 0.5 % of 352 (CPI, WPI rounded to 4 decimals), TF_rel within 0.25 %', abs(rows[ids[1]]['stitch_density'] / 352 - 1) < 0.005 and abs(rows[ids[1]]['tf_rel'] / HAND - 1) < 0.0025)
t = d['tightness']; check('campaign info: constant with the policy, note "constant in this campaign (CPI x WPI fixed)", labels say nominal and UNSOURCED', t['constant'] is True and t['constant_note'] == 'constant in this campaign (CPI x WPI fixed)' and 'UNSOURCED' in t['label_tf_rel'] and 'nominal' in t['label_stitch_density'], str(t)[:200])
rd = cl.get('/api/shell/pol/runs/' + ids[0]).json()['tightness']
check('run detail (shellview panel): S, TF_rel, TF_c_rel, TF_w_rel, Ne 12, the labels, the note and the constant note', abs(rd['tf_c_rel'] - TFC) < 1e-3 and abs(rd['tf_w_rel'] - TFW) < 1e-3 and 'UNSOURCED' in rd['label_tf_c_rel'] and 'UNSOURCED' in rd['label_tf_w_rel'] and rd['stitch_density'] == 352.0 and abs(rd['tf_rel'] - HAND) < 1e-5 and rd['ne'] == 12.0 and rd['constant_note'] and 'UNSOURCED' in rd['label_tf_rel'])
check('reading wrote nothing into any run folder (sha256 of the runs tree unchanged) and no settings gained a tightness key', tree_hash(c.runs) == before and all('tf_rel' not in q['settings'] and 'stitch_density' not in q['settings'] for q in c.queue_items()))

# ---- shell campaign without a policy and two densities: not constant
c2 = sc.Camp.create('free'); M = '1 1 2 2/1 1 2 2/2 2 1 1/2 2 1 1'
c2.enqueue(dict(matrix=M, tiles=1, cpi=22, wpi=16)); c2.enqueue(dict(matrix=M, tiles=1, cpi=11, wpi=16))
t2 = cl.get('/api/shell/free').json()['tightness']; r2 = cl.get('/api/shell/free').json()['rows']
check('no policy and S 352 / 176: not constant, no note; queued runs still get S and TF_rel', t2['constant'] is False and t2['constant_note'] == '' and sorted(r['stitch_density'] for r in r2) == [176.0, 352.0] and abs(min(r['tf_rel'] for r in r2) - HAND / math.sqrt(2)) < 1e-5)

# ---- yarn-level campaign (campaigns page): S and TF_rel from the spec's fabric merged with the run's parameters, with the run's own Ne
yd = os.path.join(os.environ['CAMPAIGN_DIR'], 'yarn1'); os.makedirs(os.path.join(yd, 'runs', '0001')); os.makedirs(os.path.join(yd, 'runs', '0002'))
json.dump(dict(name='yarn1', fabric=dict(matrix_file='pattern1_matrix.txt', ne=12, cpi=22, wpi=16, tiles=1), relax=dict(E=3700.0), vary=[dict(name='cpi', values=[22, 11])]), open(os.path.join(yd, 'spec.json'), 'w'))
json.dump(dict(name='yarn1', runs=[dict(run='0001', params=dict(cpi=22)), dict(run='0002', params=dict(cpi=11, ne=590.5))]), open(os.path.join(yd, 'manifest.json'), 'w'))
for rid in ('0001', '0002'): json.dump(dict(run=rid, status='queued'), open(os.path.join(yd, 'runs', rid, 'status.json'), 'w'))
yb = tree_hash(yd); y = cl.get('/api/campaigns/yarn1'); yj = y.json() if y.status_code == 200 else {}; yr = {r['run']: r for r in yj.get('runs', [])}
check('campaigns page: run 0001 (fabric CPI 22, WPI 16, Ne 12) gets S 352 and TF_rel 131.6105', y.status_code == 200 and yr['0001']['stitch_density'] == 352.0 and abs(yr['0001']['tf_rel'] - HAND) < 1e-5, y.text[:200])
check('campaigns page: run 0002 uses its own CPI 11 and Ne 590.5 (tex 1): S 176, TF_rel sqrt(176)', yr['0002']['stitch_density'] == 176.0 and abs(yr['0002']['tf_rel'] - math.sqrt(176)) < 1e-9)
check('campaigns page: S differs between runs, so not constant (the charts then add S and TF_rel as x axes)', yj['tightness']['constant'] is False)
rj = cl.get('/api/campaigns/yarn1/runs/0002').json(); check('campaigns run detail carries the nominal values, including TF_c_rel = 1 x 11 (tex 1) and TF_w_rel = 1 x 16', rj['nominal']['stitch_density'] == 176.0 and abs(rj['nominal']['tf_c_rel'] - 11.0) < 1e-9 and abs(rj['nominal']['tf_w_rel'] - 16.0) < 1e-9)
check('reading the yarn campaign wrote nothing', tree_hash(yd) == yb)
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
