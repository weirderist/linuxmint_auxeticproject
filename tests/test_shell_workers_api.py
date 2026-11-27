#!/usr/bin/env python3
"""API tests of the worker boxes of the /shell page (webui/campaign_api.py /api/shell/<c>/workers and /kaggle, 2026-11-27) with FastAPI's TestClient; no server, no Kaggle (Popen is faked).
Run from the repository root:  python tests/test_shell_workers_api.py"""
import json, os, sys, tempfile, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='workersapi_')
os.environ.update(SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), CAMPAIGN_DIR=os.path.join(TMP, 'campaigns'), SHELL_RUN_STUB='1', CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'))
os.makedirs(os.environ['CAMPAIGN_DIR'])
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs; sys.path.insert(0, os.path.join(ROOT, 'webui'))
from fastapi import FastAPI
from fastapi.testclient import TestClient
import campaign_api as api, shell_campaign as sc
app = FastAPI(); api.setup(app, lambda: False); cl = TestClient(app)
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)

c = sc.Camp.create('demo'); M = '1 1 2 2/1 1 2 2/2 2 1 1/2 2 1 1'; ids = [c.enqueue(dict(matrix=M, tiles=1, cpi=22, wpi=16, seed=i)) for i in range(4)]
api._sc = lambda: sc          # the route module must use the same shell_campaign as this test (temporary SHELL_RUNS_DIR)
w = cl.get('/api/shell/demo/workers').json()
check('workers: three boxes, Kaggle off, nothing leased', set(w) == {'server', 'kaggle', 'workstation'} and w['kaggle']['enabled'] is False and w['kaggle']['leased'] == 0 and w['workstation']['built'] is True and w['workstation']['enabled'] is False, w)
c.claim('kaggle:k1', 2, 3600, 'test'); w = cl.get('/api/shell/demo/workers').json(); check('leased runs counted for Kaggle', w['kaggle']['leased'] == 2, w['kaggle'])
rows = cl.get('/api/shell/demo').json()['rows']; check('rows carry worker and leased (page colours)', [r['worker'] for r in rows][:2] == ['kaggle:k1'] * 2 and rows[0]['leased'] and not rows[3]['leased'] and rows[3]['worker'] == '', [(r['worker'], r['leased']) for r in rows])
r = cl.post('/api/shell/demo/kaggle', json={'enabled': False}); cfg = json.load(open(os.path.join(c.d, 'kaggle', 'config.json')))
check('Kaggle off writes enabled false, starts nothing', r.status_code == 200 and cfg['enabled'] is False and 'no new sessions' in r.json()['message'])
started = []
real = api.subprocess.Popen; api.subprocess.Popen = lambda cmd, **kw: started.append(cmd) or types.SimpleNamespace(pid=1)
r = cl.post('/api/shell/demo/kaggle', json={'enabled': True}); cfg = json.load(open(os.path.join(c.d, 'kaggle', 'config.json')))
check('Kaggle on writes enabled true and starts the controller for this campaign', r.status_code == 200 and cfg['enabled'] is True and started and started[0][-1] == 'demo' and started[0][-2].endswith('kaggle_controller.py'), started)
r = cl.post('/api/shell/demo/kaggle', json={'enabled': True}); check('a second start right after is refused (409)', r.status_code == 409 and len(started) == 1, r.text[:120])
api.subprocess.Popen = real
r = cl.post('/api/shell/demo/kaggle', json={'enabled': 'maybe'}); check('a non-boolean is refused (422)', r.status_code == 422)
html = cl.get('/shell').text; check('/shell page has the worker boxes and colours', 'id="workers"' in html and '--w-kaggle' in html and '#e0832f' in html)
print('ALL OK' if not bad else 'FAILED: %s' % bad); sys.exit(1 if bad else 0)
