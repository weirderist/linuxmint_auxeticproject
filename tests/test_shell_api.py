#!/usr/bin/env python3
"""API tests of the 'Pure shell only runs' section (webui/campaign_api.py /api/shell/...) with FastAPI's TestClient, WITHOUT a server or the solver (shell_run.py in stub mode).
Temporary folders only. Run from the repository root: python tests/test_shell_api.py"""
import json, os, subprocess, sys, tempfile, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='shellapi_')
os.environ.update(SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), CAMPAIGN_DIR=os.path.join(TMP, 'campaigns'), SHELL_RUN_STUB='1', CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'))
os.makedirs(os.environ['CAMPAIGN_DIR'])
sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'webui'))
from fastapi import FastAPI
from fastapi.testclient import TestClient
import campaign_api as api, shell_campaign as sc
app = FastAPI(); api.setup(app, lambda: False); cl = TestClient(app)
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)

check('no shell campaign yet: empty list', cl.get('/api/shell').json() == {'campaigns': []})
check('unknown campaign: 404, bad name: 404', cl.get('/api/shell/nope').status_code == 404 and cl.get('/api/shell/..%2Fx').status_code in (404, 422))
c = sc.Camp.create('demo'); M = '1 1 2 2/1 1 2 2/2 2 1 1/2 2 1 1'
for i in range(3): c.enqueue(dict(matrix=M, tiles=1, cpi=22, wpi=16, seed=i))
j = cl.get('/api/shell').json()['campaigns']; check('list shows the campaign (control pause, 3 queued)', len(j) == 1 and j[0]['control'] == 'pause' and j[0]['counts']['queued'] == 3, str(j))
d = cl.get('/api/shell/demo').json()
check('detail: status, limits, rows, disk figures', d['name'] == 'demo' and d['limits']['min_free_gb'] == 20.0 and len(d['rows']) == 3 and d['free_gb'] > 0 and d['runner_alive'] is False)
check('pause / stop / continue set control.json', cl.post('/api/shell/demo/pause').json()['control'] == 'pause' and cl.post('/api/shell/demo/stop').json()['control'] == 'stop' and c.get_control() == 'stop')
r = cl.post('/api/shell/demo/continue'); check('continue without a runner is refused (409)', r.status_code == 409, r.text[:100])
calls = []
def fake_run(cmd, **kw): calls.append(cmd); return types.SimpleNamespace(returncode=0, stdout='shell runner started (fake)', stderr='')
real = api.subprocess.run; api.subprocess.run = fake_run
r = cl.post('/api/shell/demo/start'); check('start: control = run and the detached runner command of the project interpreter', r.status_code == 200 and c.get_control() == 'run' and calls and calls[0][-3:] == ['shell-run', 'demo', '--detach'] and calls[0][1] == os.path.join(c.code_dir(), 'campaign.py'), str(calls))
r2 = cl.post('/api/shell/demo/start'); check('a second start a moment later is refused (double-click guard)', r2.status_code == 409, r2.text[:100])
api._last_start.clear(); api._any_web_job = lambda: True; r3 = cl.post('/api/shell/demo/start'); api._any_web_job = lambda: False
check('start refused while a web simulation runs', r3.status_code == 409 and 'simulation' in r3.text)
api.subprocess.run = real
import shutil; nf = sc.Camp.create('nofreeze'); shutil.move(nf.code_dir(), nf.code_dir() + '_moved'); api._last_start.clear()
r4 = cl.post('/api/shell/nofreeze/start'); check('start refused for a campaign without frozen code (nothing started, control unchanged)', r4.status_code == 409 and 'frozen' in r4.text and nf.get_control() == 'pause', r4.text[:120])
# finish the 3 runs with the stub, then delete
c.set_control('run'); sc.serve(c, poll_s=0.2, stop_when_empty=True, lock=False, run_poll=0.1)
d = cl.get('/api/shell/demo').json(); check('runs finished and listed with descriptors', d['counts']['done'] == 3 and d['rows'][0]['plan_x'] and d['mean_wall_s'] is not None)
pv = cl.get('/api/shell/demo/delete-preview?ids=S000001,S000002').json(); check('delete preview: ids and bytes, nothing removed', pv['ids'] == ['S000001', 'S000002'] and pv['bytes'] > 0 and os.path.exists(os.path.join(c.runs, 'S000001', 'shell.npz')))
r = cl.post('/api/shell/demo/delete', json=dict(ids=['S000001', 'S000002'], confirm='yes')); check('delete without the typed confirmation: 422, nothing removed', r.status_code == 422 and os.path.exists(os.path.join(c.runs, 'S000001', 'shell.npz')), r.text[:120])
r = cl.post('/api/shell/demo/delete', json=dict(ids=['S000001', 'S000002'], confirm='DELETE 3')); check('wrong number in the confirmation: 422', r.status_code == 422)
for ids in (['../S000001'], ['S000001/../S000002'], ['..'], ['S000099'], []):
    r = cl.post('/api/shell/demo/delete', json=dict(ids=ids, confirm='DELETE %d' % len(ids))); check('delete refused for %s' % ids, r.status_code == 422, str(r.status_code))
r = cl.post('/api/shell/demo/delete', json=dict(ids=['S000001', 'S000002'], confirm='DELETE 2')); j = r.json()
check('delete with the right confirmation removes the heavy files of exactly those runs', r.status_code == 200 and j['files'] == 4 and not os.path.exists(os.path.join(c.runs, 'S000001', 'shell.npz')) and os.path.exists(os.path.join(c.runs, 'S000003', 'shell.npz')) and os.path.exists(os.path.join(c.runs, 'S000001', 'metrics.json')))
check('the deletion is logged with the client address', len(sc.read_lines(c.delog)) == 1 and sc.read_lines(c.delog)[0]['ids'] == ['S000001', 'S000002'] and sc.read_lines(c.delog)[0]['client'] != None)
d = cl.get('/api/shell/demo').json(); check('detail marks heavy_deleted rows', [r['heavy_deleted'] for r in d['rows']] == [True, True, False])
# ---- one run: detail, 3D mesh, strain colours (stub shell.npz), and the separate pages
import struct
d3 = cl.get('/api/shell/demo/runs/S000003').json()
check('run detail: settings, descriptors, provenance, files', d3['run'] == 'S000003' and d3['settings']['cpi'] == 22.0 and d3['metrics']['plan_x'] and d3['provenance']['shell_sim_sha256'] and d3['files'] == {'shell.npz': True, 'shell.bin': True} and 'log_tail' in d3)
r = cl.get('/api/shell/demo/runs/S000003/mesh'); nv, nf = struct.unpack_from('<II', r.content, 8)
check('mesh endpoint returns the KMSH v1 file of the run', r.status_code == 200 and r.content[:4] == b'KMSH' and r.content == open(os.path.join(c.runs, 'S000003', 'shell.bin'), 'rb').read() and nv > 0)
for fld in ('area_strain', 'stretch_max', 'stretch_min', 'z', 'tilt'):
    r = cl.get('/api/shell/demo/runs/S000003/colors?field=' + fld); lg = json.loads(r.headers['X-Legend'])
    check('colors %s: one RGB triple per mesh vertex, legend with label, range and 3 ticks' % fld, r.status_code == 200 and len(r.content) == 3 * nv == 3 * int(r.headers['X-Count']) and lg['field'] == fld and len(lg['ticks']) == 3 and lg['label'] and lg['vmax'] >= lg['vmin'])
check('colors: unknown field 422, clip out of range 422', cl.get('/api/shell/demo/runs/S000003/colors?field=bogus').status_code == 422 and cl.get('/api/shell/demo/runs/S000003/colors?field=z&clip=10').status_code == 422)
for bad_id in ('S000099', '%2e%2e', 'S000003%2f..', '..%2fS000003', 'S3'):
    check('run endpoints refuse %s' % bad_id, all(cl.get('/api/shell/demo/runs/%s%s' % (bad_id, suf)).status_code in (404, 422) for suf in ('', '/mesh', '/colors')))
check('after the delete of S000001 its mesh and colours say the 3D data was deleted (404), detail still works', cl.get('/api/shell/demo/runs/S000001/mesh').status_code == 404 and cl.get('/api/shell/demo/runs/S000001/colors').status_code == 404 and cl.get('/api/shell/demo/runs/S000001').json()['heavy_deleted'] is True)
check('status carries the frozen code (commit, dirty flag, shell_sim hash)', 'git_commit' in cl.get('/api/shell/demo').json()['code'] and cl.get('/api/shell/demo').json()['code']['shell_sim_sha256'])
check('runner_busy (used by the Simulate endpoint) is empty with no live runner', api.runner_busy() == '')
html = cl.get('/campaigns').text; js = cl.get('/campaigns.js').text; sh = cl.get('/shell').text; shjs = cl.get('/shell.js').text; sv = cl.get('/shellview').text; svjs = cl.get('/shellview.js').text
check('the Pure shell section is gone from the Campaigns page and replaced by a link to /shell', 'shellsec' not in html and 'api/shell' not in js and 'href="/shell"' in html)
check('/shell is its own page with the controls, table, comparison and graph', all(k in sh for k in ('bStart', 'bPause', 'bCont', 'bStop', 'bDel', 'bCompare', 'gx')) and 'api/shell' in shjs and 'shellview' in shjs)
check('/shellview is the viewer page (importmap, viewer_core, two views)', 'viewer_core' in svjs and 'vA' in sv and 'vB' in sv and 'importmap' in sv and 'colors?field=' in svjs)
try:
    import esprima; [esprima.parseScript(x) for x in (js, shjs)]; esprima.parseModule(svjs); check('campaigns.js, shell.js (script) and shellview.js (module) parse (esprima)', True)
except ImportError: print('SKIP  esprima not installed')
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
