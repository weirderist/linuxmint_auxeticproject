#!/usr/bin/env python3
"""Tests of 2026-11-28: single shell runs from the Simulator page (webui/shellsim.py), the shell2 settings they use, the viewer route, the rules in two formats (shell_rules_model.py,
shell_rules.py --originals-only) and the compressed mesh download. No solver: shell_run.py runs in its stub mode (SHELL_RUN_STUB=1). Temporary folders only.
Run from the repository root: python tests/test_shellsim.py"""
import gzip, json, os, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='shellsim_')
os.environ.update(AUX_SHELLSIM_DIR=os.path.join(TMP, 'sims'), SHELL_RUN_STUB='1', SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), CAMPAIGN_DIR=os.path.join(TMP, 'campaigns'), AUX_AUTORESUME='0')
os.makedirs(os.environ['CAMPAIGN_DIR'])
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs; sys.path.insert(0, os.path.join(ROOT, 'webui'))
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)

from fastapi import FastAPI
from fastapi.testclient import TestClient
import campaign_api as api, shellsim
app = FastAPI(); api.setup(app, lambda: False); shellsim.setup(app, api, lambda: False); cl = TestClient(app)

M = '1 1 2 2/1 1 2 2/2 2 1 1/2 2 1 1'
check('empty list at first', cl.get('/api/shellsim').json() == [])
for body, why in ((dict(name='x', matrix='1 3/1 1', cpi=22, wpi=16, tiles=2), 'digit 3'), (dict(name='x', matrix='1 1/1', cpi=22, wpi=16, tiles=2), 'ragged rows'),
                  (dict(name='', matrix=M, cpi=22, wpi=16), 'no name'), (dict(name='x', matrix=M, cpi=2, wpi=16), 'CPI out of range')):
    r = cl.post('/api/shellsim', json=body)
    check('refused: %s (422)' % why, r.status_code == 422, r.text[:80])
big = '/'.join([' '.join(['1'] * 40)] * 40)
r = cl.post('/api/shellsim', json=dict(name='big', matrix=big, cpi=22, wpi=16, tiles=2)); check('refused: more than 3600 stitches after tiling', r.status_code == 422 and '3600' in r.text, r.text[:90])
r = cl.post('/api/shellsim', json=dict(name='blocks', matrix=M, cpi=22, wpi=16, tiles=2)); j = r.json(); rid = j.get('id')
check('started: id from date and time, stitch count', r.status_code == 200 and shellsim.ID_RE.match(rid or '') and j['stitches'] == 64, r.text[:120])
pol = json.load(open(os.path.join(ROOT, 'campaigns', 'shell2_policy.json')))
s = json.load(open(os.path.join(TMP, 'sims', rid, 'settings.json')))
check('settings = the shell2 essential settings + the pattern, CPI, WPI, tiles', all(s[k] == v for k, v in pol['essential'].items()) and s['cpi'] == 22 and s['tiles'] == 2 and s['matrix'][0] == [1, 1, 2, 2] and s['ne'] == 12)
r2 = cl.post('/api/shellsim', json=dict(name='second', matrix=M, cpi=22, wpi=16, tiles=2))
check('a second run while one is running is refused (409) or the first already finished', r2.status_code == 409 or cl.get('/api/shellsim/' + rid).json()['state'] == 'done', r2.text[:80])
for _ in range(120):
    d = cl.get('/api/shellsim/' + rid).json()
    if d['state'] != 'running': break
    time.sleep(0.5)
check('the stub run finishes: state done, metrics, log', d['state'] == 'done' and d['metrics'].get('plan_x') and d['log_tail'], str(d.get('state')) + ' ' + str(d.get('log_tail', [])[-2:]))
check('mesh served (KMSH)', cl.get('/api/shellsim/%s/mesh' % rid).content[:4] == b'KMSH')
c = cl.get('/api/shellsim/%s/colors?field=area_strain&clip=98' % rid); lg = json.loads(c.headers.get('X-Legend') or '{}')
check('strain colours and legend served', c.status_code == 200 and len(c.content) % 3 == 0 and lg.get('field') == 'area_strain')
check('list shows the run with its name and state', [x['name'] for x in cl.get('/api/shellsim').json()] [-1:] == ['blocks'] or any(x['name'] == 'blocks' and x['state'] == 'done' for x in cl.get('/api/shellsim').json()))
check('bad ids are 404', cl.get('/api/shellsim/..%2F..').status_code in (404, 422) and cl.get('/api/shellsim/20260101-000000').status_code == 404)
check('cancel of a finished run is refused (409)', cl.post('/api/shellsim/%s/cancel' % rid).status_code == 409)

# ---- the pages: simulator dialog and viewer know the shell runs
idx = open(os.path.join(ROOT, 'webui', 'index.html'), encoding='utf8').read(); sv = open(os.path.join(ROOT, 'webui', 'shellview.js'), encoding='utf8').read()
check('New dialog has Yarn mesh / Shell and posts to /api/shellsim; the list shows shell runs', 'id="nKShell"' in idx and "'/api/shellsim'" in idx and 'renderShells' in idx)
check('shell viewer opens ?sim=<id> from /api/shellsim and waits for a running one', "P.get('sim')" in sv and "'/api/shellsim/'" in sv and 'waitSim' in sv)
check('legend end words sit on their own row (same place on both sides)', 'lg-ends' in sv)

# ---- rules in two formats
import shell_rules_model as srm, shell_features as sf
import numpy as np
rng = np.random.default_rng(0); rows = []
for i in range(140):
    Mx = rng.integers(1, 3, size=(6, 6)).tolist(); f = sf.features(Mx, 2, 22, 16)
    rows.append(dict(features=f, targets=dict(thickness_rel=0.2 + 0.5 * f['knit_fraction'] + 0.01 * rng.standard_normal(), shrink_x=-0.3, shrink_y=None, fold_turn_x_rad=None, fold_turn_y_rad=None, wave_amp_rel=None), source='seed:x', control=False))
import shell_rules as sr
res = dict(campaign='t', generated='now', n_finished=140, originals_only=True, label=sr.LABEL, targets=[sr.analyse_target(rows, t, boots=6) for t in sf.TARGETS])
model, md = srm.export(res)
check('model file: schema, only the targets with a rule, intercept + terms with feature means', model['schema'] == srm.SCHEMA and 'thickness_rel' in model['targets'] and 'shrink_y' not in model['targets'] and all('mean' in c for c in model['targets']['thickness_rel']['terms']))
Mx = [[1, 1, 2], [1, 2, 2], [2, 2, 1]]; p = srm.predict(model, Mx, 2, 22, 16)['thickness_rel']; f = sf.features(Mx, 2, 22, 16)
check('predict() recovers the planted rule (0.2 + 0.5 x knit share) within 0.03', abs(p - (0.2 + 0.5 * f['knit_fraction'])) < 0.03, '%.4f vs %.4f' % (p, 0.2 + 0.5 * f['knit_fraction']))
check('readable file: plain names, up/down sentences, R2, the label, GP runs left out', 'share of knit loops' in md and 'Goes **up** with' in md and 'R2 =' in md and 'not confirmed on real fabric' in md and 'GP-chosen runs left out' in md)
d2 = tempfile.mkdtemp(dir=TMP); srm.write(res, d2)
check('write(): rules_model.json loads with load(), rules_readable.md exists', srm.load(os.path.join(d2, 'rules_model.json'))['targets'] and os.path.isfile(os.path.join(d2, 'rules_readable.md')))
src = open(os.path.join(ROOT, 'scripts', 'shell_learning', 'shell_rules.py'), encoding='utf8').read()
check('shell_rules.py: --originals-only leaves out proposer runs and writes both formats', "'--originals-only' in a" in src and "startswith('proposer')" in src and 'srm.write(res, d)' in src)

# ---- compressed mesh download (main.py)
import main
f = os.path.join(TMP, 'm.bin'); open(f, 'wb').write(b'KMSH' + bytes(3 << 20))
from pathlib import Path
main.GZ_CACHE = Path(TMP) / "gz"; g = main._gz_copy(Path(f)); check('mesh gzip copy: unpacks to the same bytes, cached once per version', gzip.open(g).read() == open(f, 'rb').read() and main._gz_copy(Path(f)) == g and str(main.GZ_CACHE) in str(g))
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
