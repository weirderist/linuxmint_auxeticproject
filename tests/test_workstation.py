#!/usr/bin/env python3
"""End-to-end test of the workstation worker (scripts/workstation/ws_agent.py + ws_remote.py + the /api/shell/<c>/workstation route, 2026-11-27) WITHOUT ssh or the solver: a local transport stands
in for ssh/scp, shell_run.py runs in stub mode. Temporary folders only. Run from the repository root:  python tests/test_workstation.py
Checks: nothing is claimed until K allows it; allowed runs are claimed one at a time, run from the checked frozen copy, uploaded and imported with worker workstation:<host>; specific runs go first;
the allowance is respected; Cancel stops the run, releases it and keeps its folder in quarantine; a power cut (agent and run gone) releases the run at the next start; during a network outage the
run continues and is imported later; an expired lease is not renewed back."""
import json, os, shutil, subprocess, sys, tempfile, time, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='wstest_')
os.environ.update(SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), CAMPAIGN_DIR=os.path.join(TMP, 'campaigns'), SHELL_RUN_STUB='1', CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'))
os.makedirs(os.environ['CAMPAIGN_DIR'])
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs; sys.path.insert(0, os.path.join(ROOT, 'webui')); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'workstation'))
from fastapi import FastAPI
from fastapi.testclient import TestClient
import campaign_api as api, shell_campaign as sc, ws_agent as wa
api._sc = lambda: sc; app = FastAPI(); api.setup(app, lambda: False); cl = TestClient(app)
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)

class Local:
    """Stands in for ssh/scp: ws_remote.py runs here (same SHELL_RUNS_DIR), files are copied locally. net=False simulates an outage."""
    net = True
    def remote(s, args, timeout=300):
        if not s.net: return None, 'network unreachable'
        p = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'workstation', 'ws_remote.py')] + [str(a) for a in args], capture_output=True, text=True)
        try: return json.loads(p.stdout.strip().splitlines()[-1]), ''
        except (ValueError, IndexError): return None, p.stderr[-400:]
    def exists(s, path):
        if not s.net: return None
        return os.path.isdir(path)
    def rename(s, a, b):
        if not s.net: return False
        os.rename(a, b); return True
    def get(s, rp, lp, timeout=600):
        if not s.net: return False
        shutil.copy2(rp, lp); return True
    def put_dir(s, ld, rd, timeout=1800):
        if not s.net: return False
        shutil.copytree(ld, rd); return True

M = '1 1 2 2/1 1 2 2/2 2 1 1/2 2 1 1'
c = sc.Camp.create('W1'); ids = [c.enqueue(dict(matrix=M, tiles=1, cpi=22, wpi=16, seed=i, stub_seconds=0)) for i in range(6)]
c.enqueue(dict(matrix=M, tiles=1, cpi=22, wpi=16, seed=99, stub_seconds=60)); slow = c.queue_items()[-1]['id']
T = Local(); g = types.SimpleNamespace(campaign='W1', python=sys.executable, work=os.path.join(TMP, 'ws'), host='server', remote_root='~', worker='workstation:test', stub=True)
ag = wa.Agent(g, T)
def ticks(n, agent=None):
    for _ in range(n): (agent or ag).tick(); time.sleep(0.3)

ticks(2); check('not allowed: nothing claimed', not c.active_leases() and not os.listdir(os.path.join(g.work, 'runs')))
r = cl.post('/api/shell/W1/workstation', json={'enabled': True, 'n': 2, 'ids': [ids[3]]}); check('page: allow 2 runs, S000004 first', r.status_code == 200 and r.json()['config']['n'] == 2, r.text[:200])
ticks(1); check('the volunteered run is claimed first', ag.state().get('rid') == ids[3] and c.lease_of(ids[3])['worker'] == 'workstation:test')
check('frozen code fetched and checked', any(os.path.isfile(os.path.join(g.work, 'code', d, 'shell_sim.py')) for d in os.listdir(os.path.join(g.work, 'code'))))
for _ in range(20):
    ag.tick(); time.sleep(0.3)
    if c.index_last().get(ids[3], {}).get('status') == 'done': break
last = c.index_last(); check('run uploaded and imported with the workstation worker', last.get(ids[3], {}).get('status') == 'done' and last[ids[3]].get('worker') == 'workstation:test', last.get(ids[3]))
w = cl.get('/api/shell/W1/workstation' if False else '/api/shell/W1/workers').json()['workstation']; check('workers box: agent seen, 1 of 2 used', w['granted'] >= 1 and w['agent_seen_s_ago'] is not None and w['done'] == 1, w)
for _ in range(25):
    ag.tick(); time.sleep(0.3)
    if sum(1 for x in c.index_last().values() if x.get('worker') == 'workstation:test') == 2: break
done_ws = [k for k, x in c.index_last().items() if x.get('worker') == 'workstation:test']; check('second run (head of the queue) done; allowance then used up', len(done_ws) == 2 and ids[0] in done_ws, done_ws)
ticks(2); check('allowance used up: no third claim', not [r for r, L in c.active_leases().items() if L['worker'].startswith('workstation')])

# cancel during a long run
for rid in ids:
    if rid not in c.index_last(): c.claim('kaggle:x', 1, 3600)          # occupy the quick runs so the slow one is next
cl.post('/api/shell/W1/workstation', json={'enabled': True, 'n': 1}); ticks(1); check('slow run started', ag.state().get('rid') == slow, ag.state())
T.net = False; ticks(2); check('network outage: the run continues, no crash', ag.state().get('phase') == 'running' and wa.proc_alive(ag.state()['pid']))
T.net = True; ticks(1); check('back online: lease renewed', c.lease_of(slow) and c.lease_of(slow)['worker'] == 'workstation:test')
cl.post('/api/shell/W1/workstation', json={'enabled': False}); ticks(1)
check('Cancel: run stopped, lease released, folder in quarantine', not c.lease_of(slow) and any(x.startswith(slow + '.cancelled') for x in os.listdir(os.path.join(g.work, 'quarantine'))) and slow not in c.index_last())

# power cut: a run in progress, then both agent and run are gone; a new agent starts
cl.post('/api/shell/W1/workstation', json={'enabled': True, 'n': 1}); ticks(1); st = ag.state(); check('run started again', st.get('rid') == slow)
wa.kill_tree(st['pid']); time.sleep(1); ag2 = wa.Agent(g, T); ticks(1, ag2)
check('after a power cut: the interrupted run is released and kept in quarantine', not c.lease_of(slow) and any(x.startswith(slow + '.interrupted') for x in os.listdir(os.path.join(g.work, 'quarantine'))), os.listdir(os.path.join(g.work, 'quarantine')))
check('an expired lease is not renewed back', c.renew([slow], 'workstation:test', 60) == [])
print('ALL OK' if not bad else 'FAILED: %s' % bad); shutil.rmtree(TMP, ignore_errors=True); sys.exit(1 if bad else 0)
