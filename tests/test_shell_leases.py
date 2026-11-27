#!/usr/bin/env python3
"""Tests of the shared queue with leases (scripts/campaigns/shell_campaign.py, 2026-11-27) WITHOUT the solver (SHELL_RUN_STUB=1). Everything in a temporary folder. Run from the repository root:
    python tests/test_shell_leases.py
Checks: a claimed run is skipped by everyone else; two claims never share a run; expiry and release put it back; only the holder releases; the server runner leases what it runs and releases after;
stale server leases are released at runner start; import_result accepts a correct remote run (worker recorded, idempotent) and refuses changed settings and foreign code; status counts by worker."""
import json, os, shutil, subprocess, sys, tempfile, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='leasetest_')
os.environ['SHELL_RUNS_DIR'] = os.path.join(TMP, 'shell_runs'); os.environ['SHELL_RUN_STUB'] = '1'; os.environ['CAMPAIGN_LOG_DIR'] = os.path.join(TMP, 'logs')
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs
import shell_campaign as sc, campaign as cm
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)

M = '1 1 2 2/1 1 2 2/2 2 1 1/2 2 1 1'
def S(**kw): d = dict(matrix=M, tiles=1, cpi=22, wpi=16, stub_seconds=0); d.update(kw); return d
c = sc.Camp.create('L1'); ids = [c.enqueue(S(seed=i)) for i in range(8)]
a = c.claim('kaggle:1', 3, 3600, 'b1'); ida = [x['id'] for x in a]
check('claim gives the first 3 queued runs', ida == ids[:3], ida)
check('claimed runs are skipped by pending()', [x['id'] for x in c.pending()] == ids[3:])
b = c.claim('kaggle:2', 3, 3600, 'b2'); check('a second claim never shares a run', not set(ida) & {x['id'] for x in b})
res = []; th = [threading.Thread(target=lambda: res.extend(x['id'] for x in c.claim('w%d' % k, 1, 3600))) for k in range(6)]; [t.start() for t in th]; [t.join() for t in th]
check('concurrent claims: each run at most once', len(res) == len(set(res)) and len(res) == 2, res)
check('only the holder releases', c.release(ida[:1], 'kaggle:2') == [] and c.release(ida[:1], 'kaggle:1') == ida[:1])
check('a released run is queued again', ida[0] in [x['id'] for x in c.pending()])
c.claim('kaggle:9', 1, -1); check('an expired lease does not hold the run', ida[0] in [x['id'] for x in c.pending()])
check('status counts leases by worker kind', c.status()['leased_by_worker'].get('kaggle') == 5, c.status()['leased_by_worker'])
check('leases.log keeps claims and releases', sum(1 for l in open(c.leaselog)) >= 10)

# remote result import: run a stub run by hand from the frozen copy (as a Kaggle worker does) and import it
rid = ida[1]; q = next(x for x in c.queue_items() if x['id'] == rid); out = os.path.join(TMP, 'remote', rid); os.makedirs(out)
json.dump(dict(q['settings'], fold_min_turn_rad=0.2), open(os.path.join(out, 'settings.json'), 'w'))
p = subprocess.run([sys.executable, os.path.join(c.code_dir(), 'shell_run.py'), '--settings', os.path.join(out, 'settings.json'), '--out', out], capture_output=True, text=True)
check('stub run from the frozen copy', p.returncode == 0 and os.path.isfile(os.path.join(out, 'metrics.json')), p.stderr[-300:])
check('import_result accepts it', c.import_result(rid, out, 'kaggle:1', wall_s=12.0) == 'done' and os.path.isdir(os.path.join(c.runs, rid)))
last = c.index_last()[rid]; check('the index records the worker', last.get('worker') == 'kaggle:1' and last['status'] == 'done')
check('import is idempotent', c.import_result(rid, out, 'kaggle:1') == 'exists')
check('its lease is released', c.lease_of(rid) is None)
rid2 = ida[2]; out2 = os.path.join(TMP, 'remote', rid2); shutil.copytree(out, out2)
try: c.import_result(rid2, out2, 'kaggle:1'); check('refuses a result whose settings differ from the queue', False)
except ValueError as e: check('refuses a result whose settings differ from the queue', 'settings differ' in str(e), str(e))
rid3 = [x['id'] for x in b][0]; q3 = next(x for x in c.queue_items() if x['id'] == rid3); out3 = os.path.join(TMP, 'remote', rid3); shutil.copytree(out, out3)
json.dump(dict(q3['settings'], fold_min_turn_rad=0.2), open(os.path.join(out3, 'settings.json'), 'w')); pv = json.load(open(os.path.join(out3, 'provenance.json'))); pv['shell_sim_sha256'] = '0' * 64
json.dump(pv, open(os.path.join(out3, 'provenance.json'), 'w'))
try: c.import_result(rid3, out3, 'kaggle:2'); check('refuses a result from other code', False)
except ValueError as e: check('refuses a result from other code', 'provenance' in str(e), str(e))
check('status counts done runs by worker', c.status()['done_by_worker'].get('kaggle') == 1)
check('rows carry the worker', next(r for r in c.rows() if r['run'] == rid)['worker'] == 'kaggle:1' and next(r for r in c.rows() if r['run'] == ida[2])['leased'])

# the server runner: leases what it runs, skips leased runs, releases stale server leases at start
c2 = sc.Camp.create('L2'); j = [c2.enqueue(S(seed=100 + i)) for i in range(3)]; c2.claim('kaggle:1', 1, 3600); c2.claim('server', 1, 3600)          # j[1] has a stale server lease
c2.set_control('run'); sc.serve(c2, poll_s=0.1, stop_when_empty=True, lock=False, run_poll=0.05)
st = {r['run']: r['status'] for r in c2.rows()}
check('server runner skipped the Kaggle-leased run', st[j[0]] == 'queued' and st[j[1]] == 'done' and st[j[2]] == 'done', st)
check('server runner left no server lease behind', not [r for r, L in c2.active_leases().items() if L['worker'] == 'server'])
check('server results record worker server', c2.index_last()[j[2]].get('worker') == 'server')
print('ALL OK' if not bad else 'FAILED: %s' % bad); shutil.rmtree(TMP, ignore_errors=True); sys.exit(1 if bad else 0)
