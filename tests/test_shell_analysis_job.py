#!/usr/bin/env python3
"""Tests of the scheduled analysis job (scripts/shell_learning/shell_analysis_job.py) and its read-only API routes (webui/campaign_api.py /api/shell/<c>/analysis...), on SYNTHETIC finished runs in temporary
folders: threshold, counting, idempotence, lock, interrupted attempts, failures, repeats at every 100 runs, nothing written outside analysis/scheduled. No solver, no server.
Run from the repository root: python tests/test_shell_analysis_job.py   (about two minutes: the real tools run twice with small settings)"""
import hashlib, json, os, subprocess, sys, tempfile, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='shellajob_')
os.environ.update(SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), CAMPAIGN_DIR=os.path.join(TMP, 'campaigns'), CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'))
os.makedirs(os.environ['CAMPAIGN_DIR'])
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs; sys.path.insert(0, os.path.join(ROOT, 'webui'))
import shell_campaign as sc, shell_features as sf, shell_analysis_job as aj, campaign as cm
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info), flush=True); ok or bad.append(name)
def tree_hash(paths):
    h = hashlib.sha256()
    for top in paths:
        if os.path.isfile(top): h.update(top.encode()); h.update(open(top, 'rb').read()); continue
        for r, ds, fs in sorted(os.walk(top)):
            ds.sort()
            for f in sorted(fs): p = os.path.join(r, f); h.update(p.encode()); h.update(open(p, 'rb').read())
    return h.hexdigest()
quiet = lambda s: None
dead_pid = subprocess.Popen([sys.executable, '-c', 'pass']); dead_pid.wait(); DEAD = dead_pid.pid          # a process that has exited

# ---- a synthetic seeded campaign (same construction as tests/test_shell_learning.py) and a helper that finishes k more runs
r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'campaigns', 'campaign.py'), 'shell-plan', 'syn', '--policy', os.path.join(ROOT, 'campaigns', 'shell_policy.json'), '--filler', '400', '--seeds', os.path.join(ROOT, 'campaigns', 'shell_seeds', 'seeds.json'), '--pattern1-file', os.path.join(ROOT, 'campaigns', 'pattern1_matrix.txt')], capture_output=True, text=True, env=dict(os.environ))
c = sc.Camp('syn'); rng = np.random.default_rng(1); pending = [x for x in c.queue_items() if not x.get('control')]; controls = [x for x in c.queue_items() if x.get('control')]
check('synthetic campaign planned with more than 500 non-control runs', r.returncode == 0 and len(pending) > 500, (r.stdout + r.stderr)[-200:])
def metrics_for(s):
    ff = sf.features(s['matrix'], s['tiles'], s['cpi'], s['wpi']); fx, fy = 76.2, 69.3
    return dict(thickness=float((0.10 + 0.20 * ff['tm_wale_share'] + 0.05 * ff['log_ratio'] + 0.01 * rng.standard_normal()) * np.sqrt(fx * fy)), flat_x=fx, flat_y=fy, shrink_x=float(-0.5 + 0.1 * ff['knit_fraction'] + 0.01 * rng.standard_normal()),
                shrink_y=float(-0.1 - 0.2 * ff['boundary_density'] + 0.01 * rng.standard_normal()), fold_turn_x_rad=float(0.5 + 0.3 * ff['axis_direction'] + 0.02 * rng.standard_normal()), fold_turn_y_rad=0.01, wave_amp_median_mm=float(1 + 2 * ff['sections_per_100'] / 25))
done = []
def finish(k):
    for x in pending[len(done):len(done) + k]: sc.append_line(c.index, dict(id=x['id'], status='done', t=cm.now(), metrics=metrics_for(x['settings']), wall_s=100.0)); done.append(x['id'])
FAST = dict(proposer_eval=dict(repeats=3), rules_sr=dict(boots=12, perms=2))
calls = []
def fake(name, fail=False):
    def f(rows, params=None):
        calls.append((name, len(rows)))
        if fail: raise RuntimeError('deliberate failure')
        return dict(status='ok', n=len(rows), first=rows[0]['run'], last=rows[-1]['run'])
    return f
FAKES = dict(proposer_eval=fake('proposer_eval'), rules_sr=fake('rules_sr'))

# ---- counting and threshold
finish(99); sc.append_line(c.index, dict(id=controls[0]['id'], status='done', t=cm.now(), metrics=metrics_for(controls[0]['settings']), wall_s=1.0))
sc.append_line(c.index, dict(id=pending[150]['id'], status='failed', t=cm.now(), reason='synthetic failure'))
ds = [x['run'] for x in sf.dataset(c) if not x['control']]
check('counting = the tools\' own counting (shell_features.dataset without controls): 99, the control and the failed run excluded', aj.finished_order(c) == done and sorted(ds) == sorted(done) and len(done) == 99)
h0 = tree_hash([c.runs, c.queue, c.index, c.code_dir(), c.spec]); out = aj.run_once(c, log=quiet, runners=FAKES)
check('99 finished runs: nothing is run, no run-count folder, state says next 100', out['due'] == [] and not calls and not any(n.isdigit() for n in os.listdir(aj.adir(c))) and aj.rjson(os.path.join(aj.adir(c), 'state.json'))['next_milestone'] == 100)
# a run that finishes a second time (index line repeated) is still one run, counted at its first finish
finish(1); sc.append_line(c.index, dict(id=done[3], status='done', t=cm.now(), metrics=metrics_for(pending[3]['settings']), wall_s=100.0, heavy_deleted=True))
check('100 finished runs; a repeated index line for an earlier run does not change the order or the count', aj.finished_order(c) == done and len(aj.finished_order(c)) == 100)

# ---- first analysis with the REAL tools (small settings), proposer first, then rule search
hp = tree_hash([c.runs, c.queue, c.index, c.code_dir(), c.spec]); t0 = time.time(); out = aj.run_once(c, params=FAST, log=quiet); ha = tree_hash([c.runs, c.queue, c.index, c.code_dir(), c.spec]); d100 = aj.mdir(c, 100); st = aj.rjson(os.path.join(d100, 'status.json'))
pe_res = aj.rjson(os.path.join(d100, 'proposer_eval.json')); sr_res = aj.rjson(os.path.join(d100, 'rules_sr.json'))
check('at 100: both real tools ran and wrote their results with the run count they used', out['processed'] == [100] and st['tools']['proposer_eval']['status'] == 'done' and st['tools']['rules_sr']['status'] == 'done'
      and pe_res['scheduled']['n_used'] == 100 and sr_res['scheduled']['n_used'] == 100 and pe_res['status'] == 'ok' and sr_res['label'] == 'learned from the shell, not confirmed on real fabric', '%.0f s' % (time.time() - t0))
check('one at a time: the rule search started after the proposer evaluation finished', st['tools']['rules_sr']['attempts'][0]['started'] >= st['tools']['proposer_eval']['attempts'][0]['finished'])
check('exactly the first 100 finished runs were used, in finishing order', pe_res['scheduled']['run_ids'] == done[:100] and sr_res['n_finished'] == 100)
check('nothing written outside analysis/scheduled: run folders, queue.jsonl, index.jsonl, frozen code and spec.json have the same sha256 before and after the job', hp == ha and os.path.isdir(c.code_dir()))
check('the job wrote only its own results folder inside the campaign folder (analysis/scheduled_v2 since task 53: method version 2 is never mixed with version 1)', set(os.listdir(os.path.join(c.d, 'analysis'))) == {os.path.basename(aj.adir(c))} and aj.adir(c).endswith('scheduled_v2'))
check('no temporary file left behind', not [f for r_, ds_, fs in os.walk(aj.adir(c)) for f in fs if '.tmp' in f])

# ---- idempotence: a second check changes nothing
before = tree_hash([d100]); out = aj.run_once(c, params=FAST, log=quiet, runners=FAKES)
check('idempotent: a second check at 100 runs runs nothing and changes no file of run count 100', out['processed'] == [] and not calls and tree_hash([d100]) == before)
finish(50); out = aj.run_once(c, log=quiet, runners=FAKES)
check('150 finished runs: nothing new (next analysis at 200)', out['processed'] == [] and not calls and aj.rjson(os.path.join(aj.adir(c), 'state.json'))['next_milestone'] == 200)

# ---- lock
import platform_compat as pc
lk = os.path.join(aj.adir(c), 'job.lock'); held = pc.try_lock(os.path.join(aj.adir(c), 'job.flock'))          # task 53d: another live copy holds the OPERATING-SYSTEM lock
finish(50); out = aj.run_once(c, log=quiet, runners=FAKES)
check('a live lock holder (OS lock held): this copy does nothing (no second copy runs)', held is not None and out['status'] == 'locked' and not calls and not os.path.isdir(aj.mdir(c, 200)))
held.close()                                                                                                   # the holder ends: the system releases its lock
aj.wjson_atomic(lk, dict(pid=DEAD, boot_id='x', start_ticks='1', t=cm.now()))         # and its job.lock record stays behind (dead process / previous boot)
out = aj.run_once(c, log=quiet, runners=FAKES); s200 = aj.rjson(os.path.join(aj.mdir(c, 200), 'status.json'))
check('a stale lock (holder dead) is taken over; at 200 runs both tools run on the first 200 runs; the lock is released afterwards', out['processed'] == [200] and calls == [('proposer_eval', 200), ('rules_sr', 200)] and not os.path.exists(lk)
      and aj.rjson(os.path.join(aj.mdir(c, 200), 'proposer_eval.json'))['scheduled']['run_ids'] == done[:200], str(calls))
check('run count 100 is kept unchanged next to 200 (no result is overwritten)', tree_hash([d100]) == before and s200['n_used'] == 200)

# ---- interrupted attempt (power cut while a tool was running) and a cut between result and status
calls.clear(); finish(100); m3 = aj.mdir(c, 300); os.makedirs(m3)
aj.wjson_atomic(os.path.join(m3, 'status.json'), dict(campaign='syn', milestone=300, n_used=300, created=cm.now(), tools=dict(
    proposer_eval=dict(status='running', attempts=[dict(pid=DEAD, boot_id='old', status='running', started=cm.now())], result='proposer_eval.json'),
    rules_sr=dict(status='queued', attempts=[], result='rules_sr.json'))))
open(os.path.join(m3, 'proposer_eval.json.tmp.%d' % DEAD), 'w').write('{"half": ')                          # the half-written temporary file of the cut attempt
out = aj.run_once(c, log=quiet, runners=FAKES); s3 = aj.rjson(os.path.join(m3, 'status.json')); a3 = s3['tools']['proposer_eval']['attempts']
check('interrupted attempt: recorded as interrupted with a reason, the tool runs again and finishes; the half-written temporary file never becomes the result',
      [x['status'] for x in a3] == ['interrupted', 'done'] and a3[0]['reason'] and s3['tools']['proposer_eval']['status'] == 'done' and aj.rjson(os.path.join(m3, 'proposer_eval.json'))['n'] == 300, str([x['status'] for x in a3]))
calls.clear(); finish(100); m4 = aj.mdir(c, 400); os.makedirs(m4)
aj.wjson_atomic(os.path.join(m4, 'proposer_eval.json'), dict(status='ok', n=400))
aj.wjson_atomic(os.path.join(m4, 'status.json'), dict(campaign='syn', milestone=400, n_used=400, created=cm.now(), tools=dict(proposer_eval=dict(status='running', attempts=[dict(pid=DEAD, status='running', started=cm.now())], result='proposer_eval.json'), rules_sr=dict(status='queued', attempts=[], result='rules_sr.json'))))
out = aj.run_once(c, log=quiet, runners=FAKES); s4 = aj.rjson(os.path.join(m4, 'status.json'))
check('cut after the result was renamed into place but before the status: the complete result counts, the tool is not rerun', calls == [('rules_sr', 400)] and s4['tools']['proposer_eval']['status'] == 'done', str(calls))

# ---- MAX_ATTEMPTS and a failing tool
calls.clear(); finish(100); m5 = aj.mdir(c, 500); os.makedirs(m5)
atts = [dict(pid=DEAD, status='interrupted', started=cm.now(), reason='cut') for _ in range(aj.MAX_ATTEMPTS - 1)] + [dict(pid=DEAD, status='running', started=cm.now())]
aj.wjson_atomic(os.path.join(m5, 'status.json'), dict(campaign='syn', milestone=500, n_used=500, created=cm.now(), tools=dict(proposer_eval=dict(status='running', attempts=atts, result='proposer_eval.json'), rules_sr=dict(status='queued', attempts=[], result='rules_sr.json'))))
out = aj.run_once(c, log=quiet, runners=dict(FAKES, rules_sr=fake('rules_sr', fail=True))); s5 = aj.rjson(os.path.join(m5, 'status.json'))
check('after %d interrupted attempts the tool is marked failed and not started again' % aj.MAX_ATTEMPTS, s5['tools']['proposer_eval']['status'] == 'failed' and 'interrupted' in s5['tools']['proposer_eval']['reason'] and ('proposer_eval', 500) not in calls)
check('a tool that raises is recorded as failed with the error text; no result file is written', s5['tools']['rules_sr']['status'] == 'failed' and 'deliberate failure' in s5['tools']['rules_sr']['reason'] and not os.path.exists(os.path.join(m5, 'rules_sr.json')))
calls.clear(); out = aj.run_once(c, log=quiet, runners=FAKES)
check('a failed run count is not repeated at the next check', out['processed'] == [] and not calls)

# ---- skipped run counts: a campaign that jumps past several multiples gets each one, on its own first m runs
c2 = sc.Camp('syn2'); sc.Camp.create('syn2', c.spec_())
for x in pending[:230]: c2.enqueue(x['settings'], source=x.get('source', 'test'), tier=x.get('tier', 5))
q2 = [x['id'] for x in c2.queue_items()]
for rid in q2[:230]: sc.append_line(c2.index, dict(id=rid, status='done', t=cm.now(), metrics=metrics_for(next(y for y in c2.queue_items() if y['id'] == rid)['settings']), wall_s=1.0))
calls.clear(); out = aj.run_once(c2, log=quiet, runners=FAKES)
check('230 finished runs at the first check: run counts 100 and 200 are both computed, each on its own first runs', out['processed'] == [100, 200] and calls == [('proposer_eval', 100), ('rules_sr', 100), ('proposer_eval', 200), ('rules_sr', 200)], str(calls))

# ---- the read-only API routes
from fastapi import FastAPI
from fastapi.testclient import TestClient
import campaign_api as api
app = FastAPI(); api.setup(app, lambda: False); cl = TestClient(app)
hb = tree_hash([aj.adir(c)]); a = cl.get('/api/shell/syn/analysis').json()
check('API: counted runs, next run count, run counts 100-500 in order, per-tool status', a['finished'] == len(done) and a['next_milestone'] == (len(done) // 100 + 1) * 100 and [m['milestone'] for m in a['milestones']] == [100, 200, 300, 400, 500]
      and a['milestones'][0]['tools']['rules_sr']['status'] == 'done' and a['milestones'][4]['tools']['rules_sr']['status'] == 'failed' and 'deliberate' in a['milestones'][4]['tools']['rules_sr']['reason'], str([m['milestone'] for m in a['milestones']]))
check('API: an interrupted earlier attempt is visible in the attempts of run count 300', [x['status'] for x in a['milestones'][2]['tools']['proposer_eval']['attempts']] == ['interrupted', 'done'])
check('API: latest complete result per tool (proposer 400, rule search 400) and the label', a['latest']['proposer_eval']['milestone'] == 400 and a['latest']['rules_sr']['milestone'] == 400 and a['label'] == 'learned from the shell, not confirmed on real fabric')
one = cl.get('/api/shell/syn/analysis/100').json()
check('API history: run count 100 returns both real results (verdict and rule-search targets)', one['results']['proposer_eval']['verdict_short'] in ('better than random', 'no different from random', 'worse than random') and len(one['results']['rules_sr']['targets']) == 6)
check('API: unknown or malformed run count gives 404', cl.get('/api/shell/syn/analysis/700').status_code == 404 and cl.get('/api/shell/syn/analysis/..%2F..').status_code in (404, 422) and cl.get('/api/shell/syn/analysis/abc').status_code == 404)
c3 = sc.Camp.create('fresh'); e = cl.get('/api/shell/fresh/analysis').json()
check('API on a campaign without runs: waiting, no run counts, scheduler never run', e['finished'] == 0 and e['milestones'] == [] and e['next_milestone'] == 100 and e['scheduler']['last_check'] is None and e['scheduler']['alive'] is False)
check('API reading changed nothing', tree_hash([aj.adir(c)]) == hb)
check('the existing rules route still answers as before (not run yet: status and label)', cl.get('/api/shell/syn/rules').json().get('status') == 'not run yet')
check('the API has no route that starts a tool (only GET routes under /analysis)', all(set(rt.methods) <= {'GET', 'HEAD'} for rt in app.routes if '/analysis' in getattr(rt, 'path', '')))
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
