#!/usr/bin/env python3
"""Tests of the pure-shell campaign machinery (scripts/campaigns/shell_campaign.py, shell_run.py, shell_metrics.py) WITHOUT the solver: shell_run.py runs in its stub mode
(SHELL_RUN_STUB=1: a synthetic sheet, no JAX). Everything happens in a temporary folder; nothing under campaigns/ or scripts/ is written. Run from the repository root: python tests/test_shell_campaign.py"""
import json, os, struct, sys, tempfile, threading, time, glob
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='shelltest_')
os.environ['SHELL_RUNS_DIR'] = os.path.join(TMP, 'shell_runs'); os.environ['SHELL_RUN_STUB'] = '1'; os.environ['CAMPAIGN_LOG_DIR'] = os.path.join(TMP, 'logs')
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs
import shell_campaign as sc, shell_metrics as sm, campaign as cm
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)
def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False

M = '1 1 2 2/1 1 2 2/2 2 1 1/2 2 1 1'
def S(**kw): d = dict(matrix=M, tiles=1, cpi=22, wpi=16, stub_seconds=0); d.update(kw); return d
n = [0]
def newcamp(**spec):
    n[0] += 1; return sc.Camp.create('t%d' % n[0], spec or None)

# ---- settings
s1 = sc.normalise(S()); s2 = sc.normalise(S(sub=2, steps=15, kappa_scale=1.0))
check('normalise fills defaults, equal settings hash equally', sc.sha(s1) == sc.sha(s2) and s1['iter'] == 400 and s1['nu'] == 0.4)
check('matrix text parsed (rows /, spaces optional)', sc.matrix_from_text('112/221') == [[1, 1, 2], [2, 2, 1]])
for label, bs in (('non-rectangular', S(matrix='1 1/1')), ('digit 3', S(matrix='1 3/1 1')), ('cpi too small', S(cpi=1)), ('too many stitches', S(matrix='/'.join(['1' * 50] * 50), tiles=2)), ('kappa_scale 0', S(kappa_scale=0)), ('missing wpi', dict(matrix=M, tiles=1, cpi=22))):
    check('refused: ' + label, raises(lambda bs=bs: sc.normalise(bs), ValueError))
check('timeout rule: floor 900 s, 4 x predicted above it', sc.timeout_for(s1, sc.DEFAULT_LIMITS) == 900.0 and abs(sc.timeout_for(sc.normalise(S(matrix='/'.join(['1' * 48] * 60), tiles=1)), sc.DEFAULT_LIMITS) - 4 * 285 * (97 * 121) / 11737) < 1e-6)

# ---- ids, queue, duplicates
c = newcamp()
ids = [c.enqueue(S(seed=i)) for i in range(5)]
check('ids S000001.. consecutive', ids == ['S%06d' % k for k in range(1, 6)], str(ids))
check('same settings are not queued twice', c.enqueue(S(seed=3)) is None and c.enqueue(S(seed=3), allow_duplicate=True) == 'S000006')
res = []
def worker(k): res.append(c.next_id())
th = [threading.Thread(target=worker, args=(k,)) for k in range(20)]; [t.start() for t in th]; [t.join() for t in th]
check('concurrent next_id: 20 distinct ids', len(set(res)) == 20, str(len(set(res))))
os.remove(c.nextid); check('counter lost: next id still above every queued id', int(c.next_id()[1:]) > max(int(x['id'][1:]) for x in c.queue_items()))
check('a campaign is never overwritten', raises(lambda: sc.Camp.create(c.name), FileExistsError))
check('bad campaign names refused', raises(lambda: sc.Camp('../x'), ValueError) and raises(lambda: sc.Camp('a b'), ValueError))

# ---- runner end to end with the stub
c = newcamp(); [c.enqueue(S(seed=i)) for i in range(3)]; c.set_control('run')
sc.serve(c, poll_s=0.2, stop_when_empty=True, lock=False, run_poll=0.1)
runs = sorted(os.listdir(c.runs)); check('3 runs published', runs == ['S000001', 'S000002', 'S000003'], str(runs))
r1 = os.path.join(c.runs, 'S000001'); fs = sorted(os.listdir(r1))
check('run folder holds the expected files', all(f in fs for f in ('shell.npz', 'shell.bin', 'metrics.json', 'settings.json', 'status.json', 'shell.log', 'worker.json')), str(fs))
check('no .tmp folder left', os.listdir(c.tmp) == [])
st = c.status(); check('status counts', st['counts']['done'] == 3 and st['counts']['queued'] == 0 and st['mean_wall_s'] is not None, str(st['counts']))
check('results.csv has one row per run', len(open(c.results).read().strip().splitlines()) == 4)
rows = c.rows(); check('row carries settings and descriptors', rows[0]['status'] == 'done' and rows[0]['plan_x'] and rows[0]['cpi'] == 22.0)
before = sorted(os.listdir(c.runs)); sc.serve(c, poll_s=0.2, stop_when_empty=True, lock=False)
check('restart skips finished runs and repeats nothing', sorted(os.listdir(c.runs)) == before and c.pending() == [])
check('publish refuses to overwrite an existing run folder', raises(lambda: (os.makedirs(os.path.join(c.tmp, 'S000001.9')), c.publish(os.path.join(c.tmp, 'S000001.9'), 'S000001', {})), FileExistsError))
os.rename(os.path.join(c.tmp, 'S000001.9'), os.path.join(c.quar, 'S000001.manual'))

# ---- controls: pause, continue, stop after current
c = newcamp(); [c.enqueue(S(seed=i, stub_seconds=0.6)) for i in range(4)]; c.set_control('pause')
t = threading.Thread(target=lambda: sc.serve(c, poll_s=0.1, lock=False, run_poll=0.05)); t.start(); time.sleep(1.0)
check('pause: nothing starts', not os.listdir(c.runs) and c.get_runner_state() == 'paused', c.get_runner_state())
c.set_control('run'); time.sleep(0.9); c.set_control('pause')           # in the middle of a run: pause lets it finish
time.sleep(1.2); k = len(os.listdir(c.runs)); time.sleep(1.0)
check('pause during a run: the run finishes, no further run starts', 1 <= k <= 2 and len(os.listdir(c.runs)) == k and c.get_runner_state() == 'paused', 'finished %d' % k)
c.set_control('run'); time.sleep(0.5); c.set_control('stop'); t.join(timeout=15)
check('stop after current run: runner exits after the run in progress', not t.is_alive() and c.get_runner_state() == 'stopped' and os.listdir(c.tmp) == [] and 0 < len(os.listdir(c.runs)) < 4, str(len(os.listdir(c.runs))))

# ---- per-run timeout: the stuck run is killed and logged, the next run still happens
c = newcamp(limits=dict(timeout_floor_s=1.5, timeout_factor=0.0)); a = c.enqueue(S(seed=1, stub_hang=True)); b = c.enqueue(S(seed=2)); c.set_control('run')
t0 = time.time(); sc.serve(c, poll_s=0.2, stop_when_empty=True, lock=False, run_poll=0.1)
last = c.index_last()
check('timeout: stuck run logged as timeout', last[a]['status'] == 'timeout' and 'timeout' in last[a]['reason'], str(last[a]))
check('timeout: partial folder moved to quarantine, nothing in runs/', len(glob.glob(os.path.join(c.quar, a + '.timeout.*'))) == 1 and a not in os.listdir(c.runs))
check('timeout: runner continued with the next run', last[b]['status'] == 'done' and b in os.listdir(c.runs), 'took %.1f s' % (time.time() - t0))
c2 = newcamp(); f = c2.enqueue(S(seed=1, stub_fail=True)); c2.set_control('run'); sc.serve(c2, poll_s=0.2, stop_when_empty=True, lock=False, run_poll=0.1)
check('a crashing run is logged as failed with its exit code', c2.index_last()[f]['status'] == 'failed' and 'exit 3' in c2.index_last()[f]['reason'])

# ---- free-disk guard
c = newcamp(); c.enqueue(S()); c.set_control('run'); os.environ['SHELL_MIN_FREE_GB'] = '1e9'
sc.serve(c, poll_s=0.2, lock=False); os.environ.pop('SHELL_MIN_FREE_GB')
check('disk guard: runner stops itself, starts no run, control set to stop, reason recorded', c.get_runner_state() == 'guard_stopped' and c.get_control() == 'stop' and not os.listdir(c.runs) and 'free disk' in (cm.rjson(c.runner) or {}).get('reason', ''))
os.environ['SHELL_MAX_TOTAL_GB'] = '0'; check('total-size guard: the shell-runs folder above its limit stops the runner', 'above the limit' in sc.disk_guard(c), sc.disk_guard(c)); os.environ.pop('SHELL_MAX_TOTAL_GB')

# ---- power cut: interrupted attempts, complete-but-unpublished attempts, published-but-unindexed runs
c = newcamp(); a = c.enqueue(S(seed=1)); b = c.enqueue(S(seed=2)); d_ = c.enqueue(S(seed=3))
os.makedirs(os.path.join(c.tmp, a + '.1')); open(os.path.join(c.tmp, a + '.1', 'shell.log'), 'w').write('growth 0.07: U 1')           # cut mid-run
tb = os.path.join(c.tmp, b + '.1'); os.makedirs(tb); json.dump(dict(plan_x=1.0), open(os.path.join(tb, 'metrics.json'), 'w'))           # cut after the solver, before the rename
os.makedirs(os.path.join(c.runs, d_)); json.dump(dict(plan_x=2.0), open(os.path.join(c.runs, d_, 'metrics.json'), 'w'))                    # cut between rename and index line
check('reconcile returns empty (no live worker)', sc.reconcile(c) == '')
check('interrupted attempt: renamed into quarantine, run requeued, nothing deleted', len(glob.glob(os.path.join(c.quar, a + '.interrupted.*'))) == 1 and [x['id'] for x in c.pending()] == [a])
check('complete attempt of a dead worker is published, not rerun', b in os.listdir(c.runs) and c.index_last()[b]['status'] == 'done' and c.index_last()[b].get('recovered'))
check('published but unindexed run gets its index line', c.index_last()[d_]['status'] == 'done')
for k in range(2): os.makedirs(os.path.join(c.tmp, a + '.%d' % (k + 2))); sc.reconcile(c)
check('after 3 interruptions the run is marked needs_approval and skipped', c.index_last()[a]['status'] == 'needs_approval' and c.pending() == [])

# ---- delete (heavy files only) and its path guard
c = newcamp(); [c.enqueue(S(seed=i)) for i in range(3)]; c.set_control('run'); sc.serve(c, poll_s=0.2, stop_when_empty=True, lock=False, run_poll=0.1)
sz = sc.delete_heavy(c, ['S000001', 'S000002'], dry=True)
check('dry run reports ids and bytes, removes nothing', sz['ids'] == ['S000001', 'S000002'] and sz['bytes'] > 0 and os.path.exists(os.path.join(c.runs, 'S000001', 'shell.npz')))
for label, ids in (('traversal', ['../S000001']), ('traversal 2', ['S000001/../S000002']), ('absolute path', [os.path.join(c.runs, 'S000001')]), ('wrong shape', ['S1']), ('unknown run', ['S000099']), ('empty list', []), ('not a list', 'S000001'), ('non-string', [1])):
    check('delete refused: ' + label, raises(lambda ids=ids: sc.delete_heavy(c, ids), ValueError))
out = sc.delete_heavy(c, ['S000001', 'S000002'], client='test')
check('delete removes only the heavy files', not os.path.exists(os.path.join(c.runs, 'S000001', 'shell.npz')) and not os.path.exists(os.path.join(c.runs, 'S000001', 'shell.bin')) and os.path.exists(os.path.join(c.runs, 'S000001', 'metrics.json')) and os.path.exists(os.path.join(c.runs, 'S000003', 'shell.npz')))
lines = sc.read_lines(c.delog); check('deletion log written (ids, bytes, client)', len(lines) == 1 and lines[0]['ids'] == ['S000001', 'S000002'] and lines[0]['client'] == 'test' and lines[0]['bytes'] == out['bytes'])
check('index keeps the run with heavy_deleted and its descriptors', c.index_last()['S000001']['heavy_deleted'] and c.index_last()['S000001']['status'] == 'done' and c.index_last()['S000001']['metrics'].get('plan_x'))
check('second delete of the same run is harmless', sc.delete_heavy(c, ['S000001'])['bytes'] == 0)
if hasattr(os, 'symlink'):
    try:
        os.symlink(TMP, os.path.join(c.runs, 'S000009'), target_is_directory=True)
        check('delete refused: a symlink in runs/', raises(lambda: sc.delete_heavy(c, ['S000009']), ValueError))
    except (OSError, NotImplementedError): print('SKIP  symlink test (no permission to create symlinks)')
cur = newcamp(); cur.enqueue(S()); sc.Camp.create  # noqa
os.makedirs(os.path.join(cur.runs, 'S000001')); open(os.path.join(cur.runs, 'S000001', 'shell.npz'), 'w').write('x')
json.dump(dict(pid=os.getpid(), boot_id=cm.boot_id(), start_ticks=(cm.proc_info(os.getpid()) or ['', ''])[0]), open(cur.lock, 'w')); cur.set_runner(state='running', current='S000001')
if cm.proc_info(os.getpid()): check('delete refused for the run in progress', raises(lambda: sc.delete_heavy(cur, ['S000001']), ValueError))
else: print('SKIP  in-progress delete test (needs /proc)')

# ---- busy message (a running shell runner blocks other jobs; a paused one does not)
if cm.proc_info(os.getpid()):
    check('a live runner in state running blocks other jobs (message names the campaign)', cur.name in sc.busy_message(), sc.busy_message())
    cur.set_runner(state='paused', current=None); check('a live but paused runner does not block (it checks busy_reason before each run)', sc.busy_message() == '')
else: check('without a live runner nothing blocks', sc.busy_message() == '')

# ---- frozen copy of the code and provenance (a later git pull cannot change a running campaign)
import subprocess
SCR = os.path.join(ROOT, 'scripts'); c = newcamp(); man = c.manifest()
check('freeze: code/ holds the 5 files, byte copies of scripts/', all(sc.file_sha256(os.path.join(c.code_dir(), f)) == sc.file_sha256(sc.frozen_source(os.path.join(SCR, 'campaigns'), f)) for f in sc.FROZEN_FILES))
check('MANIFEST.json: sha256 of every file, shell_sim hash, git commit and dirty flag, versions', set(man['files']) == set(sc.FROZEN_FILES) and man['shell_sim_sha256'] == sc.file_sha256(os.path.join(SCR, 'shell_model', 'shell_sim.py')) and 'git_commit' in man and 'git_dirty' in man and man['python'], str({k: man[k] for k in ('git_commit', 'git_dirty')}))
check('freeze never overwrites (new code = new campaign)', raises(c.freeze, FileExistsError))
check('intact frozen copy verifies', c.verify_code() == '')
c.enqueue(S()); c.set_control('run'); sc.serve(c, poll_s=0.2, stop_when_empty=True, lock=False, run_poll=0.1)
pv = cm.rjson(os.path.join(c.runs, 'S000001', 'provenance.json')); ix = c.index_last()['S000001']; st = cm.rjson(os.path.join(c.runs, 'S000001', 'status.json'))
check('every run folder has provenance.json with the shell_sim.py hash, git fields and versions', pv and pv['shell_sim_sha256'] == man['shell_sim_sha256'] and 'git_commit' in pv and pv['python'] and set(pv['files']) == set(sc.FROZEN_FILES), str(sorted(pv or {})))
check('status.json, the index line and results.csv carry the commit and the shell_sim hash', st['code']['shell_sim_sha256'] == man['shell_sim_sha256'] and ix['code']['shell_sim_sha256'] == man['shell_sim_sha256'] and 'shell_sim_sha256' in open(c.results).readline() and 'git_commit' in open(c.results).readline())
fc = newcamp(); fc.enqueue(S()); fc.set_control('run'); open(os.path.join(fc.code_dir(), 'shell_metrics.py'), 'a').write('\n# edited after freezing\n')
check('verify_code names the changed file', 'shell_metrics.py' in fc.verify_code())
sc.serve(fc, poll_s=0.2, lock=False, run_poll=0.1)
check('changed frozen code: the runner starts no run, stops itself (control = stop) and says why', not os.listdir(fc.runs) and fc.get_runner_state() == 'guard_stopped' and fc.get_control() == 'stop' and 'frozen code' in (cm.rjson(fc.runner) or {}).get('reason', ''))
os.makedirs(os.path.join(TMP, 'w')); wj = os.path.join(TMP, 'w', 's.json'); json.dump(sc.normalise(S()), open(wj, 'w'))
fz = newcamp(); r = subprocess.run([sys.executable, os.path.join(fz.code_dir(), 'shell_run.py'), '--settings', wj, '--out', os.path.join(TMP, 'w', 'o')], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
pw = cm.rjson(os.path.join(TMP, 'w', 'o', 'provenance.json')) or {}
check('a worker started from the frozen copy records frozen = true and matches_manifest = true', r.returncode == 0 and pw.get('frozen') is True and pw.get('matches_manifest') is True and pw['shell_sim_sha256'] == fz.manifest()['shell_sim_sha256'], r.stderr[-200:])
check('no .pyc written into the frozen code folder (the runner starts workers with PYTHONDONTWRITEBYTECODE=1)', not os.path.exists(os.path.join(fz.code_dir(), '__pycache__')) and fz.verify_code() == '')
env0 = {k: v for k, v in os.environ.items() if k != 'SHELL_RUNS_DIR'}
r = subprocess.run([sys.executable, os.path.join(fz.code_dir(), 'campaign.py'), 'shell-status', fz.name], capture_output=True, text=True, env=env0)
check('the frozen copy finds its own campaign without SHELL_RUNS_DIR (root derived from its location)', r.returncode == 0 and json.loads(r.stdout)['name'] == fz.name, (r.stderr or r.stdout)[-200:])
r = subprocess.run([sys.executable, os.path.join(fz.code_dir(), 'campaign.py'), 'shell-run', 'some_other_campaign', '--poll-s', '0.2'], capture_output=True, text=True, env=env0)
check('a frozen copy refuses to run another campaign', r.returncode != 0 and 'refuses to run' in (r.stderr + r.stdout), (r.stderr + r.stdout)[-160:])
if os.name == 'posix':           # the runner locks with flock and re-executes into the frozen copy: Linux only
    pc = newcamp(); pc.enqueue(S()); pc.set_control('run')
    r = subprocess.run([sys.executable, os.path.join(SCR, 'campaigns', 'campaign.py'), 'shell-run', pc.name, '--stop-when-empty', '--poll-s', '0.2'], capture_output=True, text=True)
    pp = cm.rjson(os.path.join(pc.runs, 'S000001', 'provenance.json')) or {}
    check('shell-run from the repo re-executes into the frozen copy: the run records frozen = true', r.returncode == 0 and pp.get('frozen') is True and pp.get('matches_manifest') is True, (r.stderr + r.stdout)[-300:])
else: print('SKIP  runner re-exec test (Linux only)')
nf = newcamp()
import shutil; shutil.move(nf.code_dir(), nf.code_dir() + '_moved')
r = subprocess.run([sys.executable, os.path.join(SCR, 'campaigns', 'campaign.py'), 'shell-run', nf.name, '--stop-when-empty'], capture_output=True, text=True)
check('a campaign without a frozen copy is refused unless --allow-unfrozen', r.returncode != 0 and 'frozen' in (r.stderr + r.stdout), (r.stderr + r.stdout)[-160:])

# ---- metrics and export
jj, ii = np.meshgrid(np.arange(121), np.arange(97), indexing='ij'); W, C = 1.5875, 1.1545
flat = np.stack([ii * W / 2, jj * C / 2, np.zeros_like(ii, float)], -1)
d = sm.describe(flat, W, C, 2); check('flat sheet: no fold, no shrink', d['thickness'] == 0 and abs(d['shrink_x']) < 1e-12 and d['fold_direction'] == 'none', str(d['fold_direction']))
cyl = flat.copy(); R = 20.0; th = flat[..., 0] / R; cyl[..., 0] = R * np.sin(th); cyl[..., 2] = R * (1 - np.cos(th)); d = sm.describe(cyl, W, C, 2)
check('rolled cylinder: course-direction fold, plan shrinks', d['fold_direction'] in ('course', 'both') and d['shrink_x'] < 0 and d['fold_kx_per_mm'] > 0, '%s kx %.4f (1/R=%.4f)' % (d['fold_direction'], d['fold_kx_per_mm'], 1 / R))
wav = flat.copy(); wav[..., 0] += 2.0 * np.sin(2 * np.pi * 3 * jj / 120); d = sm.describe(wav, W, C, 2)
check('sideways wave of 3 periods is recovered, amplitude about 2 mm (the linear detrend of a finite sine adds up to about 15 %)', d['wave_k'] == 3 and abs(d['wave_amp_median_mm'] - 2.0) < 0.5, 'k %s amp %s' % (d['wave_k'], d['wave_amp_median_mm']))
def build_tris(nx, ny):          # the loop of shell_sim.build, copied
    gx = nx + 1; tri = []
    for j in range(ny):
        for i in range(nx):
            a = j * gx + i; b = a + 1; dd = a + gx; cc = dd + 1
            if (i + j) % 2 == 0: tri += [(a, b, cc), (a, cc, dd)]
            else: tri += [(a, b, dd), (b, cc, dd)]
    return np.array(tri)
check('triangle list equals shell_sim.build', np.array_equal(sm.tri_index(6, 5), build_tris(6, 5)) and np.array_equal(sm.tri_index(48 * 2, 60 * 2), build_tris(96, 120)))
bts = sm.kmsh_bytes(flat); nv, nf = struct.unpack_from('<II', bts, 8)
check('KMSH v1 header and size as build_knit.write_obj / viewer_core.loadTubeBase', bts[:4] == b'KMSH' and struct.unpack_from('<I', bts, 4)[0] == 1 and nv == 121 * 97 and nf == 2 * 120 * 96 and len(bts) == 16 + nv * 12 + nf * 12)
pos = np.frombuffer(bts, '<f4', nv * 3, 16).reshape(-1, 3); idx = np.frombuffer(bts, '<u4', nf * 3, 16 + nv * 12)
check('KMSH vertices and indices round trip', np.allclose(pos, flat.reshape(-1, 3), atol=1e-5) and idx.max() == nv - 1)

# ---- shell_run.py as a subprocess (stub): files, order, nothing outside --out
import subprocess
o = os.path.join(TMP, 'one'); sp = os.path.join(TMP, 'one_settings.json'); json.dump(sc.normalise(S()), open(sp, 'w'))
r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'campaigns', 'shell_run.py'), '--settings', sp, '--out', o], capture_output=True, text=True)
z = np.load(os.path.join(o, 'shell.npz'))
check('shell_run.py (stub) exits 0 and writes shell.npz with the keys relax_job writes', r.returncode == 0 and set(z.files) == {'X', 'W', 'C', 'sub', 'nx', 'ny', 'kappa', 'h'} and z['X'].shape == (9, 9, 3), r.stderr[-200:])
check('metrics.json is written last (newest file)', max(os.listdir(o), key=lambda f: os.path.getmtime(os.path.join(o, f))) in ('metrics.json',))

print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad)))
sys.exit(1 if bad else 0)
