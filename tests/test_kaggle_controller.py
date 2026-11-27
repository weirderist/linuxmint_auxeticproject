#!/usr/bin/env python3
"""Tests of scripts/kaggle/kaggle_controller.py (2026-11-27) with a FAKE Kaggle CLI (no network) and the stub solver (SHELL_RUN_STUB=1). Everything in a temporary folder. Run from the repository root:
    python tests/test_kaggle_controller.py
Checks: disabled = nothing claimed; a push claims a batch for the slot; while Kaggle is unreachable nothing changes; on completion the done runs are imported with worker kaggle:<slot>, a failed run is
marked failed, the rest of the batch goes back to the queue; a failed push releases its leases; a session lost for longer than budget + margin is closed and its runs requeued; the worker template runs
(on this machine, stub solver) and writes the manifest the controller reads."""
import json, os, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='kctl_')
os.environ.update(SHELL_RUNS_DIR=os.path.join(TMP, 'shell_runs'), SHELL_RUN_STUB='1', CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'), AUXETIC_ENV=os.path.join(TMP, 'env'))
open(os.path.join(TMP, 'env'), 'w').write('KAGGLE_USERNAME=tester\nKAGGLE_KEY=KGAT_fake\n')
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'kaggle')); sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs
import kaggle_cli as kc, kaggle_controller as kctl, shell_campaign as sc, campaign as cm
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)

F = dict(net=True, push_ok=True, status={}, outputs={}, calls=[])
def fake(args, timeout=600, cwd=None):
    F['calls'].append(list(args))
    if not F['net']: return 1, 'ConnectionError: network unreachable'
    if args[:2] == ['datasets', 'status']: return 0, 'ready'
    if args[:2] == ['kernels', 'push']:
        if not F['push_ok']: return 1, '403 Forbidden'
        md = json.load(open(os.path.join(args[3], 'kernel-metadata.json'))); F['status'][md['id']] = 'running'; F['pushed'] = args[3]; return 0, 'Kernel version 1 successfully pushed.'
    if args[:2] == ['kernels', 'status']: return 0, '%s has status "KernelWorkerStatus.%s"' % (args[2], F['status'].get(args[2], 'running').upper())
    if args[:2] == ['kernels', 'output']:
        src = F['outputs'].get(args[2])
        if src: shutil.copytree(src, args[4], dirs_exist_ok=True)
        return 0, 'Output file downloaded'
    return 1, 'unknown call %s' % args
kc.run = fake; kctl.kc.run = fake

M = '1 1 2 2/1 1 2 2/2 2 1 1/2 2 1 1'
c = sc.Camp.create('K1'); ids = [c.enqueue(dict(matrix=M, tiles=1, cpi=22, wpi=16, seed=i, stub_seconds=0)) for i in range(6)]
k = kctl.Ctl('K1'); cfg = dict(enabled=False, slots=[dict(name='g', gpu=True)], max_batch=3, budget_s=dict(cpu=3600, gpu=3600), est_s_per_stitch=dict(cpu=1.0, gpu=1.0), lease_margin_s=600)
def setcfg(**kw): cfg.update(kw); cm.wjson(os.path.join(k.d, 'config.json'), cfg)
setcfg(); k.tick(); check('disabled: nothing claimed, nothing pushed', not c.active_leases() and not any(a[:2] == ['kernels', 'push'] for a in F['calls']))
setcfg(enabled=True); k.tick(); st = k.state()
check('enabled: a batch of max_batch runs is claimed and pushed', st['g']['status'] == 'running' and st['g']['ids'] == ids[:3] and set(c.active_leases()) == set(ids[:3]), st.get('g'))
src = open(os.path.join(F['pushed'], 'worker.py')).read(); check('the pushed worker has the batch filled in', '__BATCH__' not in src and ids[0] in src)
F['net'] = False; k.tick(); check('Kaggle unreachable: slot still running, leases kept', k.state()['g']['status'] == 'running' and set(c.active_leases()) == set(ids[:3]))
F['net'] = True
L = c.lease_of(ids[0]); L['expires'] = __import__('time').time() + 5; cm.wjson(os.path.join(c.leases, ids[0] + '.json'), L); k.tick()
check('a session Kaggle reports running gets its leases renewed', c.lease_of(ids[0])['expires'] > __import__('time').time() + 300 and k.state()['g']['status'] == 'running')

# run the generated worker here (stub solver) to make the output folder the controller downloads; its code folder is the campaign's frozen copy
kw = os.path.join(TMP, 'kaggle'); os.makedirs(os.path.join(kw, 'input', 'code')); os.makedirs(os.path.join(kw, 'working'))
for f in os.listdir(c.code_dir()): shutil.copy2(os.path.join(c.code_dir(), f), os.path.join(kw, 'input', 'code', f))
w = src.replace("'/kaggle/input/**/shell_run.py'", repr(os.path.join(kw, 'input', '**', 'shell_run.py'))).replace("W = '/kaggle/working'", 'W = %r' % os.path.join(kw, 'working'))
expr = w.split('BATCH = ', 1)[1].split('          #', 1)[0]; b = eval(expr, {'json': json}); b[1]['settings']['stub_fail'] = True
w = w.replace(expr, 'json.loads(%r)' % json.dumps(b), 1)
open(os.path.join(kw, 'worker.py'), 'w').write(w); p = subprocess.run([sys.executable, os.path.join(kw, 'worker.py')], capture_output=True, text=True, env=dict(os.environ, JAX_PLATFORMS='cpu'))
man = json.load(open(os.path.join(kw, 'working', 'manifest.json')))
check('worker template runs and writes its manifest', p.returncode == 0 and man.get('finished') and man['runs'][ids[0]]['status'] == 'done' and man['runs'][ids[1]]['status'] == 'failed', p.stderr[-400:])
# the second run failed on purpose; the worker gave it settings with stub_fail, so make its import use the queued settings for run 0 and 2 only
mp = os.path.join(kw, 'working', 'manifest.json'); mj = json.load(open(mp)); mj['runs'][ids[2]] = dict(status='failed', rc='timeout', wall_s=99.0); json.dump(mj, open(mp, 'w'))          # a Kaggle-side timeout
F['outputs'][('tester/' + k.state()['g']['slug'])] = os.path.join(kw, 'working'); F['status'][('tester/' + k.state()['g']['slug'])] = 'complete'; k.tick(); last = c.index_last()
check('on completion: done runs imported with the worker', last[ids[0]]['status'] == 'done' and last[ids[0]].get('worker') == 'kaggle:g', {i: last.get(i, {}).get('status') for i in ids[:3]})
check('a Kaggle timeout is not marked failed: the run goes back to the queue', ids[2] not in last, last.get(ids[2]))
check('a failed Kaggle run is marked failed', last[ids[1]]['status'] == 'failed' and 'kaggle run failed' in last[ids[1]].get('reason', ''))
check('the slot is idle again, no lease of that batch left', k.state()['g']['status'] in ('idle', 'running') and not (set(ids[:2]) & set(c.active_leases())))
check('the next tick pushed the next batch (the requeued run first)', k.state()['g']['status'] == 'running' and k.state()['g']['ids'] == [ids[2]] + ids[3:5], k.state()['g'].get('ids'))

# lost session: older than budget + margin, no output: closed and requeued
st = k.state(); st['g']['t_push'] -= 10 ** 6; k.save(st); F['status'][('tester/' + k.state()['g']['slug'])] = 'running'; F['outputs'].pop(('tester/' + k.state()['g']['slug']), None); F['push_ok'] = False; k.tick()
check('a session overdue beyond the hard limit is closed and its runs go back to the queue', k.state()['g']['status'] == 'idle' and not c.active_leases() and {x['id'] for x in c.pending()} == {ids[2]} | set(ids[3:6]), k.state()['g'])
check('a failed push releases its claim', not c.active_leases())
ev = [json.loads(l)['event'] for l in open(os.path.join(k.d, 'events.jsonl'))]; check('events.jsonl records pushes, finishes and the failed push', 'pushed' in ev and 'finished' in ev and 'push_failed' in ev, ev)
# finished on Kaggle but the server cannot download: parked, slot freed for a new session, leases kept; a workstation hand-over (COMPLETE) is imported later
c2 = sc.Camp.create('K2'); j2 = [c2.enqueue(dict(matrix=M, tiles=1, cpi=22, wpi=16, seed=50 + i, stub_seconds=0)) for i in range(6)]
k2 = kctl.Ctl('K2'); cm.wjson(os.path.join(k2.d, 'config.json'), dict(cfg, enabled=True)); F['push_ok'] = True; k2.tick(); first = k2.state()['g']
F['status']['tester/' + first['slug']] = 'complete'; real_run = kctl.kc.run
def failing_output(args, timeout=600, cwd=None):
    return (1, 'NameResolutionError') if args[:2] == ['kernels', 'output'] else fake(args, timeout, cwd)
kctl.kc.run = failing_output; k2.tick(); st2 = k2.state()
check('download failed: session parked, slot pushed a new session', first['slug'] in st2['_awaiting'] and st2['g']['slug'] != first['slug'] and st2['g']['status'] == 'running', (list(st2['_awaiting']), st2['g'].get('slug')))
check('parked runs stay leased (not rerun by others)', all(c2.lease_of(r) for r in first['ids']))
od = os.path.join(k2.d, 'downloads', 'g_%d' % int(first['t_push'])); os.makedirs(od, exist_ok=True)
w2 = os.path.join(TMP, 'kaggle2'); os.makedirs(os.path.join(w2, 'input', 'code')); os.makedirs(os.path.join(w2, 'working'))
for f in os.listdir(c2.code_dir()): shutil.copy2(os.path.join(c2.code_dir(), f), os.path.join(w2, 'input', 'code', f))
src2 = open(os.path.join(k2.d, 'kernels', first['slug'], 'worker.py')).read().replace("'/kaggle/input/**/shell_run.py'", repr(os.path.join(w2, 'input', '**', 'shell_run.py'))).replace("W = '/kaggle/working'", 'W = %r' % os.path.join(w2, 'working'))
open(os.path.join(w2, 'worker.py'), 'w').write(src2); subprocess.run([sys.executable, os.path.join(w2, 'worker.py')], capture_output=True, env=dict(os.environ, JAX_PLATFORMS='cpu'))
shutil.copytree(os.path.join(w2, 'working'), od, dirs_exist_ok=True)
import tarfile; tarfile.open(os.path.join(od, 'out.tar')).extractall(od); open(os.path.join(od, 'COMPLETE'), 'w').close()
kctl.kc.run = failing_output; k2.tick(); kctl.kc.run = fake
check('the workstation hand-over is imported without Kaggle', not k2.state()['_awaiting'] and all(c2.index_last().get(r, {}).get('status') == 'done' for r in first['ids']), {r: c2.index_last().get(r, {}).get('status') for r in first['ids']})
print('ALL OK' if not bad else 'FAILED: %s' % bad); shutil.rmtree(TMP, ignore_errors=True); sys.exit(1 if bad else 0)
