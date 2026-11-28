#!/usr/bin/env python3
"""yarn_kaggle.py - run yarn relaxations (scripts/yarn_model/relax_job.py) on a Kaggle GPU (tasks 56 and 58, 2026-11-28).

    python yarn_kaggle.py push  <result folder> --job NAME "relax_job arguments" [--job NAME2 "..."] [--cpu]
    python yarn_kaggle.py status <result folder>
    python yarn_kaggle.py fetch  <result folder> [--into NAME]       (NAME: which job's output to copy into the folder; default the first job)
    python yarn_kaggle.py run    <result folder> --job ...            (push, wait, fetch: what the Simulate page starts in the background)

What it sends: ONE private dataset per push, auxetic-yarn-<stamp>, with code.tar (the whole scripts/ folder, so relax_job finds its modules the same way as here) and input.tar (the built
fabric of the result folder: fab_axis.csv, fab_nodes.csv, meta.json, and shell.npz / relaxed files when present). The kernel runs every job one after the other, each on a fresh copy of the
input, with the job's own arguments (relax_job --folder is set by the kernel); --solver fire is the GPU solver. Each job's folder is packed into out_<NAME>.tar with its wall time in
yarn_manifest.json. Nothing else is uploaded (no .env: the credentials stay on this machine, read by kaggle_cli.py from AUXETIC_ENV, default %USERPROFILE%\\.auxetic_kaggle.env here).
Records: <result folder>/kaggle.json (slug, dataset, jobs, state) - the Simulate page reads it. Work folder: platform_compat.ws_home()/kaggle_yarn/<slug>/ (outside the repository).
fetch copies ONLY files the job wrote or changed (relaxed_*, state.json, meta.json, checkpoint*, diagnostics.csv, shell.npz, relax.log); the built fab_* files are never overwritten."""
import io, json, os, shutil, sys, tarfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); SCRIPTS = os.path.join(ROOT, 'scripts')
if os.name == 'nt': os.environ.setdefault('AUXETIC_ENV', os.path.join(os.path.expanduser('~'), '.auxetic_kaggle.env'))
import kaggle_cli as kc
sys.path.insert(0, os.path.join(SCRIPTS, 'campaigns')); import platform_compat as pc

INPUT_FILES = ('fab_axis.csv', 'fab_nodes.csv', 'meta.json', 'shell.npz', 'relaxed_axis.csv', 'relaxed_l0.csv', 'checkpoint.npz')
FETCH_PREFIX = ('relaxed_', 'state.json', 'meta.json', 'checkpoint', 'diagnostics.csv', 'shell.npz', 'relax.log', 'fold_')
BUDGET_S = 8 * 3600          # stop starting jobs after this (Kaggle GPU sessions end at 12 h; UNSOURCED margin)

KERNEL = r'''
import json, os, subprocess, sys, tarfile, time, glob, shutil
JOBS = json.loads(__JOBS__); BUDGET_S = __BUDGET__
W = '/kaggle/working'; t0 = time.time(); man = dict(jobs={}, t_start=time.strftime('%Y-%m-%dT%H:%M:%S'))
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/code.tar', recursive=True))[0])
with tarfile.open(os.path.join(src, 'code.tar')) as tf: tf.extractall(os.path.join(W, 'code'))
try:
    import jax; man['jax'] = jax.__version__; man['devices'] = [str(d) for d in jax.devices()]
except Exception as e: man['jax_error'] = repr(e)
def save():
    with open(os.path.join(W, 'yarn_manifest.json.tmp'), 'w') as f: json.dump(man, f, indent=1)
    os.replace(os.path.join(W, 'yarn_manifest.json.tmp'), os.path.join(W, 'yarn_manifest.json'))
save()
for j in JOBS:
    if time.time() - t0 > BUDGET_S: man['jobs'][j['name']] = dict(status='skipped', reason='budget'); save(); continue
    d = os.path.join(W, 'run_' + j['name']); os.makedirs(d, exist_ok=True)
    with tarfile.open(os.path.join(src, 'input.tar')) as tf: tf.extractall(d)
    cmd = [sys.executable, os.path.join(W, 'code', 'scripts', 'yarn_model', 'relax_job.py'), '--folder', d] + j['args']
    t1 = time.time(); log = open(os.path.join(d, 'relax.log'), 'w'); p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT); log.close()
    man['jobs'][j['name']] = dict(status='done' if p.returncode == 0 else 'failed', exit=p.returncode, seconds=round(time.time() - t1, 1), args=j['args']); save()
    with tarfile.open(os.path.join(W, 'out_%s.tar' % j['name']), 'w') as tf: tf.add(d, arcname='run')
    shutil.rmtree(d, ignore_errors=True)
man['t_end'] = time.strftime('%Y-%m-%dT%H:%M:%S'); save()
'''


def _rj(p, d=None):
    try: return json.load(open(p, encoding='utf8'))
    except (OSError, ValueError): return d


def _wj(p, x):
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf8') as f: json.dump(x, f, indent=1)
    os.replace(tmp, p)


def _tar(path, items):
    with tarfile.open(path, 'w') as tf:
        for arc, src in items: tf.add(src, arcname=arc)


def push(folder, jobs, gpu=True, log=print):
    user = kc.username()
    if not user: raise SystemExit('no KAGGLE_USERNAME in %s' % kc.ENV)
    folder = os.path.abspath(folder); stamp = time.strftime('%m%d%H%M%S'); slug = 'auxetic-yarn-%s' % stamp
    wd = os.path.join(pc.ws_home(), 'kaggle_yarn', slug); dsd = os.path.join(wd, 'dataset'); kd = os.path.join(wd, 'kernel'); os.makedirs(dsd); os.makedirs(kd)
    skip = lambda tarinfo: None if ('__pycache__' in tarinfo.name or tarinfo.name.endswith('.pyc') or '/legacy' in tarinfo.name) else tarinfo
    with tarfile.open(os.path.join(dsd, 'code.tar'), 'w') as tf: tf.add(SCRIPTS, arcname='scripts', filter=skip)
    _tar(os.path.join(dsd, 'input.tar'), [(f, os.path.join(folder, f)) for f in INPUT_FILES if os.path.isfile(os.path.join(folder, f))])
    json.dump(dict(title=slug, id='%s/%s' % (user, slug), licenses=[dict(name='CC0-1.0')]), open(os.path.join(dsd, 'dataset-metadata.json'), 'w'))
    rc, o = kc.run(['datasets', 'create', '-p', dsd, '-q'], timeout=1800); log('dataset:', rc, o.strip()[-200:])
    if rc != 0: raise SystemExit('dataset upload failed: ' + o.strip()[-300:])
    for _ in range(90):
        rc, o = kc.run(['datasets', 'status', '%s/%s' % (user, slug)])
        if 'ready' in o.lower(): break
        time.sleep(10)
    open(os.path.join(kd, 'yarn_kernel.py'), 'w').write(KERNEL.replace('__JOBS__', repr(json.dumps(jobs))).replace('__BUDGET__', repr(float(BUDGET_S))))
    ks = slug + '-k'          # the kernel needs a name of its own: the same name as its dataset is refused (409 Conflict, 2026-11-28)
    json.dump(dict(id='%s/%s' % (user, ks), title=ks, code_file='yarn_kernel.py', language='python', kernel_type='script', is_private=True, enable_gpu=bool(gpu), enable_internet=False,
                   dataset_sources=['%s/%s' % (user, slug)], competition_sources=[], kernel_sources=[]), open(os.path.join(kd, 'kernel-metadata.json'), 'w'))
    rc, o = kc.run(['kernels', 'push', '-p', kd], timeout=600); log('push:', rc, o.strip()[-200:])
    rec = dict(slug=ks, dataset=slug, user=user, gpu=bool(gpu), jobs=jobs, workdir=wd, pushed=time.strftime('%Y-%m-%dT%H:%M:%S'), state='pushed' if rc == 0 and 'success' in o.lower() else 'push failed', push_out=o.strip()[-300:])
    _wj(os.path.join(folder, 'kaggle.json'), rec); return rec


def status(folder):
    rec = _rj(os.path.join(folder, 'kaggle.json'))
    if not rec: raise SystemExit('no kaggle.json in this folder')
    rc, o = kc.run(['kernels', 'status', '%s/%s' % (rec['user'], rec['slug'])]); s = o.strip().lower()
    st = 'complete' if 'complete' in s else 'error' if 'error' in s else 'cancelled' if 'cancel' in s else 'running' if 'running' in s else 'queued' if 'queued' in s else ('unknown: ' + o.strip()[-120:])
    if rec.get('state') not in ('fetched',): rec['state'] = st; rec['checked'] = time.strftime('%Y-%m-%dT%H:%M:%S'); _wj(os.path.join(folder, 'kaggle.json'), rec)
    return st, rec


def fetch(folder, into=None, log=print):
    folder = os.path.abspath(folder); rec = _rj(os.path.join(folder, 'kaggle.json')); od = os.path.join(rec['workdir'], 'output'); os.makedirs(od, exist_ok=True)
    rc, o = kc.run(['kernels', 'output', '%s/%s' % (rec['user'], rec['slug']), '-p', od], timeout=3600); log('output:', rc, o.strip()[-200:])
    man = _rj(os.path.join(od, 'yarn_manifest.json'), {}) or {}; name = into or rec['jobs'][0]['name']; tp = os.path.join(od, 'out_%s.tar' % name)
    if not os.path.isfile(tp): raise SystemExit('no output of job %s (manifest: %s)' % (name, json.dumps(man.get('jobs', {}))[:300]))
    ex = os.path.join(od, 'x_' + name); os.makedirs(ex, exist_ok=True)
    with tarfile.open(tp) as tf:
        try: tf.extractall(ex, filter='data')
        except TypeError: tf.extractall(ex)
    run = os.path.join(ex, 'run'); copied = []
    for f in sorted(os.listdir(run)):
        if f.startswith(FETCH_PREFIX) and os.path.isfile(os.path.join(run, f)): shutil.copy2(os.path.join(run, f), os.path.join(folder, f)); copied.append(f)
    rec.update(state='fetched', fetched=time.strftime('%Y-%m-%dT%H:%M:%S'), fetched_job=name, manifest=man, copied=copied); _wj(os.path.join(folder, 'kaggle.json'), rec)
    return rec


def run(folder, jobs, gpu=True, poll_s=60, log=print):
    rec = push(folder, jobs, gpu, log)
    if rec['state'] != 'pushed': raise SystemExit('push failed: ' + rec.get('push_out', ''))
    while True:
        time.sleep(poll_s); st, rec = status(folder); log(time.strftime('%H:%M:%S'), 'kaggle', st)
        if st in ('complete', 'error', 'cancelled'): break
    return fetch(folder, log=log) if st == 'complete' else rec


def _jobs(a):
    out = []; i = 0
    while i < len(a):
        if a[i] == '--job': out.append(dict(name=a[i + 1], args=a[i + 2].split())); i += 3
        else: i += 1
    return out


def main():
    a = sys.argv[1:]
    if len(a) < 2: sys.exit(__doc__)
    cmd, folder = a[0], a[1]
    if cmd in ('push', 'run'):
        jobs = _jobs(a[2:])
        if not jobs: sys.exit('give at least one --job NAME "relax_job arguments"')
        r = (push if cmd == 'push' else run)(folder, jobs, gpu='--cpu' not in a); print(json.dumps({k: r.get(k) for k in ('slug', 'state', 'copied')}))
    elif cmd == 'status': print(status(folder)[0])
    elif cmd == 'fetch': print(json.dumps(fetch(folder, a[a.index('--into') + 1] if '--into' in a else None).get('copied')))
    else: sys.exit(__doc__)


if __name__ == '__main__':
    main()
