#!/usr/bin/env python3
"""probe_push.py - on the server: upload the frozen code of a campaign as a PRIVATE Kaggle dataset and push the probe kernel (CPU and GPU variants), then wait and download the outputs.
    python3 scripts/kaggle/probe_push.py shell2
Work folder ~/kaggle_work (outside the repository). Credentials from .env through kaggle_cli (never printed)."""
import json, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kaggle_cli as kc
camp = sys.argv[1]; ROOT = kc.ROOT; W = os.path.expanduser('~/kaggle_work'); user = kc.username()
if not user: sys.exit('no KAGGLE_USERNAME in .env')
ds = 'auxetic-%s-code' % camp; dsd = os.path.join(W, ds); os.makedirs(dsd, exist_ok=True)
src = os.path.join(ROOT, 'campaigns', 'shell_runs', camp, 'code')
for f in os.listdir(src): shutil.copy2(os.path.join(src, f), os.path.join(dsd, f))
json.dump(dict(title=ds, id='%s/%s' % (user, ds), licenses=[dict(name='CC0-1.0')]), open(os.path.join(dsd, 'dataset-metadata.json'), 'w'))
rc, o = kc.run(['datasets', 'status', '%s/%s' % (user, ds)])
rc, o = kc.run(['datasets', 'version', '-p', dsd, '-m', 'frozen code', '-q'] if rc == 0 and 'ready' in o.lower() else ['datasets', 'create', '-p', dsd, '-q'], timeout=900); print('dataset:', rc, o.strip()[-300:], flush=True)
for _ in range(60):
    rc, o = kc.run(['datasets', 'status', '%s/%s' % (user, ds)]); print('dataset status:', o.strip(), flush=True)
    if 'ready' in o.lower(): break
    time.sleep(20)
slugs = []
for gpu in (False, True):
    slug = 'auxetic-probe-%s' % ('gpu' if gpu else 'cpu'); kd = os.path.join(W, slug); os.makedirs(kd, exist_ok=True); shutil.copy2(os.path.join(HERE, 'probe_kernel.py'), os.path.join(kd, 'probe_kernel.py'))
    json.dump(dict(id='%s/%s' % (user, slug), title=slug, code_file='probe_kernel.py', language='python', kernel_type='script', is_private=True, enable_gpu=gpu, enable_internet=True,
                   dataset_sources=['%s/%s' % (user, ds)], competition_sources=[], kernel_sources=[]), open(os.path.join(kd, 'kernel-metadata.json'), 'w'))
    rc, o = kc.run(['kernels', 'push', '-p', kd], timeout=600); print('push', slug, rc, o.strip()[-300:], flush=True); slugs.append(slug)
pend = set(slugs); t0 = time.time()
while pend and time.time() - t0 < 3 * 3600:
    time.sleep(60)
    for slug in sorted(pend):
        rc, o = kc.run(['kernels', 'status', '%s/%s' % (user, slug)]); st = o.strip().lower()
        if any(w in st for w in ('complete', 'error', 'cancel')):
            od = os.path.join(W, slug + '_out'); os.makedirs(od, exist_ok=True); rc2, o2 = kc.run(['kernels', 'output', '%s/%s' % (user, slug), '-p', od], timeout=900)
            print(slug, 'finished:', o.strip(), '| output rc', rc2, flush=True)
            try: print(open(os.path.join(od, 'probe.json')).read(), flush=True)
            except OSError: print(o2[-1500:], flush=True)
            pend.discard(slug)
print('PROBE DONE', flush=True)
