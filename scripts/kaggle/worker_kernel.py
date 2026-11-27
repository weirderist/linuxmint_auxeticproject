"""worker_kernel.py - template of the Kaggle worker (2026-11-27). The controller (kaggle_controller.py) fills BATCH and pushes it as a private kernel; it runs ON KAGGLE.
Each run: the campaign's FROZEN shell_run.py (private dataset <user>/auxetic-<campaign>-code, mounted under /kaggle/input/), output /kaggle/working/out/<id>/. manifest.json is rewritten (atomic) after
every run: the controller imports only runs listed there as done. The worker stops starting runs when the next one would end after BUDGET_S (it must finish before Kaggle's session limit, or Kaggle may
keep no output). With 2 GPUs two runs go in parallel (CUDA_VISIBLE_DEVICES 0 and 1); on CPU one at a time."""
import glob, json, os, subprocess, sys, threading, time
BATCH = __BATCH__          # list of {id, settings, est_s}
BUDGET_S = __BUDGET_S__
CAMPAIGN = __CAMPAIGN__
t0 = time.time(); W = '/kaggle/working'; OUT = os.path.join(W, 'out'); os.makedirs(OUT, exist_ok=True)
code = os.path.dirname(sorted(glob.glob('/kaggle/input/**/shell_run.py', recursive=True))[0])
man = dict(campaign=CAMPAIGN, t_start=time.strftime('%Y-%m-%dT%H:%M:%S'), code_dir=code, runs={}, budget_s=BUDGET_S)
try:
    import jax; man['jax'] = jax.__version__; man['devices'] = [str(d) for d in jax.devices()]
except Exception as e: man['jax_error'] = repr(e)
gpus = [d for d in man.get('devices', []) if d.startswith('cuda')]; lanes = [str(i) for i in range(len(gpus))] or [None]
lock = threading.Lock(); queue = list(BATCH)

def save():
    with lock:
        man['t_saved'] = time.strftime('%Y-%m-%dT%H:%M:%S'); man['elapsed_s'] = round(time.time() - t0, 1)
        with open(os.path.join(W, 'manifest.json.tmp'), 'w') as f: json.dump(man, f, indent=1); f.flush(); os.fsync(f.fileno())
        os.replace(os.path.join(W, 'manifest.json.tmp'), os.path.join(W, 'manifest.json'))

def lane(dev):
    while True:
        with lock:
            if not queue: return
            it = queue[0]
            if time.time() - t0 + 1.3 * it.get('est_s', 600) > BUDGET_S: return          # 1.3: margin on the estimate (UNSOURCED)
            queue.pop(0)
        rid = it['id']; d = os.path.join(OUT, rid); os.makedirs(d, exist_ok=True)
        json.dump(it['settings'], open(os.path.join(d, 'settings.json'), 'w'))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1');
        if dev is not None: env['CUDA_VISIBLE_DEVICES'] = dev
        t = time.time(); log = open(os.path.join(d, 'shell.log'), 'w')
        try:
            p = subprocess.run([sys.executable, '-u', os.path.join(code, 'shell_run.py'), '--settings', os.path.join(d, 'settings.json'), '--out', d], stdout=log, stderr=subprocess.STDOUT, env=env,
                               timeout=max(600, 4 * it.get('est_s', 600)))
            rc = p.returncode
        except subprocess.TimeoutExpired: rc = 'timeout'
        log.close(); ok = rc == 0 and os.path.isfile(os.path.join(d, 'metrics.json'))
        with lock: man['runs'][rid] = dict(status='done' if ok else 'failed', rc=rc, wall_s=round(time.time() - t, 1), device=dev or 'cpu')
        save()

save(); th = [threading.Thread(target=lane, args=(dv,)) for dv in lanes]; [x.start() for x in th]; [x.join() for x in th]
with lock: man['not_started'] = [it['id'] for it in queue]; man['finished'] = True
save(); print(json.dumps({k: v for k, v in man.items() if k != 'runs'}, indent=1)); print('runs:', {k: v['status'] for k, v in man['runs'].items()})
