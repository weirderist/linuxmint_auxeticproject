#!/usr/bin/env python3
"""fetch_on_workstation.py - download finished Kaggle sessions on the WORKSTATION (good internet) and hand them to the server's controller (2026-11-27).

    python fetch_on_workstation.py shell2            (Kaggle token in %USERPROFILE%\\.auxetic_kaggle.env, KAGGLE_* lines; never printed)

Why: the server's internet is slow and often down (16 kB/s and DNS failures on 2026-11-27) while Tailscale ssh to it mostly works. For every slot that the server's state.json shows as running and
Kaggle reports complete: download the output here (resumable: files already here are skipped), unpack out.tar if the worker packed one, copy the folder to the server into the slot's download
folder (downloads/<slot>_<push time>/, the same name the controller uses), and mark it COMPLETE. The controller then imports it without contacting Kaggle (kaggle_controller.finish).
Nothing on the server is overwritten: the copy goes to <folder>.ws<time> and is merged with cp -an (existing files kept), then COMPLETE is written."""
import json, os, subprocess, sys, tarfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('AUXETIC_ENV', os.path.join(os.path.expanduser('~'), '.auxetic_kaggle.env'))
if os.name == 'nt': os.environ.setdefault('KAGGLE_BIN', os.path.join(os.path.expanduser('~'), 'auxetic_ws_venv', 'Scripts', 'kaggle.exe'))
import kaggle_cli as kc          # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'campaigns')); import platform_compat as pc          # noqa: E402  ws_home (2026-11-28)

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', 'server']; ROOT = '~/auxetic_project'

def sh(cmd, timeout=300):
    try: p = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout); return p.returncode, p.stdout
    except (subprocess.TimeoutExpired, OSError) as e: return 1, str(e)

def main(camp):
    user = kc.username()
    if not user: sys.exit('no KAGGLE_USERNAME in %s' % os.environ['AUXETIC_ENV'])
    kd = '%s/campaigns/shell_runs/%s/kaggle' % (ROOT, camp); rc, o = sh('cat %s/state.json' % kd)
    if rc != 0: sys.exit('server not reachable: %s' % o[-200:])
    st = json.loads(o); local = os.path.join(pc.ws_home(), 'kaggle_downloads'); os.makedirs(local, exist_ok=True)
    ents = [(n, e) for n, e in st.items() if not n.startswith('_') and isinstance(e, dict) and e.get('status') == 'running'] + [(e.get('slot') or k.split('-')[2], e) for k, e in (st.get('_awaiting') or {}).items()]
    for name, ent in ents:
        rc, o = kc.run(['kernels', 'status', '%s/%s' % (user, ent['slug'])]); print(name, o.strip())
        if 'complete' not in o.lower(): continue
        folder = '%s_%d' % (name, int(ent['t_push'])); ld = os.path.join(local, folder); os.makedirs(ld, exist_ok=True)
        stg = os.path.join(pc.ws_home(), 'kaggle_staging', ent['slug'])          # downloaded earlier while the server was away (2026-11-27): reuse it
        if os.path.isfile(os.path.join(stg, 'manifest.json')) and (json.load(open(os.path.join(stg, 'manifest.json'))).get('finished')):
            import shutil; shutil.copytree(stg, ld, dirs_exist_ok=True); rc = 0; print(name, 'using the staged download', stg)
        else: rc = 1
        for attempt in range(5 if rc else 0):
            rc, o = kc.run(['kernels', 'output', '%s/%s' % (user, ent['slug']), '-p', ld], timeout=3600)
            if rc == 0: break
            print('download attempt %d failed: %s' % (attempt + 1, o.strip()[-200:])); time.sleep(20)
        if rc != 0: print('giving up on', name, 'for now'); continue
        tp = os.path.join(ld, 'out.tar')
        if os.path.isfile(tp) and not os.path.isdir(os.path.join(ld, 'out')):
            with tarfile.open(tp) as tf: tf.extractall(ld, filter='data')
        man = json.load(open(os.path.join(ld, 'manifest.json'))); runs = man.get('runs') or {}
        missing = [r for r, v in runs.items() if v.get('status') == 'done' and not os.path.isfile(os.path.join(ld, 'out', r, 'metrics.json'))]
        if missing: print(name, ': %d done runs are incomplete here, not handed over:' % len(missing), missing[:5]); continue
        remote = '%s/downloads/%s' % (kd, folder); tmp = '%s.ws%s' % (remote, time.strftime('%H%M%S'))
        p = subprocess.run(['scp', '-q', '-r', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', ld, 'server:%s' % tmp], capture_output=True, text=True)
        if p.returncode != 0: print('upload failed:', p.stderr[-200:]); continue
        rc, o = sh("mkdir -p '%s' && cp -an '%s'/. '%s'/ && touch '%s'/COMPLETE && echo ok" % (remote, tmp, remote, remote))
        print(name, 'handed to the server:' if 'ok' in o else 'merge FAILED:', remote, o.strip()[-200:])

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'shell2')
