#!/usr/bin/env python3
"""ws_agent.py - WORKSTATION side of the workstation worker (2026-11-27). Standard library only; runs on Windows or Linux.

    python ws_agent.py --campaign shell2 --python <python with jax, numpy, scipy> [--work C:\\auxetic_ws] [--poll-s 60]

It works only while K allows it on the /shell page (Workstation box: how many runs, or which runs; Cancel). Every poll (default 60 s) it asks the server (ssh server ... ws_remote.py status):
  - allowed and idle      -> claim ONE run (lease 30 min), fetch the campaign's frozen code once (every file checked against its sha256 in MANIFEST.json), run shell_run.py from that copy;
  - running               -> renew the lease; if K cancelled, stop the run (process tree) and release the lease, so the run goes back to the server and Kaggle;
  - finished              -> upload the run folder to the server inbox (<id>.<stamp>.part, renamed when complete), then ws_remote import (settings and frozen-code provenance are checked there).
POWER CUT on the workstation (it can lose power at any time): the lease is not renewed and expires after 30 min, so the run is queued again for the others; on the next start the agent finds its
state file, sees the run process is gone, moves the unfinished folder to quarantine/ (never deleted) and releases the lease if it still holds it. A finished but not yet imported run is uploaded again.
NETWORK OUTAGE: a local run continues; renewals fail (the lease may expire: then the import is refused if someone else finished the run first, and the result stays here in quarantine/);
uploads are retried every poll. Local state: <work>/state.json, written atomically."""
import argparse, hashlib, json, os, shutil, socket, subprocess, sys, time

TTL = 1800          # lease seconds (UNSOURCED operating choice: 30 polls of 60 s; a power cut frees the run within 30 min)

def now(): return time.strftime('%Y-%m-%dT%H:%M:%S')
def say(*a): print(now(), *a, flush=True)
def wjson(p, o):
    t = p + '.tmp'
    with open(t, 'w') as f: json.dump(o, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)
def rjson(p, d=None):
    try: return json.load(open(p))
    except (OSError, ValueError): return d
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


class SSH:
    """Transport to the server over the user's ssh config (host alias 'server', key login)."""
    def __init__(s, host='server', root='~/auxetic_project'):
        s.host, s.root = host, root; s.opt = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15']
    def remote(s, args, timeout=300):
        cmd = 'cd %s && venv/bin/python scripts/workstation/ws_remote.py %s' % (s.root, ' '.join("'%s'" % str(a).replace("'", '') for a in args))
        try: p = subprocess.run(['ssh'] + s.opt + [s.host, cmd], capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as e: return None, str(e)
        try: return json.loads(p.stdout.strip().splitlines()[-1]), ''
        except (ValueError, IndexError): return None, (p.stderr or p.stdout)[-300:]
    def sh(s, cmd, timeout=300):
        try: p = subprocess.run(['ssh'] + s.opt + [s.host, cmd], capture_output=True, text=True, timeout=timeout); return p.returncode, p.stdout + p.stderr
        except (subprocess.TimeoutExpired, OSError) as e: return 1, str(e)
    def get(s, remote_path, local_path, timeout=600):
        try: return subprocess.run(['scp', '-q'] + s.opt + ['%s:%s' % (s.host, remote_path), local_path], capture_output=True, text=True, timeout=timeout).returncode == 0
        except (subprocess.TimeoutExpired, OSError): return False
    def exists(s, remote_path):
        rc, o = s.sh("test -d '%s' && echo yes || echo no" % remote_path); return None if rc != 0 and 'yes' not in o and 'no' not in o else ('yes' in o)
    def rename(s, a, b):
        rc, o = s.sh("mv '%s' '%s'" % (a, b)); return rc == 0
    def put_dir(s, local_dir, remote_dir, timeout=1800):
        try: return subprocess.run(['scp', '-q', '-r'] + s.opt + [local_dir, '%s:%s' % (s.host, remote_dir)], capture_output=True, text=True, timeout=timeout).returncode == 0
        except (subprocess.TimeoutExpired, OSError): return False


class Local:
    """The hub (queue, leases, results) is THIS machine (2026-11-27: the workstation may replace the server): ws_remote.py runs here, files are copied locally. Selected with --host local."""
    def __init__(s, root):
        s.root = os.path.abspath(os.path.expanduser(root)); s.py = os.path.join(s.root, 'venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(s.root, 'venv', 'bin', 'python')
        if not os.path.isfile(s.py): s.py = sys.executable
    def remote(s, args, timeout=300):
        try: p = subprocess.run([s.py, os.path.join(s.root, 'scripts', 'workstation', 'ws_remote.py')] + [str(a) for a in args], capture_output=True, text=True, timeout=timeout, cwd=s.root)
        except (subprocess.TimeoutExpired, OSError) as e: return None, str(e)
        try: return json.loads(p.stdout.strip().splitlines()[-1]), ''
        except (ValueError, IndexError): return None, (p.stderr or p.stdout)[-300:]
    def exists(s, path): return os.path.isdir(path)
    def rename(s, a, b):
        try: os.rename(a, b); return True
        except OSError: return False
    def get(s, rp, lp, timeout=600):
        try: shutil.copy2(rp, lp); return True
        except OSError: return False
    def put_dir(s, ld, rd_, timeout=1800):
        try: shutil.copytree(ld, rd_); return True
        except OSError: return False


def proc_alive(pid, ctime=None):
    """Is process pid alive (and, where known, the same process: creation time)? No signal is ever sent (on Windows os.kill would TERMINATE the process)."""
    if not pid: return False
    if os.name == 'nt':
        import ctypes, ctypes.wintypes as wt
        k = ctypes.windll.kernel32; h = k.OpenProcess(0x1000 | 0x100000, False, int(pid))          # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
        if not h: return False
        try:
            code = wt.DWORD(); k.GetExitCodeProcess(h, ctypes.byref(code))
            if code.value != 259: return False          # STILL_ACTIVE
            if ctime is None: return True
            c, e, kk, u = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME(); k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kk), ctypes.byref(u))
            return ((c.dwHighDateTime << 32) | c.dwLowDateTime) == ctime
        finally: k.CloseHandle(h)
    try:
        f = open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()
        return f[0] != 'Z' and (ctime is None or f[19] == str(ctime))          # a zombie (ended, not yet reaped) is not alive (2026-11-28: the power-cut test failed on Linux)
    except (OSError, IndexError): return False

def proc_ctime(pid):
    if os.name == 'nt':
        import ctypes, ctypes.wintypes as wt
        k = ctypes.windll.kernel32; h = k.OpenProcess(0x1000, False, int(pid))
        if not h: return None
        try:
            c, e, kk, u = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME(); k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kk), ctypes.byref(u)); return (c.dwHighDateTime << 32) | c.dwLowDateTime
        finally: k.CloseHandle(h)
    try: return int(open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()[19])
    except OSError: return None

def kill_tree(pid):
    if os.name == 'nt': subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
    else:
        import signal
        try: os.killpg(pid, signal.SIGTERM)
        except OSError: pass


class Agent:
    def __init__(s, g, transport=None):
        s.g = g; s.t = transport or (Local(g.remote_root if g.remote_root != '~/auxetic_project' else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) if g.host == 'local' else SSH(g.host, g.remote_root)); s.w = os.path.abspath(g.work); s.worker = g.worker or 'workstation:' + socket.gethostname().lower()[:40]
        for d in ('runs', 'quarantine', 'done', 'code'): os.makedirs(os.path.join(s.w, d), exist_ok=True)
        s.sp = os.path.join(s.w, 'state.json')
    def state(s): return rjson(s.sp, {}) or {}
    def save(s, st): wjson(s.sp, st)
    def call(s, *args): return s.t.remote([args[0], s.g.campaign, '--worker', s.worker] + list(args[1:]))

    def quarantine(s, st, why):
        src = st.get('out')
        if src and os.path.isdir(src):
            dst = os.path.join(s.w, 'quarantine', '%s.%s.%s' % (st.get('rid'), why, time.strftime('%Y%m%d-%H%M%S'))); shutil.move(src, dst); say('moved', st.get('rid'), 'to', dst)
        s.save(dict(last=dict(st, ended=why, t=now())))

    def code(s, man, code_dir):
        """Local copy of the frozen code, checked file by file against the manifest (fetched once per manifest)."""
        h = hashlib.sha1(json.dumps(man['files'], sort_keys=True).encode()).hexdigest()[:8]; d = os.path.join(s.w, 'code', h)
        if os.path.isdir(d) and all(os.path.isfile(os.path.join(d, f)) and sha256(os.path.join(d, f)) == v for f, v in man['files'].items()): return d
        tmp = d + '.tmp'; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)          # tmp is this agent's own partial download
        for f in list(man['files']) + ['MANIFEST.json']:
            if not s.t.get(code_dir + '/' + f, os.path.join(tmp, f)): say('code download failed:', f); return None
        bad = [f for f, v in man['files'].items() if sha256(os.path.join(tmp, f)) != v]
        if bad: say('code check FAILED (sha256):', bad); return None
        if os.path.isdir(d): shutil.move(d, os.path.join(s.w, 'quarantine', 'code_%s.%s' % (h, time.strftime('%Y%m%d-%H%M%S'))))
        os.replace(tmp, d); say('frozen code fetched and checked:', d); return d

    def start(s, item, codedir):
        rid = item['id']; out = os.path.join(s.w, 'runs', '%s.%s' % (rid, time.strftime('%Y%m%d-%H%M%S'))); os.makedirs(out)
        wjson(os.path.join(out, 'settings.json'), item['settings']); log = open(os.path.join(out, 'shell.log'), 'w')
        kw = dict(creationflags=0x00000200 | 0x08000000) if os.name == 'nt' else dict(start_new_session=True)          # own process group, no window (Windows); own session (POSIX)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **({'SHELL_RUN_STUB': '1'} if s.g.stub else {}))
        p = subprocess.Popen([s.g.python, '-u', os.path.join(codedir, 'shell_run.py'), '--settings', os.path.join(out, 'settings.json'), '--out', out], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, **kw)
        s.proc = p; st = dict(rid=rid, out=out, pid=p.pid, ctime=proc_ctime(p.pid), t_start=time.time(), phase='running'); s.save(st); say('run', rid, 'started (pid %d)' % p.pid); return st

    def upload(s, st, inbox):
        rid = st['rid']; stamp = os.path.basename(st['out']).split('.', 1)[1]; final = '%s/%s.%s' % (inbox, rid, stamp)
        part = '%s.part%s' % (final, time.strftime('%H%M%S'))          # a new name per attempt: scp -r into an existing folder would nest the copy
        ex = s.t.exists(final)
        if ex is None: say('hub not reachable; retried next poll'); return False
        if not ex:          # a partial upload is never removed: each attempt copies into a new .part name, and the rename below is the commit
            if not s.t.put_dir(st['out'], part): say('upload failed; retried next poll'); return False
            if not s.t.rename(part, final): say('rename on the hub failed; retried next poll'); return False
        r, err = s.call('import', '--id', rid, '--src', final, '--wall', str(st.get('wall_s') or ''))
        if r is None: say('import call failed (network?); retried next poll', err); return False
        if r.get('ok'): say('run', rid, 'imported:', r.get('result')); shutil.move(st['out'], os.path.join(s.w, 'done', os.path.basename(st['out']))); s.save(dict(last=dict(st, ended='imported', t=now()))); return True
        say('import REFUSED for', rid, r.get('error')); s.quarantine(st, 'refused'); return True

    def tick(s):
        st = s.state(); cur = st.get('rid') if st.get('phase') else None
        r, err = s.call('status', '--current', cur or '')
        if st.get('phase') == 'running':
            p = getattr(s, 'proc', None); alive = (p.poll() is None) if (p and p.pid == st['pid']) else proc_alive(st['pid'], st.get('ctime'))
            if not alive:
                ok = os.path.isfile(os.path.join(st['out'], 'metrics.json')) and (p is None or p.pid != st['pid'] or p.returncode == 0)
                if ok: st.update(phase='finished', wall_s=round(time.time() - st['t_start'], 1)); s.save(st); say('run', st['rid'], 'finished in %.0f s' % st['wall_s'])
                else:
                    say('run', st['rid'], 'ended without a result (power cut, crash or failure): released, folder kept in quarantine')
                    if r is not None: s.call('release', '--ids', st['rid'], '--why', 'workstation run ended without a result')
                    s.quarantine(st, 'interrupted'); return
            elif r is not None:
                if not (r.get('config') or {}).get('enabled'):
                    say('cancelled on the page: stopping', st['rid']); kill_tree(st['pid']); time.sleep(2); s.call('release', '--ids', st['rid'], '--why', 'cancelled by K'); s.quarantine(st, 'cancelled'); return
                s.call('renew', '--ids', st['rid'], '--ttl', str(TTL)); return
            else: return          # offline: the run continues
        if st.get('phase') == 'finished':
            if r is not None: s.upload(st, r['inbox'])
            return
        if r is None: return
        cf = r.get('config') or {}
        if not cf.get('enabled') or r.get('allowance_left', 0) <= 0 or r.get('control') == 'stop': return
        codedir = s.code(r['manifest'], r['code_dir'])
        if not codedir: return
        c, err = s.call('claim', '--ttl', str(TTL))
        if c and c.get('items'): s.start(c['items'][0], codedir)

def main(argv=None):
    a = argparse.ArgumentParser(); a.add_argument('--campaign', required=True); a.add_argument('--python', default=sys.executable, help='interpreter with jax, numpy, scipy')
    a.add_argument('--work', default=os.environ.get('AUXETIC_WS_HOME') or os.path.join(os.path.expanduser('~'), 'auxetic_ws'))          # AUXETIC_WS_HOME: D:\\auxetic_ws (2026-11-28); a.add_argument('--poll-s', type=float, default=60); a.add_argument('--host', default='server', help="ssh host of the hub, or 'local' when the hub is this machine")
    a.add_argument('--remote-root', default='~/auxetic_project'); a.add_argument('--worker', default=''); a.add_argument('--once', action='store_true'); a.add_argument('--stub', action='store_true', help=argparse.SUPPRESS)
    g = a.parse_args(argv); ag = Agent(g)
    lk = open(os.path.join(ag.w, 'agent.lock'), 'a+')          # one agent per work folder: a second start (logon task while one runs) exits at once; the OS releases the lock when the process dies
    try:
        if os.name == 'nt':
            import msvcrt; lk.seek(0); msvcrt.locking(lk.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl; fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: say('another agent already runs in %s: exiting' % ag.w); return 0
    say('workstation agent for %s as %s, work folder %s' % (g.campaign, ag.worker, ag.w))
    while True:
        try: ag.tick()
        except Exception as e: say('tick error (retried next poll):', repr(e))
        if g.once: break
        time.sleep(g.poll_s)

if __name__ == '__main__':
    main()
