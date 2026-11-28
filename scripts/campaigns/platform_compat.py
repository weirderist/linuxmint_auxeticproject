#!/usr/bin/env python3
"""platform_compat.py - the few operating-system services the runners and the web service need, for Linux (the server) AND Windows (the workstation), 2026-11-27.
K's plan: the server may be retired, so everything that ran only on Linux must also run on the workstation. Every Linux branch below is the code the project used before (behaviour
unchanged on the server); the Windows branch is new. Standard library only (ctypes on Windows).

  boot_id()                    Linux: /proc boot id. Windows: the last boot time (CIM), read once per process (it cannot change while the process lives)
  proc_info(pid)               (start_ticks, state) or None. Linux: /proc/<pid>/stat. Windows: (creation time, 'R') from the process handle
  cmdlines()                   [(pid, [args])] of running processes. Windows: one CIM query (about a second)
  proc_table()                 {pid: (ppid, rss_kb)} for descendants_rss_kb
  try_lock(path) / write_lock_info(handle, path, text)   exclusive non-blocking lock held while the handle is open (Linux flock on the file; Windows a byte lock on <path>.lk)
  kill_tree(pid, hard=False)   the process and its children. Linux: SIGTERM / SIGKILL to the process group. Windows: taskkill /T /F (no graceful signal exists for console programs)
  detach_kwargs()              Popen arguments for a process that outlives its parent. Linux: own session, SIGHUP ignored. Windows: new process group, no console window
  child_group_kwargs()         Popen arguments so that kill_tree reaches the child's own children (Linux own session; Windows new process group)
  venv_python(root) / venv_bin(root, name)   venv/bin/python (Linux) or venv\\Scripts\\python.exe (Windows)"""
import json, os, signal, subprocess, sys, time
WIN = os.name == 'nt'


_WIN_BOOT = []
def boot_id():
    if WIN:          # the web service's auto-resume compares boot ids: an empty one would switch it off on Windows
        if not _WIN_BOOT:
            try: _WIN_BOOT.append('win-' + subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', '(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString("yyyyMMddHHmmss")'],
                                                         capture_output=True, text=True, timeout=60, creationflags=0x08000000).stdout.strip())
            except Exception: return ''
        return _WIN_BOOT[0] if _WIN_BOOT[0] != 'win-' else ''
    try: return open('/proc/sys/kernel/random/boot_id').read().strip()
    except OSError: return ''


def _win_ctime(pid):
    import ctypes, ctypes.wintypes as wt
    k = ctypes.windll.kernel32; h = k.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h: return None
    try:
        code = wt.DWORD()
        if not k.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259: return None          # 259 = STILL_ACTIVE
        c, e, kk, u = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
        if not k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kk), ctypes.byref(u)): return None
        return str((c.dwHighDateTime << 32) | c.dwLowDateTime)
    finally: k.CloseHandle(h)


def proc_info(pid):
    """(start_ticks, state) or None if the process does not exist."""
    if WIN:
        try: t = _win_ctime(int(pid))
        except Exception: t = None
        return (t, 'R') if t else None
    try:
        d = open('/proc/%d/stat' % pid).read()
        rest = d[d.rindex(')') + 2:].split()
        return rest[19], rest[0]
    except (OSError, ValueError, IndexError): return None


def _win_processes():
    """[(pid, ppid, rss_kb, commandline)] from one CIM query (PowerShell); [] if it fails."""
    ps = ('Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, WorkingSetSize, CommandLine | ConvertTo-Json -Compress')
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', ps], capture_output=True, text=True, timeout=60, creationflags=0x08000000).stdout
        rows = json.loads(out or '[]'); rows = rows if isinstance(rows, list) else [rows]
        return [(int(r['ProcessId']), int(r.get('ParentProcessId') or 0), int(r.get('WorkingSetSize') or 0) // 1024, r.get('CommandLine') or '') for r in rows]
    except Exception: return []


def _split_cmdline(s):
    import shlex
    try: return [a.strip('"') for a in shlex.split(s, posix=False)]
    except ValueError: return s.split()


def cmdlines():
    if WIN: return [(pid, _split_cmdline(cl)) for pid, _, _, cl in _win_processes() if cl]
    import glob
    out = []
    for p in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            pid = int(p.split('/')[2])
            args = open(p, 'rb').read().split(b'\0')
            pi = proc_info(pid)
            if pi and pi[1] != 'Z': out.append((pid, [a.decode('utf8', 'replace') for a in args if a]))
        except (OSError, ValueError): continue
    return out


def proc_rows():
    """[(pid, ppid, args)] of the live processes in ONE scan (2026-11-28: on Windows every scan is a CIM query of about a second; busy checks need parents and command lines together)."""
    if WIN: return [(pid, ppid, _split_cmdline(cl)) for pid, ppid, _, cl in _win_processes() if cl]
    tb = proc_table(); return [(pid, tb.get(pid, (0, 0))[0], args) for pid, args in cmdlines()]


def proc_table():
    """{pid: (ppid, rss_kb)}"""
    if WIN: return {pid: (ppid, rss) for pid, ppid, rss, _ in _win_processes()}
    import glob
    t = {}
    for p in glob.glob('/proc/[0-9]*/stat'):
        try:
            pid = int(p.split('/')[2]); ppid = int(open(p).read().rsplit(')', 1)[1].split()[1]); rss = 0
            for l in open('/proc/%d/status' % pid):
                if l.startswith('VmRSS:'): rss = int(l.split()[1])
            t[pid] = (ppid, rss)
        except (OSError, ValueError, IndexError): pass
    return t


def try_lock(path):
    """Open and lock exclusively, non-blocking. Returns the open handle (keep it open while the lock is wanted) or None if another process holds it."""
    if WIN:
        import msvcrt
        h = open(path + '.lk', 'a+')
        try: h.seek(0); msvcrt.locking(h.fileno(), msvcrt.LK_NBLCK, 1); return h
        except OSError: h.close(); return None
    import fcntl
    h = open(path, 'a+')
    try: fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB); return h
    except OSError: h.close(); return None


def write_lock_info(handle, path, text):
    """Record who holds the lock in <path> (Linux: in the locked file itself, as before; Windows: <path> beside the locked <path>.lk)."""
    if WIN:
        tmp = path + '.tmp'
        with open(tmp, 'w') as f: f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path); return
    handle.seek(0); handle.truncate(); handle.write(text); handle.flush()


def kill_tree(pid, hard=False):
    """Stop the process and its children. Never raises."""
    if WIN:
        try: subprocess.run(['taskkill', '/PID', str(int(pid)), '/T', '/F'], capture_output=True, timeout=60, creationflags=0x08000000)
        except Exception: pass
        return
    try: os.killpg(int(pid), signal.SIGKILL if hard else signal.SIGTERM)
    except OSError: pass


def detach_kwargs():
    if WIN: return dict(creationflags=0x00000200 | 0x08000000)          # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW: survives its parent, no console window
    return dict(start_new_session=True, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))


def child_group_kwargs():
    if WIN: return dict(creationflags=0x00000200 | 0x08000000)
    return dict(start_new_session=True)


def venv_bin(root, name):
    return os.path.join(root, 'venv', 'Scripts', name + '.exe') if WIN else os.path.join(root, 'venv', 'bin', name)


def venv_python(root): return venv_bin(root, 'python')

def ws_home():
    """The workstation's work folder (agent work, Kaggle downloads and staging). K 2026-11-28: the project lives on D:, so the folder is set by the
    environment variable AUXETIC_WS_HOME (D:\\auxetic_ws on the workstation, set by K with setx); without it the old place ~/auxetic_ws."""
    return os.environ.get('AUXETIC_WS_HOME') or os.path.join(os.path.expanduser('~'), 'auxetic_ws')
