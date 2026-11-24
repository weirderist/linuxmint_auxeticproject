#!/usr/bin/env python3
"""shell_analysis_job.py - the scheduled offline analysis of a pure-shell campaign: at every multiple of EVERY = 100 finished runs (100, 200, 300, ...) it runs, one after the other in ONE process,
  1. the proposer evaluation (scripts/shell_learning/shell_proposer_eval.py, evaluate_rows), then
  2. the open-ended rule search (scripts/shell_learning/shell_rules_sr.py, run_rows),
on exactly the first m finished runs (m = the multiple), so a result belongs to one run count and is reproducible. Both tools are imported unchanged.

Counting ("finished runs"): the runs the tools themselves use (shell_features.dataset without control runs): last index status 'done', present in the queue, not a control run. Their order is the order
in which they first reached 'done' in index.jsonl (finishing order), so "the first 100 finished runs" never changes as the campaign goes on.

Where it writes: ONLY <campaign>/analysis/scheduled/ (job.lock, state.json, and per run count <m:06d>/status.json, proposer_eval.json, rules_sr.json) and its own log (stdout; the crontab line sends it to
~/shell_logs/analysis_shell1.log). It never writes the run folders, queue.jsonl, index.jsonl or the frozen code: it only reads them.
Safety:
  - priority: os.nice(19) and, where the ionice program exists, ionice class 3 (idle) for this process;
  - one copy only: analysis/scheduled/job.lock holds pid, boot id and process start time; a lock whose holder is gone (dead process or a previous boot) is taken over, a live one makes this copy wait;
  - idempotent: a tool whose result file for that run count exists is never run again; a result is never overwritten;
  - atomic: every file is written to a temporary name, flushed and fsynced, then renamed;
  - interrupted attempts: status.json marks a tool 'running' before it starts; if a later copy (holding the lock) finds 'running', that attempt is recorded as 'interrupted' (power cut, kill, crash)
    and the tool is run again, at most MAX_ATTEMPTS times in all; a tool that raises an error is recorded as 'failed' with the error text and is not repeated for that run count.
Usage: python3 shell_analysis_job.py <campaign> [--every 100] [--poll-s 600] [--once]"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_S = os.path.dirname(HERE)
if os.path.basename(_S) == 'scripts': sys.path.extend(os.path.join(_S, _d) for _d in sorted(os.listdir(_S)) if _d not in ('legacy', '__pycache__') and os.path.isdir(os.path.join(_S, _d)))          # the other folders of scripts/ (not in a flat frozen copy)

EVERY = 100                 # K's request (2026-11-21): analyse at every 100 finished runs; also the tools' own minimum (shell_rules.MIN_RUNS = 100)
MAX_ATTEMPTS = 3            # UNSOURCED design choice: interrupted attempts allowed per tool and run count before it is marked failed
TOOLS = ('proposer_eval', 'rules_sr')
TOOL_TEXT = dict(proposer_eval='proposer evaluation against random picking (scripts/shell_learning/shell_proposer_eval.py)', rules_sr='open-ended rule search (scripts/shell_learning/shell_rules_sr.py)')


def now(): return time.strftime('%Y-%m-%dT%H:%M:%S')


def wjson_atomic(path, obj):
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def rjson(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError): return default


# ------------------------------------------------------------------ process identity and liveness (Linux /proc on the server, Win32 API for the tests on the workstation)
def me():
    import campaign as cm
    pi = cm.proc_info(os.getpid()) if os.name == 'posix' else None
    return dict(pid=os.getpid(), boot_id=cm.boot_id(), start_ticks=pi[0] if pi else None, t=now())


def holder_alive(rec):
    if not rec or not rec.get('pid'): return False
    if os.name == 'posix':
        import campaign as cm
        return cm.alive(rec)
    import ctypes                                                     # Windows: the process exists and has not exited (STILL_ACTIVE = 259)
    k = ctypes.windll.kernel32; h = k.OpenProcess(0x1000, False, int(rec['pid']))
    if not h: return False
    code = ctypes.c_ulong(); ok = k.GetExitCodeProcess(h, ctypes.byref(code)); k.CloseHandle(h)
    return bool(ok) and code.value == 259


# ------------------------------------------------------------------ paths and counting
def adir(camp): return os.path.join(camp.d, 'analysis', 'scheduled')
def mdir(camp, m): return os.path.join(adir(camp), '%06d' % int(m))


def finished_order(camp):
    """Ids of the finished non-control runs, in the order they first reached 'done' (read only)."""
    import shell_campaign as sc
    q = {x['id']: x for x in camp.queue_items()}; last = camp.index_last(); seen, order = set(), []
    for x in sc.read_lines(camp.index):
        rid = x.get('id')
        if x.get('status') == 'done' and rid and rid not in seen: seen.add(rid); order.append(rid)
    return [r for r in order if (last.get(r) or {}).get('status') == 'done' and r in q and not q[r].get('control')]


def due(n, every=EVERY): return list(range(every, (n // every) * every + 1, every))
def next_milestone(n, every=EVERY): return (n // every + 1) * every


# ------------------------------------------------------------------ lock
class Lock:
    def __init__(s, camp): s.path = os.path.join(adir(camp), 'job.lock'); s.held = False
    def acquire(s):
        os.makedirs(os.path.dirname(s.path), exist_ok=True); rec = me()
        try:
            fd = os.open(s.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as f: json.dump(rec, f); f.flush(); os.fsync(f.fileno())
        except FileExistsError:
            old = rjson(s.path)
            if holder_alive(old): return False                       # another live copy works: this one waits
            wjson_atomic(s.path, rec)                                # stale (dead process or a previous boot): take over
            if (rjson(s.path) or {}).get('pid') != rec['pid']: return False
        s.held = True; return True
    def release(s):
        if s.held and (rjson(s.path) or {}).get('pid') == os.getpid():
            try: os.remove(s.path)                                    # the job's own lock file only
            except OSError: pass
        s.held = False


# ------------------------------------------------------------------ the two tools (imported unchanged; numpy is needed only here)
def run_proposer_eval(rows, params=None):
    import shell_proposer_eval as pe
    return pe.evaluate_rows(rows, **(params or {}))


def run_rules_sr(rows, params=None):
    import shell_rules_sr as sr
    return sr.run_rows(rows, P=params or None)


RUNNERS = dict(proposer_eval=run_proposer_eval, rules_sr=run_rules_sr)


def process_milestone(camp, m, order, rows_by_id, params=None, log=print, runners=None):
    """Run what is still missing for run count m. Returns the status dict."""
    runners = runners or RUNNERS; params = params or {}; d = mdir(camp, m); os.makedirs(d, exist_ok=True); sp = os.path.join(d, 'status.json'); ids = order[:m]
    st = rjson(sp) or dict(campaign=camp.name, milestone=m, n_used=m, run_ids_sha1=hashlib.sha1(','.join(ids).encode()).hexdigest(), created=now(),
                           tools={t: dict(status='queued', attempts=[], result=t + '.json', what=TOOL_TEXT[t]) for t in TOOLS})
    wjson_atomic(sp, st)
    for t in TOOLS:
        tt = st['tools'][t]; out = os.path.join(d, t + '.json')
        if os.path.isfile(out):                                       # idempotent: a result is never recomputed or overwritten
            if tt['status'] != 'done':                                # cut between writing the result and the status: the result is complete (atomic rename), so it counts
                tt['status'] = 'done'
                if tt['attempts'] and tt['attempts'][-1].get('status') == 'running': tt['attempts'][-1].update(status='done', finished=now(), note='status completed after a restart')
                wjson_atomic(sp, st)
            continue
        if tt['status'] == 'failed': continue                         # an error is not repeated for this run count
        if tt['status'] == 'running' and tt['attempts']:              # we hold the lock, so the copy that started it is gone
            tt['attempts'][-1].update(status='interrupted', finished=now(), reason='the process ended before the result was written (power cut, kill or crash); detected by a later start')
            tt['status'] = 'interrupted'; wjson_atomic(sp, st); log('%s %s at %d runs: previous attempt recorded as interrupted' % (now(), t, m))
        if sum(1 for a in tt['attempts'] if a.get('status') == 'interrupted') >= MAX_ATTEMPTS:
            tt['status'] = 'failed'; tt['reason'] = 'interrupted %d times; not started again (MAX_ATTEMPTS)' % MAX_ATTEMPTS; wjson_atomic(sp, st); continue
        att = dict(me(), status='running', started=now()); tt['attempts'].append(att); tt['status'] = 'running'; wjson_atomic(sp, st); t0 = time.time()
        log('%s %s at %d runs: started' % (now(), t, m))
        try:
            rows = [rows_by_id[i] for i in ids]; res = runners[t](rows, params.get(t))
            res = dict(res, campaign=camp.name, generated=now(), scheduled=dict(milestone=m, n_used=len(rows), run_ids=ids, seconds=round(time.time() - t0, 1)))
            if not os.path.isfile(out): wjson_atomic(out, res)
            tt['status'] = 'done'; att.update(status='done', finished=now(), seconds=round(time.time() - t0, 1)); tt.pop('reason', None)
            log('%s %s at %d runs: done in %.0f s' % (now(), t, m, time.time() - t0))
        except Exception as e:
            tt['status'] = 'failed'; tt['reason'] = '%s: %s' % (type(e).__name__, e); att.update(status='failed', finished=now(), reason=tt['reason'])
            log('%s %s at %d runs: FAILED %s' % (now(), t, m, tt['reason']))
        wjson_atomic(sp, st)
    st['finished'] = now() if all(st['tools'][t]['status'] in ('done', 'failed') for t in TOOLS) else None; wjson_atomic(sp, st)
    return st


def run_once(camp, every=EVERY, params=None, log=print, runners=None, loop_rec=None):
    """One check: count, and run every due run count that is not complete. Returns a short dict (for the log and the tests)."""
    lk = Lock(camp)
    if not lk.acquire(): log('%s another copy holds %s: nothing done' % (now(), lk.path)); return dict(status='locked')
    try:
        order = finished_order(camp); n = len(order); todo = due(n, every); out = dict(status='ok', finished=n, due=todo, processed=[])
        if todo:
            import shell_features as sf
            rows_by_id = {r['run']: r for r in sf.dataset(camp) if not r['control']}
            for m in todo:
                st = rjson(os.path.join(mdir(camp, m), 'status.json'))
                if st and all(st['tools'][t]['status'] in ('done', 'failed') for t in TOOLS) and all(st['tools'][t]['status'] == 'failed' or os.path.isfile(os.path.join(mdir(camp, m), t + '.json')) for t in TOOLS): continue
                process_milestone(camp, m, order, rows_by_id, params, log, runners); out['processed'].append(m)
        wjson_atomic(os.path.join(adir(camp), 'state.json'), dict(campaign=camp.name, every=every, finished=n, next_milestone=next_milestone(n, every), last_check=now(), loop=loop_rec or me()))
        return out
    finally: lk.release()


def lower_priority():
    try: os.nice(19)
    except (AttributeError, OSError): pass
    if shutil.which('ionice'):
        try: subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], capture_output=True, timeout=20)
        except Exception: pass


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0]); a.add_argument('campaign'); a.add_argument('--every', type=int, default=EVERY); a.add_argument('--poll-s', type=float, default=600.0)
    a.add_argument('--once', action='store_true'); g = a.parse_args()
    import shell_campaign as sc
    camp = sc.Camp(g.campaign)
    if not camp.exists(): sys.exit('no such shell campaign: %s' % g.campaign)
    lower_priority(); rec = me(); print('%s analysis job for %s: every %d finished runs, check every %.0f s (pid %d)' % (now(), camp.name, g.every, g.poll_s, os.getpid()), flush=True)
    while True:
        try: r = run_once(camp, g.every, log=lambda s: print(s, flush=True), loop_rec=rec); print('%s check: %s' % (now(), r), flush=True)
        except Exception as e: print('%s check failed: %s: %s' % (now(), type(e).__name__, e), flush=True)
        if g.once: break
        time.sleep(g.poll_s)


if __name__ == '__main__':
    main()
