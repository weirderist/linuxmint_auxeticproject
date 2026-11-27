#!/usr/bin/env python3
"""shell_campaign.py - campaigns of PURE SHELL runs (shell_sim.py only: no yarn mesh). Extends campaign.py (commands shell-plan, shell-enqueue, shell-run, shell-status, shell-control).
Standard library plus numpy (metrics). Design: the scratch report "Pure shell only runs" approved by K on 2026-11-21 (root campaigns/shell_runs, delete = heavy files only, limits below).

Folder (SHELL_ROOT/<campaign>/):  spec.json  queue.jsonl (append only)  index.jsonl (append only, last line per id wins)  control.json (run|pause|stop)  runner.json  runner.lock
  runs/S000123/ (complete runs only, created by renaming a finished .tmp folder)  .tmp/S000123.<attempt>/  quarantine/ (interrupted or failed attempts, moved by rename, never deleted by the runner)
  next_id  deletions.log (append only)  results.csv (rebuilt)
Frozen code: shell-plan copies FROZEN_FILES into <campaign>/code/ (write-once, with MANIFEST.json: git commit, dirty flag, sha256 of every file, library versions). The runner and every worker run from that copy,
so a later `git pull` cannot change a running campaign; the hashes are checked before every run (mismatch: the runner stops itself) and written into every run folder (provenance.json), the index and results.csv. New code = new campaign.
Rules: ids are never reused; a run folder is only ever created by an atomic rename and never overwritten; the runner never deletes anything (only the page's delete, which removes the heavy files shell.npz and shell.bin of
chosen runs, inside runs/, with a path guard and a log written first). No campaign-wide time limit: the runner goes on until paused. Per-run timeout and the free-disk guard stop only the run / the runner.
"""
import argparse, csv, glob, hashlib, json, math, os, re, shutil, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import campaign as cm            # wjson, rjson, boot_id, proc_info, alive, busy_reason, say, LOG_DIR

ROOT = os.path.dirname(os.path.dirname(HERE)) if os.path.basename(os.path.dirname(HERE)) == 'scripts' else os.path.dirname(HERE)          # the project folder; in a frozen copy (<campaign>/code/) the old rule stays
FROZEN = os.path.basename(HERE) == 'code' and os.path.isfile(os.path.join(os.path.dirname(HERE), 'spec.json'))      # this file runs from the FROZEN COPY inside a campaign folder (<root>/<campaign>/code/)
CAMPAIGNS = os.environ.get('CAMPAIGN_DIR') or os.path.join(ROOT, 'campaigns')
SHELL_ROOT = os.environ.get('SHELL_RUNS_DIR') or (os.path.dirname(os.path.dirname(HERE)) if FROZEN else os.path.join(CAMPAIGNS, 'shell_runs'))
FROZEN_FILES = ('campaign.py', 'shell_campaign.py', 'shell_run.py', 'shell_metrics.py', 'shell_sim.py', 'shell_policy.py')            # everything the runner and its worker execute from scripts/
NAME_RE = re.compile(r'^[0-9A-Za-z_-]{1,64}$'); ID_RE = re.compile(r'^S[0-9]{6}$')
HEAVY = ('shell.npz', 'shell.bin')
# Limits. Every number is UNSOURCED (K approved them on 2026-11-21): the reference time is ONE measurement (tiles 3, 11,737 vertices, 285 s); linear scaling with vertices is not verified.
DEFAULT_LIMITS = dict(min_free_gb=20.0, max_total_gb=50.0, timeout_floor_s=900.0, timeout_factor=4.0, ref_time_s=285.0, ref_vertices=11737, max_interrupts=3)
DEFAULTS = dict(kappa_scale=1.0, h_scale=1.0, sub=2, steps=15, iter=400, nu=0.4, springs=0.0, seed=0, ne=12.0)      # the 13 Nov success shell (relax_job defaults); ne is not used by the shell
METRIC_COLS = ['plan_x', 'plan_y', 'thickness', 'shrink_x', 'shrink_y', 'fold_kx_per_mm', 'fold_ky_per_mm', 'fold_turn_x_rad', 'fold_turn_y_rad', 'fold_direction',
               'wave_k', 'wave_power_fraction', 'wave_amp_median_mm', 'mid_over_end', 'steps_at_cap', 'iterations_total', 'U_final', 'solver_seconds', 'grad_max_end', 'stability_min']
SETTING_COLS = ['tiles', 'cpi', 'wpi', 'ne', 'kappa_scale', 'h_scale', 'sub', 'steps', 'iter', 'nu', 'springs', 'seed']
TERMINAL = ('done', 'failed', 'timeout', 'needs_approval', 'rejected')


# ------------------------------------------------------------------ settings
def matrix_from_text(text):
    """'1 1 2/2 2 1' (rows separated by /, digits 1 and 2, optional spaces; first row = bottom course) -> list of lists."""
    bad = sorted(set(ch for ch in text if ch not in '12/ \t\r\n'))
    if bad: raise ValueError('matrix text may hold only the digits 1 and 2, spaces and /; found %s' % ''.join(bad))          # was silently dropped (2026-11-26): '1 3/1 3' became [[1],[1]]
    rows = [[int(ch) for ch in r if ch in '12'] for r in text.strip().split('/')]
    return rows

def normalise(s, max_stitches=6000):
    """Validated copy of a settings dict with every default filled in (so equal settings hash equally). Raises ValueError. Checks structure only; the proposer's ranges are a separate, later, K-approved file."""
    s = dict(s); M = s.get('matrix')
    if isinstance(M, str): M = matrix_from_text(M)
    if not M or not isinstance(M, list) or not all(isinstance(r, list) and r for r in M) or len({len(r) for r in M}) != 1 or any(v not in (1, 2) for r in M for v in r): raise ValueError('matrix must be rectangular with digits 1 and 2')
    if any(k not in s for k in ('cpi', 'wpi', 'tiles')): raise ValueError('settings need matrix, tiles, cpi and wpi')
    out = dict(DEFAULTS); out.update({k: v for k, v in s.items() if k != 'matrix'}); out['matrix'] = [[int(v) for v in r] for r in M]
    try:
        for k in ('tiles', 'sub', 'steps', 'iter', 'seed'): out[k] = int(out[k])
        for k in ('cpi', 'wpi', 'kappa_scale', 'h_scale', 'nu', 'springs', 'ne'): out[k] = float(out[k])
    except (TypeError, ValueError): raise ValueError('settings must be numbers')
    if not (1 <= out['tiles'] <= 8 and 1 <= out['sub'] <= 4 and 1 <= out['steps'] <= 60 and 10 <= out['iter'] <= 5000): raise ValueError('tiles 1-8, sub 1-4, steps 1-60, iter 10-5000')
    if not (5 <= out['cpi'] <= 80 and 5 <= out['wpi'] <= 80): raise ValueError('cpi and wpi must be between 5 and 80')
    if not (math.isfinite(out['kappa_scale']) and out['kappa_scale'] != 0 and out['h_scale'] > 0 and 0 <= out['nu'] < 0.5 and out['springs'] >= 0): raise ValueError('kappa_scale != 0, h_scale > 0, 0 <= nu < 0.5, springs >= 0')
    # shell2 keys (2026-11-27): only present when set, so the settings (and hashes) of shell1 runs are unchanged
    if 'bending' in out and out['bending'] not in ('hinge', 'plate'): raise ValueError("bending must be 'hinge' or 'plate'")
    if 'solver' in out and out['solver'] not in ('lbfgs', 'fire', 'fire+newton', 'newton'): raise ValueError("solver must be lbfgs, fire, fire+newton or newton")
    if 'check_stability' in out: out['check_stability'] = bool(out['check_stability'])
    if len(out['matrix']) * len(out['matrix'][0]) * out['tiles'] ** 2 > max_stitches: raise ValueError('more than %d stitches per sheet: refused (run time)' % max_stitches)
    return out

def sha(s): return hashlib.sha1(json.dumps(s, sort_keys=True).encode()).hexdigest()

def n_vertices(s): return (len(s['matrix'][0]) * s['tiles'] * s['sub'] + 1) * (len(s['matrix']) * s['tiles'] * s['sub'] + 1)

def timeout_for(s, lim):
    return max(lim['timeout_floor_s'], lim['timeout_factor'] * lim['ref_time_s'] * n_vertices(s) / lim['ref_vertices'])


# ------------------------------------------------------------------ small file tools
def append_line(path, obj):
    lead = ''
    try:          # a line torn by a power cut has no newline: start on a new line, or the next record is glued to it and lost as well (2026-11-26)
        with open(path, 'rb') as f:
            f.seek(0, 2)
            if f.tell() > 0: f.seek(-1, 2); lead = '' if f.read(1) == b'\n' else '\n'
    except OSError: pass
    with open(path, 'a') as f: f.write(lead +json.dumps(obj, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())

def read_lines(path):
    out = []
    try:
        for l in open(path):
            try: out.append(json.loads(l))
            except ValueError: continue            # a torn last line after a power cut
    except OSError: pass
    return out

def fsync_dir(d):
    try: fd = os.open(d, os.O_RDONLY); os.fsync(fd); os.close(fd)
    except OSError: pass

def frozen_source(src, f):
    """Path of one frozen file: in `src` itself (flat layout) or in a sibling folder of scripts/ (shell_sim.py and shell_metrics.py live in scripts/shell_model/)."""
    p = os.path.join(src, f)
    if os.path.isfile(p): return p
    par = os.path.dirname(os.path.abspath(src))
    for d in sorted(os.listdir(par)):
        q = os.path.join(par, d, f)
        if os.path.isfile(q): return q
    return p

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def git_info(directory):
    out = dict(git_commit='', git_dirty=None)
    try:
        out['git_commit'] = subprocess.run(['git', '-C', directory, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=20).stdout.strip()
        out['git_dirty'] = bool(subprocess.run(['git', '-C', directory, 'status', '--porcelain', '--', 'scripts'], capture_output=True, text=True, timeout=20).stdout.strip())      # uncommitted changes in scripts/: the commit then does not describe the code, the file hashes do
    except Exception: pass
    return out

def versions():
    import importlib.metadata as im, platform
    d = dict(python=platform.python_version(), host=platform.node())
    for p in ('numpy', 'scipy', 'jax', 'jaxlib'):
        try: d[p] = im.version(p)
        except Exception: d[p] = ''
    return d

class xlock:
    """Exclusive-create lock file (works on Linux and Windows); a lock older than `stale` seconds is taken over."""
    def __init__(s, path, stale=30): s.path, s.stale = path, stale
    def __enter__(s):
        t0 = time.time()
        while True:
            try: s.fd = os.open(s.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); return s
            except (FileExistsError, PermissionError) as e:
                if isinstance(e, PermissionError) and os.name == 'posix': raise          # on Windows a lock file that another thread is deleting raises PermissionError: retry
                try:
                    if time.time() - os.path.getmtime(s.path) > s.stale: os.replace(s.path, s.path + '.stale')
                except OSError: pass
                if time.time() - t0 > 60: raise TimeoutError('lock %s' % s.path)
                time.sleep(0.01)
    def __exit__(s, *a):
        os.close(s.fd)
        try: os.remove(s.path)
        except OSError: pass

def _wjson_retry(path, obj, tries=40):
    """campaign.wjson (atomic, fsync) with a short retry: on Windows os.replace fails while a reader has the file open (never on the Linux server)."""
    for k in range(tries):
        try: return cm.wjson(path, obj)
        except PermissionError:
            if k == tries - 1: raise
            time.sleep(0.05)

def dir_size(d):
    n = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            try: n += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return n


# ------------------------------------------------------------------ one campaign
class Camp:
    def __init__(s, name):
        if not isinstance(name, str) or not NAME_RE.match(name): raise ValueError('bad campaign name')
        s.name = name; s.d = os.path.join(SHELL_ROOT, name)
        for k, f in dict(spec='spec.json', queue='queue.jsonl', index='index.jsonl', control='control.json', runner='runner.json', lock='runner.lock', delog='deletions.log', nextid='next_id', results='results.csv').items(): setattr(s, k, os.path.join(s.d, f))
        s.runs = os.path.join(s.d, 'runs'); s.tmp = os.path.join(s.d, '.tmp'); s.quar = os.path.join(s.d, 'quarantine')

    def exists(s): return os.path.isfile(s.spec)
    def spec_(s): return cm.rjson(s.spec, {}) or {}
    def limits(s):
        lim = dict(DEFAULT_LIMITS); lim.update((s.spec_().get('limits') or {}))
        if os.environ.get('SHELL_MIN_FREE_GB'): lim['min_free_gb'] = float(os.environ['SHELL_MIN_FREE_GB'])         # test hooks
        if os.environ.get('SHELL_MAX_TOTAL_GB'): lim['max_total_gb'] = float(os.environ['SHELL_MAX_TOTAL_GB'])
        return lim

    # -- frozen code
    def code_dir(s): return os.path.join(s.d, 'code')
    def manifest(s): return cm.rjson(os.path.join(s.code_dir(), 'MANIFEST.json'))
    def freeze(s, src=None):
        """Copy FROZEN_FILES from scripts/ into <campaign>/code/ and write MANIFEST.json. Write-once: refuses if code/ exists (new code = new campaign)."""
        src = src or HERE; dst = s.code_dir()
        if os.path.exists(dst): raise FileExistsError('frozen code folder %s already exists; never overwritten (new code = new campaign)' % dst)
        tmp = dst + '.tmp'; os.makedirs(tmp)
        for f in FROZEN_FILES: shutil.copy2(frozen_source(src, f), os.path.join(tmp, f))
        files = {f: file_sha256(os.path.join(tmp, f)) for f in FROZEN_FILES}
        cm.wjson(os.path.join(tmp, 'MANIFEST.json'), dict(frozen=cm.now(), source_dir=os.path.abspath(src), files=files, shell_sim_sha256=files['shell_sim.py'], **git_info(ROOT), **versions()))
        os.rename(tmp, dst); fsync_dir(s.d)
    def verify_code(s):
        """'' if the frozen files match MANIFEST.json (or the campaign has no frozen copy: legacy), otherwise a sentence."""
        if not os.path.isdir(s.code_dir()): return ''
        man = s.manifest()
        if not man: return 'frozen code folder has no readable MANIFEST.json'
        for f, h in (man.get('files') or {}).items():
            p = os.path.join(s.code_dir(), f)
            if not os.path.isfile(p): return 'frozen file %s is missing' % f
            if file_sha256(p) != h: return 'frozen file %s no longer matches MANIFEST.json (changed after freezing)' % f
        return ''

    @staticmethod
    def create(name, spec_extra=None):
        s = Camp(name)
        if os.path.exists(s.d): raise FileExistsError('campaign %s already exists; never overwritten' % name)
        for d in (s.runs, s.tmp, s.quar): os.makedirs(d)
        s.freeze()
        spec = dict(name=name, kind='shell', code='code', created=time.strftime('%Y-%m-%dT%H:%M:%S'), limits=dict(DEFAULT_LIMITS), thresholds=dict(fold_min_turn_rad=0.2),
                    notes='limits are UNSOURCED (K approved 2026-11-21); fold_min_turn_rad 0.2 copied from campaigns/fidelity/spec.json (chosen by its author)')
        if spec_extra and spec_extra.get('limits'): spec['notes'] += '; limits replaced at creation: ref_time_s and ref_vertices measured for this policy'
        spec.update(spec_extra or {}); cm.wjson(s.spec, spec); cm.wjson(s.control, dict(state='pause', t=cm.now())); open(s.nextid, 'w').write('1')
        return s

    # -- control and runner state
    def get_control(s): return (cm.rjson(s.control, {}) or {}).get('state', 'pause')
    def set_control(s, state, why=''):
        if state not in ('run', 'pause', 'stop'): raise ValueError('state must be run, pause or stop')
        _wjson_retry(s.control, dict(state=state, t=cm.now(), why=why))
    def set_runner(s, **kw):
        d = dict(kw); d.update(t=cm.now(), pid=os.getpid(), boot_id=cm.boot_id()); _wjson_retry(s.runner, d)
    def runner_alive(s):
        lk = cm.rjson(s.lock)
        try: return bool(isinstance(lk, dict) and lk.get('pid') and cm.alive(lk))
        except Exception: return False

    # -- ids and queue
    def next_id(s):
        with xlock(s.nextid + '.lock'):
            try: n = int(open(s.nextid).read().strip())
            except (OSError, ValueError): n = 1
            seen = [int(x['id'][1:]) for x in read_lines(s.queue) if ID_RE.match(str(x.get('id', '')))] + [int(x[1:]) for x in (os.listdir(s.runs) if os.path.isdir(s.runs) else []) if ID_RE.match(x)]
            n = max([n] + [m + 1 for m in seen]); tmp = s.nextid + '.tmp'
            with open(tmp, 'w') as f: f.write(str(n + 1)); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, s.nextid)
        return 'S%06d' % n

    def queue_items(s): return [x for x in read_lines(s.queue) if ID_RE.match(str(x.get('id', '')))]
    def index_last(s):
        d = {}
        for x in read_lines(s.index):
            if x.get('id'): d[x['id']] = x
        return d
    def known_sha(s): return {x.get('sha1') for x in s.queue_items()}

    def policy(s): return s.spec_().get('policy')
    def policy_problems(s, st, control=None):
        pol = s.policy()
        if not pol: return []
        import shell_policy as sp
        return sp.violations(st, pol, control)
    def enqueue(s, settings, source='manual', allow_duplicate=False, tier=5, control=None, label=''):
        """Append one run to the queue. Returns the new id, or None if the same settings are already queued or run (the shell is deterministic: give another seed to repeat). If the campaign has a policy
        (spec.json 'policy', approved ranges and ESSENTIAL settings), an entry that breaks it is REFUSED (ValueError) - except a labelled control run that changes only the allowed keys. tier: lower runs first
        (0 seeds, 1 controls, 2 variants, 5 proposer, 9 filler)."""
        pol = s.policy(); st = normalise(settings, (pol or {}).get('ranges', {}).get('max_stitches', 6000)); h = sha(st); bad = s.policy_problems(st, control)
        if bad: raise ValueError('refused by the campaign policy: ' + '; '.join(bad))
        if control and pol and sum(1 for x in s.queue_items() if x.get('control')) >= pol['controls']['max_per_campaign']: raise ValueError('refused by the campaign policy: at most %d control runs per campaign' % pol['controls']['max_per_campaign'])
        with xlock(os.path.join(s.d, 'queue.lock')):
            if not allow_duplicate and h in s.known_sha(): return None
            rid = s.next_id(); line = dict(id=rid, t=cm.now(), source=source, settings=st, sha1=h, tier=tier)
            if control: line['control'] = control
            if label: line['label'] = label
            append_line(s.queue, line)
        return rid

    def finished_ids(s):
        ids = {x for x in (os.listdir(s.runs) if os.path.isdir(s.runs) else []) if ID_RE.match(x)}
        return ids | {i for i, x in s.index_last().items() if x.get('status') in TERMINAL}
    def pending(s):
        fin = s.finished_ids(); return sorted([x for x in s.queue_items() if x['id'] not in fin], key=lambda x: (x.get('tier', 5), x['id']))

    # -- results
    def rows(s):
        last = s.index_last(); out = []
        for q in s.queue_items():
            x = last.get(q['id'], {}); st = q['settings']; m = x.get('metrics') or {}
            r = dict(run=q['id'], status=x.get('status', 'queued'), source=q.get('source', ''), tier=q.get('tier', 5), control=bool(q.get('control')), label=q.get('label', ''), heavy_deleted=bool(x.get('heavy_deleted')), wall_s=x.get('wall_s'), reason=x.get('reason', ''),
                     matrix='%dx%d' % (len(st['matrix']), len(st['matrix'][0])), knit_fraction=round(sum(v == 1 for r_ in st['matrix'] for v in r_) / (len(st['matrix']) * len(st['matrix'][0])), 4))
            r.update({k: st.get(k) for k in SETTING_COLS}); r.update({k: m.get(k) for k in METRIC_COLS}); cd = x.get('code') or {}
            r.update(git_commit=(cd.get('git_commit') or '')[:10], git_dirty=cd.get('git_dirty'), shell_sim_sha256=(cd.get('shell_sim_sha256') or '')[:12], frozen=cd.get('frozen')); out.append(r)
        return out
    def write_results(s):
        rows = s.rows()
        if not rows: return
        head = list(rows[0].keys()); tmp = '%s.tmp.%d' % (s.results, os.getpid())          # per-process name: the page's delete also rewrites results.csv (2026-11-26)
        with open(tmp, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(head); [w.writerow(['' if r[k] is None else r[k] for k in head]) for r in rows]; f.flush(); os.fsync(f.fileno())
        os.replace(tmp, s.results)

    # -- status
    def progress(s, rid):
        for d in sorted(glob.glob(os.path.join(s.tmp, rid + '.*'))):
            try:
                g = [l for l in open(os.path.join(d, 'shell.log'), errors='replace') if l.strip().startswith('growth')]
                if g: return dict(growth_steps_done=len(g), last=g[-1].strip())
            except OSError: pass
        return {}
    def status(s):
        lim = s.limits(); rn = cm.rjson(s.runner, {}) or {}; last = s.index_last(); fin = s.finished_ids(); counts = dict(queued=0, running=0, done=0, failed=0, timeout=0, needs_approval=0, rejected=0)
        for q in s.queue_items():
            st = (last.get(q['id']) or {}).get('status', 'done' if q['id'] in fin else 'queued')
            if st == 'approved': st = 'queued'          # shell-approve put it back in the queue
            if st == 'queued' and rn.get('current') == q['id'] and s.runner_alive(): st = 'running'
            counts[st] = counts.get(st, 0) + 1
        walls = [x.get('wall_s') for x in last.values() if x.get('status') == 'done' and isinstance(x.get('wall_s'), (int, float))]
        mean = sum(walls) / len(walls) if walls else None
        man = s.manifest() or {}; du = shutil.disk_usage(s.d if os.path.isdir(s.d) else SHELL_ROOT)
        return dict(name=s.name, control=s.get_control(), runner_alive=s.runner_alive(), runner=rn, counts=counts, mean_wall_s=mean, eta_s=(mean * counts['queued'] if mean else None),
                    progress=(s.progress(rn['current']) if rn.get('current') and s.runner_alive() else {}), code=dict(git_commit=man.get('git_commit'), git_dirty=man.get('git_dirty'), shell_sim_sha256=man.get('shell_sim_sha256'), frozen=man.get('frozen')), free_gb=round(du.free / 1e9, 2), total_gb=round(dir_size(s.d) / 1e9, 4), limits=lim)

    # -- one run (read only; used by the page)
    def run_path(s, rid):
        """Real path of a FINISHED run folder runs/<id>. Guard: id must be S + 6 digits, a real directory (no symlink) directly inside runs/, inside the shell-runs root. ValueError otherwise."""
        if not isinstance(rid, str) or not ID_RE.match(rid): raise ValueError('bad run id')
        p = os.path.join(s.runs, rid)
        if os.path.islink(p) or not os.path.isdir(p): raise ValueError('run %s is not a finished run' % rid)
        rp = os.path.realpath(p); root = os.path.realpath(SHELL_ROOT)
        if os.path.dirname(rp) != os.path.realpath(s.runs) or os.path.commonpath([root, rp]) != root: raise ValueError('run %s resolves outside the runs folder' % rid)
        return rp
    def seed_info(s, source):
        """The seed a run descends from, with the real-fold text of K's report (seeds.json): source 'seed:seed03', 'seed:pattern1:tiles4', 'variant:seed03', 'filler:seed03'."""
        parts = (source or '').split(':'); kind = parts[0]; sid = parts[1] if len(parts) > 1 else ''; f = (s.spec_().get('plan') or {}).get('seeds_file')
        try: ent = next(x for x in json.load(open(f, encoding='utf8'))['seeds'] if x['id'] == sid)
        except Exception: return {}
        return dict(kind=kind, seed=sid, sample=ent.get('sample'), real_tm_axis=ent.get('real_tm_axis'), real_fold=ent.get('real_fold'), report_sections=[ent.get('report_section_prep'), ent.get('report_section_result')])
    def run_detail(s, rid):
        rp = s.run_path(rid); q = next((x for x in s.queue_items() if x['id'] == rid), {}); ix = s.index_last().get(rid, {})
        try: tail = [l.rstrip() for l in open(os.path.join(rp, 'shell.log'), errors='replace')][-25:]
        except OSError: tail = []
        return dict(run=rid, campaign=s.name, source=q.get('source', ''), tier=q.get('tier'), label=q.get('label', ''), control=q.get('control'), settings=q.get('settings') or cm.rjson(os.path.join(rp, 'settings.json')),
                    status=ix.get('status', 'done'), heavy_deleted=bool(ix.get('heavy_deleted')), wall_s=ix.get('wall_s'), metrics=cm.rjson(os.path.join(rp, 'metrics.json'), {}), provenance=cm.rjson(os.path.join(rp, 'provenance.json'), {}),
                    files={f: os.path.isfile(os.path.join(rp, f)) for f in HEAVY}, seed=s.seed_info(q.get('source', '')), log_tail=tail)

    # -- finishing runs
    def publish(s, tmp, rid, status_extra):
        """Atomic: complete .tmp folder -> runs/<id>. Refuses to overwrite."""
        final = os.path.join(s.runs, rid)
        if os.path.exists(final): raise FileExistsError('run folder %s already exists; never overwritten' % final)
        m = cm.rjson(os.path.join(tmp, 'metrics.json'), {}) or {}; pv = cm.rjson(os.path.join(tmp, 'provenance.json'), {}) or {}
        code = dict(git_commit=pv.get('git_commit', ''), git_dirty=pv.get('git_dirty'), shell_sim_sha256=pv.get('shell_sim_sha256', ''), frozen=pv.get('frozen'), matches_manifest=pv.get('matches_manifest'))
        cm.wjson(os.path.join(tmp, 'status.json'), dict(run=rid, status='done', t=cm.now(), code=code, **status_extra))
        os.rename(tmp, final); fsync_dir(s.runs)
        append_line(s.index, dict(id=rid, status='done', t=cm.now(), metrics=m, code=code, **status_extra)); s.write_results()
    def quarantine(s, tmp, rid, tag):
        dst = os.path.join(s.quar, '%s.%s.%s' % (rid, tag, time.strftime('%Y%m%d-%H%M%S')))
        n = 0
        while os.path.exists(dst + ('' if not n else '-%d' % n)): n += 1
        dst += '' if not n else '-%d' % n; os.rename(tmp, dst); fsync_dir(s.quar); return dst
    def fail(s, tmp, rid, status, reason, wall):
        s.quarantine(tmp, rid, status); append_line(s.index, dict(id=rid, status=status, t=cm.now(), reason=reason, wall_s=wall)); s.write_results()


# ------------------------------------------------------------------ the runner
def kill_group(p, grace=60):
    if os.name != 'posix': p.terminate(); return
    try: os.killpg(p.pid, signal.SIGTERM)
    except OSError: pass
    for _ in range(int(grace)):
        if p.poll() is not None: return
        time.sleep(1)
    try: os.killpg(p.pid, signal.SIGKILL)
    except OSError: pass

def last_line(path):
    try:
        l = [x.strip() for x in open(path, errors='replace') if x.strip()]; return l[-1][:300] if l else ''
    except OSError: return ''

def disk_guard(c):
    lim = c.limits(); du = shutil.disk_usage(c.d)
    if du.free / 1e9 < lim['min_free_gb']: return 'free disk %.1f GB is below the limit of %.1f GB' % (du.free / 1e9, lim['min_free_gb'])
    tot = dir_size(SHELL_ROOT) / 1e9
    if tot > lim['max_total_gb']: return 'the shell-runs folder holds %.2f GB, above the limit of %.1f GB' % (tot, lim['max_total_gb'])
    return ''

def reconcile(c):
    """On runner start and between runs. Returns '' when done, or a sentence if a worker of an earlier runner is still alive (wait for it). Complete .tmp folders of dead workers are published; the rest
    is moved to quarantine (renamed, nothing deleted) and the run goes back to the queue; after max_interrupts interruptions it is marked needs_approval."""
    for d in sorted(glob.glob(os.path.join(c.tmp, '*'))):
        base = os.path.basename(d); rid = base.split('.')[0]
        if not ID_RE.match(rid) or not os.path.isdir(d): continue
        wk = cm.rjson(os.path.join(d, 'worker.json'), {}) or {}
        try: alive = bool(wk.get('pid') and cm.alive(wk))
        except Exception: alive = False
        if alive:
            try:          # a worker adopted from an earlier runner still gets its time limit (it had none, 2026-11-26)
                tmo = timeout_for(normalise(cm.rjson(os.path.join(d, 'settings.json'), {}) or {}), c.limits())
                if time.time() - float(wk.get('t_start') or time.time()) > tmo:
                    cm.say('adopted worker of %s exceeded %.0f s: terminating its process group' % (rid, tmo)); os.killpg(int(wk['pid']), signal.SIGTERM)
            except (ValueError, KeyError, TypeError, OSError, AttributeError): pass
            return 'worker of %s is still running (pid %s)' % (rid, wk.get('pid'))
        if os.path.exists(os.path.join(c.runs, rid)): c.quarantine(d, rid, 'duplicate'); continue
        if os.path.exists(os.path.join(d, 'metrics.json')):                       # finished, the runner died before the rename
            try: wall = round(os.path.getmtime(os.path.join(d, 'metrics.json')) - float(wk.get('t_start') or 0), 1)
            except OSError: wall = None
            c.publish(d, rid, dict(wall_s=wall, recovered=True)); continue
        c.quarantine(d, rid, 'interrupted'); n = len(glob.glob(os.path.join(c.quar, rid + '.interrupted.*')))
        if n >= c.limits()['max_interrupts']: append_line(c.index, dict(id=rid, status='needs_approval', t=cm.now(), reason='interrupted %d times' % n)); c.write_results()
    for rid in sorted(x for x in os.listdir(c.runs) if ID_RE.match(x)) if os.path.isdir(c.runs) else []:      # published but not indexed (cut between rename and index line)
        if rid not in c.index_last():
            st = cm.rjson(os.path.join(c.runs, rid, 'status.json'), {}) or {}; m = cm.rjson(os.path.join(c.runs, rid, 'metrics.json'), {}) or {}
            append_line(c.index, dict(id=rid, status='done', t=cm.now(), metrics=m, wall_s=st.get('wall_s'), recovered=True, code=st.get('code')))          # code = provenance, as on a normal publish (was missing, 2026-11-26); c.write_results()
    return ''

def run_one(c, item, poll=1.0):
    rid = item['id']; s = item['settings']; lim = c.limits(); att = 1 + len(glob.glob(os.path.join(c.quar, rid + '.*'))) + len(glob.glob(os.path.join(c.tmp, rid + '.*')))
    tmp = os.path.join(c.tmp, '%s.%d' % (rid, att)); os.makedirs(tmp)
    sp = os.path.join(tmp, 'settings.json'); cm.wjson(sp, dict(s, fold_min_turn_rad=(c.spec_().get('thresholds') or {}).get('fold_min_turn_rad', 0.2)))
    cmd = [sys.executable, '-u', os.path.join(HERE, 'shell_run.py'), '--settings', sp, '--out', tmp]; tmo = timeout_for(s, lim); t0 = time.time(); m0 = time.monotonic()          # elapsed time on the monotonic clock: a clock step after a power cut must not kill or spare a run (2026-11-26)
    log = open(os.path.join(tmp, 'shell.log'), 'a')
    kw = dict(start_new_session=True, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN)) if os.name == 'posix' else {}      # own session and SIGHUP ignored, like the runner
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=HERE, close_fds=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), **kw)      # no .pyc written next to the frozen code
    pi = cm.proc_info(p.pid); cm.wjson(os.path.join(tmp, 'worker.json'), dict(pid=p.pid, boot_id=cm.boot_id(), start_ticks=pi[0] if pi else '', t_start=t0))
    c.set_runner(state='running', current=rid, timeout_s=round(tmo), worker_pid=p.pid); cm.say('run %s started (pid %d, timeout %.0f s)' % (rid, p.pid, tmo)); killed = False
    while p.poll() is None:
        if time.monotonic() - m0 > tmo: killed = True; cm.say('run %s exceeded %.0f s: terminating its own process group' % (rid, tmo)); kill_group(p); p.wait(); break
        time.sleep(poll)
    log.close(); wall = round(time.monotonic() - m0, 1)
    if killed: c.fail(tmp, rid, 'timeout', 'timeout after %.0f s (limit %.0f s, UNSOURCED rule)' % (wall, tmo), wall)
    elif p.returncode == 0 and os.path.exists(os.path.join(tmp, 'metrics.json')): c.publish(tmp, rid, dict(wall_s=wall, attempt=att)); cm.say('run %s done in %.0f s' % (rid, wall))
    else: c.fail(tmp, rid, 'failed', 'exit %s: %s' % (p.returncode, last_line(os.path.join(tmp, 'shell.log')) or 'no output'), wall); cm.say('run %s failed (exit %s)' % (rid, p.returncode))
    c.set_runner(state='between', current=None)

def acquire_lock(c):
    import fcntl
    lock = open(c.lock, 'a+')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: sys.exit('another runner already holds %s' % c.lock)
    pi = cm.proc_info(os.getpid()); lock.seek(0); lock.truncate(); lock.write(json.dumps(dict(pid=os.getpid(), boot_id=cm.boot_id(), start_ticks=pi[0] if pi else '', started=cm.now()))); lock.flush()
    return lock

def serve(c, poll_s=5.0, stop_when_empty=False, lock=True, run_poll=1.0):
    """Runner loop. control.json: run = take the next queued run; pause = finish the run in progress, then wait; stop = finish the run in progress, then exit. Checked BETWEEN runs only; nothing is killed
    for a control change. Ends on its own only for the free-disk guard (control set to stop, reason recorded) or when stop_when_empty."""
    if not c.exists(): sys.exit('no such shell campaign: %s' % c.name)
    lk = acquire_lock(c) if lock else None
    cm.say('shell runner started for %s (pid %d)' % (c.name, os.getpid())); c.set_runner(state='starting', current=None)
    while True:
        ctl = c.get_control()
        if ctl == 'stop': break
        wait = reconcile(c)
        if wait: c.set_runner(state='waiting', current=None, reason=wait); time.sleep(poll_s); continue
        if ctl == 'pause': c.set_runner(state='paused', current=None); time.sleep(poll_s); continue
        pend = c.pending()
        if not pend:
            if stop_when_empty: break
            c.set_runner(state='idle', current=None); time.sleep(poll_s); continue
        g = disk_guard(c)
        if g: c.set_runner(state='guard_stopped', current=None, reason=g); c.set_control('stop', why='disk guard: ' + g); cm.say('guard: ' + g); break
        item = pend[0]; viol = c.policy_problems(item['settings'], item.get('control'))
        if viol: append_line(c.index, dict(id=item['id'], status='rejected', t=cm.now(), reason='policy: ' + '; '.join(viol))); c.write_results(); cm.say('run %s rejected by the policy: %s' % (item['id'], viol[0])); continue
        bad = c.verify_code()
        if bad: c.set_runner(state='guard_stopped', current=None, reason='frozen code: ' + bad); c.set_control('stop', why='frozen code check: ' + bad); cm.say('guard: frozen code: ' + bad); break
        why = cm.busy_reason()
        if why: c.set_runner(state='waiting', current=None, reason='another job is running: ' + why); time.sleep(poll_s); continue
        run_one(c, pend[0], run_poll)
    if c.get_runner_state() != 'guard_stopped': c.set_runner(state='stopped', current=None)
    cm.say('shell runner finished'); return lk

Camp.get_runner_state = lambda s: (cm.rjson(s.runner, {}) or {}).get('state', '')


# ------------------------------------------------------------------ delete (heavy files only), used by the page
def delete_heavy(c, ids, client='', dry=False):
    """Remove shell.npz and shell.bin of the chosen finished runs. Path guard: id must be S + 6 digits, runs/<id> must be a real directory (no symlink) directly inside runs/, which is inside SHELL_ROOT;
    runs in progress or not finished are refused. The log line is written and fsynced BEFORE anything is removed. settings.json, metrics.json, status.json, the log and the index line stay (index gets heavy_deleted)."""
    if not isinstance(ids, list) or not ids or len(ids) > 500: raise ValueError('give 1 to 500 run ids')
    root = os.path.realpath(SHELL_ROOT); runs = os.path.realpath(c.runs); plan = []
    if os.path.dirname(runs) != os.path.realpath(c.d) or os.path.commonpath([root, runs]) != root: raise ValueError('campaign folder is outside the shell-runs root')
    cur = (cm.rjson(c.runner, {}) or {}).get('current')
    for rid in ids:
        if not isinstance(rid, str) or not ID_RE.match(rid): raise ValueError('bad run id %r' % (rid,))
        p = os.path.join(c.runs, rid)
        if os.path.islink(p) or not os.path.isdir(p): raise ValueError('run %s is not a finished run' % rid)
        rp = os.path.realpath(p)
        if os.path.dirname(rp) != runs or os.path.commonpath([root, rp]) != root: raise ValueError('run %s resolves outside the runs folder' % rid)
        if rid == cur and c.runner_alive(): raise ValueError('run %s is in progress' % rid)
        files = [(f, os.path.join(rp, f)) for f in HEAVY if os.path.isfile(os.path.join(rp, f)) and not os.path.islink(os.path.join(rp, f))]
        plan.append((rid, files, sum(os.path.getsize(fp) for _, fp in files)))
    total = sum(b for _, _, b in plan); summary = dict(ids=[p[0] for p in plan], bytes=total, files=sum(len(p[1]) for p in plan))
    if dry: return summary
    append_line(c.delog, dict(t=cm.now(), client=client, campaign=c.name, ids=summary['ids'], bytes=total, files=[f for _, fl, _ in plan for f, _ in fl], settings_sha1={q['id']: q['sha1'] for q in c.queue_items() if q['id'] in summary['ids']}))
    last = c.index_last()
    for rid, files, _ in plan:
        for _, fp in files:
            try: os.remove(fp)
            except OSError: pass
        base = dict(last.get(rid) or dict(id=rid, status='done')); base.update(heavy_deleted=True, t=cm.now()); append_line(c.index, base)
    c.write_results(); return summary


def busy_message():
    """'' or a sentence: a shell runner is working (a run in progress or waiting to start). A paused or idle runner does not block other jobs; it checks busy_reason() before every run."""
    try:
        for d in sorted(glob.glob(os.path.join(SHELL_ROOT, '*'))):
            if not os.path.isfile(os.path.join(d, 'spec.json')): continue
            c = Camp(os.path.basename(d))
            if c.runner_alive() and (cm.rjson(c.runner, {}) or {}).get('state') in ('running', 'starting'): return "shell campaign '%s' is running a run. Pause it or wait for the run to finish." % c.name
            for w in glob.glob(os.path.join(c.tmp, '*', 'worker.json')):
                wk = cm.rjson(w, {}) or {}
                if wk.get('pid') and cm.alive(wk): return "a shell run of campaign '%s' is still running." % c.name
    except OSError: pass
    return ''


def launch_detached(name, poll_s=5):
    """Start the runner fully detached: own session (setsid), SIGHUP ignored, stdin /dev/null, stdout to a log file, so it survives a closed terminal or SSH tunnel. It starts from the campaign's FROZEN copy.
    (setsid does not protect against systemd: restarting a service kills every process in its cgroup unless that unit has KillMode=process; see deploy/ and CLAUDE.md TO DO 4.)"""
    os.makedirs(cm.LOG_DIR, exist_ok=True); lp = os.path.join(cm.LOG_DIR, 'shell_campaign_%s.log' % name)
    script = os.path.join(Camp(name).code_dir(), 'campaign.py'); script = script if os.path.isfile(script) else os.path.join(HERE, 'campaign.py')
    cmd = [sys.executable, script, 'shell-run', name, '--poll-s', str(poll_s)]
    kw = dict(start_new_session=True, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN)) if os.name == 'posix' else {}
    p = subprocess.Popen(cmd, stdout=open(lp, 'a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, close_fds=True, env=dict(os.environ, SHELL_RUNS_DIR=SHELL_ROOT, CAMPAIGN_LOG_DIR=cm.LOG_DIR, PYTHONDONTWRITEBYTECODE='1'), **kw)
    return p.pid, lp


# ------------------------------------------------------------------ command line (called from campaign.py: shell-plan, shell-enqueue, shell-run, shell-status, shell-control)
def cli(cmd, argv):
    ap = argparse.ArgumentParser(prog='campaign.py ' + cmd); ap.add_argument('name')
    if cmd == 'shell-plan':
        ap.add_argument('--pattern1-file', default=os.path.join(CAMPAIGNS, 'pattern1_matrix.txt'), help='matrix file of the control run S000001 (13 Nov success shell: Ne 12, CPI 22, WPI 16, tiles 3, defaults)')
        ap.add_argument('--policy', help='policy JSON (campaigns/shell_policy.json): builds the full seeded queue (seeds, controls, variants, filler) and enforces the policy')
        ap.add_argument('--seeds', default=os.path.join(CAMPAIGNS, 'shell_seeds', 'seeds.json')); ap.add_argument('--variants', type=int, default=91); ap.add_argument('--filler', type=int, default=0); ap.add_argument('--rng-seed', type=int, default=20261121)
        ap.add_argument('--n', type=int, default=1000, help='policy version 2 (shell2): number of generated runs after the seeds'); ap.add_argument('--ref-time-s', type=float, help='timeout reference: seconds of one run of --ref-vertices vertices with this policy'); ap.add_argument('--ref-vertices', type=int)
    if cmd == 'shell-enqueue': ap.add_argument('file', help='JSON file: one settings object or a list of them')
    if cmd == 'shell-run': ap.add_argument('--detach', action='store_true'); ap.add_argument('--poll-s', type=float, default=5.0); ap.add_argument('--stop-when-empty', action='store_true'); ap.add_argument('--allow-unfrozen', action='store_true', help='run a campaign that has no code/ folder from the live scripts (not recommended)')
    if cmd == 'shell-control': ap.add_argument('state', choices=['run', 'pause', 'stop'])
    if cmd == 'shell-approve': ap.add_argument('run_id', help='a run marked needs_approval or timeout'); ap.add_argument('--why', default='approved by K on the command line')
    a = ap.parse_args(argv)
    if FROZEN and cmd in ('shell-run', 'shell-plan', 'shell-freeze') and os.path.basename(os.path.dirname(HERE)) != a.name: sys.exit('this is the frozen copy of campaign %s; it refuses to run campaign %s' % (os.path.basename(os.path.dirname(HERE)), a.name))
    if cmd == 'shell-plan':
        if a.policy:
            import shell_policy as sp
            pol = sp.load(a.policy); sj = json.load(open(a.seeds, encoding='utf8')); base = os.path.dirname(os.path.abspath(a.seeds))
            seeds = [dict(id=x['id'], matrix=matrix_from_text(open(os.path.join(base, x['file']), encoding='utf8').read())) for x in sj['seeds']]
            v2 = pol.get('version', 1) >= 2
            plan = sp.build_plan_v2(pol, seeds, a.rng_seed, a.n) if v2 else sp.build_plan(pol, seeds, a.rng_seed, a.variants, a.filler)
            extra = dict(policy=pol, plan=dict(rng_seed=a.rng_seed, variants=a.variants, filler=a.filler, n=a.n if v2 else None, policy_version=pol.get('version', 1), seeds_file=os.path.abspath(a.seeds), n_entries=len(plan), seeds_sha1=hashlib.sha1(open(a.seeds, 'rb').read()).hexdigest()))
            if a.ref_time_s and a.ref_vertices: extra['limits'] = dict(DEFAULT_LIMITS, ref_time_s=a.ref_time_s, ref_vertices=a.ref_vertices)
            c = Camp.create(a.name, extra)
            n = 0
            for e in plan:
                if c.enqueue(e['settings'], source=e['source'], tier=e['tier'], control=e['control'], label=e['label']): n += 1
            print('created %s with policy; %d runs queued (paused: shell-control %s run, then shell-run %s --detach)' % (c.d, n, a.name, a.name)); return
        c = Camp.create(a.name); rid = c.enqueue(dict(matrix=open(a.pattern1_file).read(), tiles=3, cpi=22, wpi=16, ne=12), source='seed:13Nov-success')
        print('created %s; control run %s queued (paused: use shell-control %s run, then shell-run %s --detach)' % (c.d, rid, a.name, a.name)); return
    c = Camp(a.name)
    if not c.exists(): sys.exit('no such shell campaign: %s' % a.name)
    if cmd == 'shell-enqueue':
        j = json.load(open(a.file)); ids = [c.enqueue(x, source='file', tier=4) for x in (j if isinstance(j, list) else [j])]; print('queued: %s' % ids)          # tier 4 (2026-11-26): ahead of the proposer and the filler, and never counted as a proposer run (tier 5)
    elif cmd == 'shell-approve':          # a needs_approval / timeout run goes back to the queue: pending() reads only the last index line, and 'approved' is not terminal (works for a frozen runner too)
        x = c.index_last().get(a.run_id)
        if not x or x.get('status') not in ('needs_approval', 'timeout'): sys.exit('%s is not marked needs_approval or timeout (status %s)' % (a.run_id, (x or {}).get('status')))
        append_line(c.index, dict(id=a.run_id, status='approved', t=cm.now(), reason=a.why)); c.write_results(); print('%s approved: it goes back to the queue (one more attempt per approval)' % a.run_id)
    elif cmd == 'shell-control': c.set_control(a.state, why='command line'); print('control = %s' % a.state)
    elif cmd == 'shell-status': print(json.dumps(c.status(), indent=1))
    elif cmd == 'shell-freeze': c.freeze(); print('frozen copy written: %s (MANIFEST.json holds the git commit and the file hashes)' % c.code_dir())
    elif cmd == 'shell-run':
        frozen_script = os.path.join(c.code_dir(), 'campaign.py')
        if FROZEN and os.path.basename(os.path.dirname(HERE)) != a.name: sys.exit('this is the frozen copy of campaign %s; it refuses to run campaign %s' % (os.path.basename(os.path.dirname(HERE)), a.name))
        if not FROZEN:
            if os.path.isfile(frozen_script): os.execve(sys.executable, [sys.executable, frozen_script, 'shell-run'] + argv, dict(os.environ, SHELL_RUNS_DIR=SHELL_ROOT))          # always run from the frozen copy
            if not a.allow_unfrozen: sys.exit('campaign %s has no frozen code folder: run  campaign.py shell-freeze %s  first (or --allow-unfrozen)' % (a.name, a.name))
        if a.detach: pid, lp = launch_detached(a.name, a.poll_s); print('shell runner started (own session, SIGHUP ignored): pid %d, log %s' % (pid, lp)); return
        serve(c, a.poll_s, a.stop_when_empty)
