#!/usr/bin/env python3
"""campaign.py - run many relax_job.py simulations one after another and record every result in one table (runs on the server).

    campaign.py plan   <name>              write manifest.json (numbered runs with exact parameter values) from spec.json
    campaign.py run    <name> [--dry-run] [--detach] [--poll-s 60] [--ignore-spec-change]
    campaign.py status <name>              counts and a table of the runs
    campaign.py stop   <name>              create the STOP file; the runner checks it BETWEEN runs and never kills anything for it

A campaign is the folder campaigns/<name>/ with spec.json, manifest.json, results.csv, STOP (only while a stop is requested) and runs/<NNNN>/.
Behaviour after a power cut, the metric definitions and the spec format are in campaigns/README.md.
Standard library plus numpy (numpy only for the fold metrics). Paths: ROOT (default ~/auxetic_project, override with CAMPAIGN_ROOT or LADDER_ROOT) and
LOG_DIR (default ROOT/logs, override with CAMPAIGN_LOG_DIR or LADDER_LOG_DIR); CAMPAIGN_DIR (default ROOT/campaigns) and CAMPAIGN_SCRIPTS (default ROOT/scripts) move the campaigns and the solver scripts.
"""
import argparse, csv, glob, hashlib, json, math, os, random, re, signal, socket, subprocess, sys, time

HOME = os.path.expanduser('~')
ROOT = os.environ.get('CAMPAIGN_ROOT') or os.environ.get('LADDER_ROOT') or os.path.join(HOME, 'auxetic_project')
LOG_DIR = os.environ.get('CAMPAIGN_LOG_DIR') or os.environ.get('LADDER_LOG_DIR') or os.path.join(ROOT, 'logs')
SCRIPTS = os.environ.get('CAMPAIGN_SCRIPTS') or os.path.join(ROOT, 'scripts')
CAMPAIGNS = os.environ.get('CAMPAIGN_DIR') or os.path.join(ROOT, 'campaigns')
WEB_RESULTS = os.path.join(ROOT, 'webui', 'results')
ACCEPTED_EXIT = (0, 2)            # relax_job: 0 = settled, 2 = ended as "failed" (not settled, e.g. outer cap reached). Both leave a usable relaxed state.
BUILD_PARAMS = {'tiles': int, 'ne': float, 'cpi': float, 'wpi': float}
RESERVED_RELAX = {'--folder', '--resume'}
FOREIGN_JOB_SCRIPTS = ('relax_job.py', 'tile_sim.py', 'shell_run.py', 'shell_fold.py', 'shell_target.py')      # shell_fold / shell_target run relax_job in-process (task 47), added 2026-11-26      # shell_run.py: one pure-shell run (shell_campaign.py); one job at a time across all job types
METRIC_COLS = ['exit_code', 'relax_state', 'settled_by', 'build_s', 'wall_s', 'vertices', 'rounds_used', 'outer_max', 'E_start', 'E_final', 'rel_dE_last', 'E_stretch', 'E_bend', 'E_contact',
               'min_dist_over_2R_start', 'min_dist_over_2R_end', 'rest_len_change_total', 'rest_len_change_max_abs_wale', 'shrink_x', 'shrink_y', 'z_range_over_R_end',
               'fold_kx_per_mm', 'fold_ky_per_mm', 'fold_turn_x_rad', 'fold_turn_y_rad', 'fold_direction', 'fold_amplitude_mm', 'fold_local_amplitude_mm',
               'wavelength_course_mm', 'wavelength_wale_mm', 'wl_course_power_fraction', 'wl_wale_power_fraction', 'reason']


def say(msg): print('%s  %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)


# ------------------------------------------------------------------ files
def wjson(path, obj):
    """Atomic and durable: write a temporary file, fsync it, rename it over the target."""
    tmp = '%s.tmp.%d' % (path, os.getpid())          # per-process name: the runner and the web page may write the same file at once (2026-11-26)
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(os.path.dirname(path) or '.', os.O_RDONLY); os.fsync(fd); os.close(fd)
    except OSError: pass

def rjson(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError): return default

def cdir(name): return os.path.join(CAMPAIGNS, name)
def rdir(name, rid): return os.path.join(CAMPAIGNS, name, 'runs', rid)
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S')


# ------------------------------------------------------------------ processes (Linux /proc)
def boot_id():
    try: return open('/proc/sys/kernel/random/boot_id').read().strip()
    except OSError: return ''

def proc_info(pid):
    """(start_ticks, state) from /proc/<pid>/stat, or None if the process does not exist."""
    try:
        d = open('/proc/%d/stat' % pid).read()
        rest = d[d.rindex(')') + 2:].split()
        return rest[19], rest[0]
    except (OSError, ValueError, IndexError): return None

def alive(att):
    """True if the process recorded in an attempt is the same process, on the same boot, and not a zombie."""
    if not att or not att.get('pid'): return False
    if att.get('boot_id') and att['boot_id'] != boot_id(): return False
    pi = proc_info(att['pid'])
    return bool(pi and pi[0] == att.get('start_ticks') and pi[1] != 'Z')

def cmdlines():
    out = []
    for p in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            pid = int(p.split('/')[2])
            args = open(p, 'rb').read().split(b'\0')
            pi = proc_info(pid)
            if pi and pi[1] != 'Z': out.append((pid, [a.decode('utf8', 'replace') for a in args if a]))
        except (OSError, ValueError): continue
    return out

def busy_reason():
    """Why a new run must wait: another relax_job.py / tile_sim.py process is alive (web-UI job, or a job of another campaign). Stale job_stage.json files
    (left by a power cut; no process works on that folder) are ignored. Returns '' if the machine is free."""
    for pid, args in cmdlines():
        if pid != os.getpid() and any(os.path.basename(a) in FOREIGN_JOB_SCRIPTS for a in args):
            return 'pid %d: %s' % (pid, ' '.join(args)[:140])
    return ''

def stale_stage_files():
    return sorted(glob.glob(os.path.join(WEB_RESULTS, '*', 'job_stage.json')))

def descendants_rss_kb(root_pid):
    kids = {root_pid}; total = 0; changed = True; table = {}
    for pid, _ in cmdlines():
        try: table[pid] = int(open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError): pass
    while changed:
        changed = False
        for pid, ppid in table.items():
            if ppid in kids and pid not in kids: kids.add(pid); changed = True
    for pid in kids:
        try:
            for l in open('/proc/%d/status' % pid):
                if l.startswith('VmRSS:'): total += int(l.split()[1])
        except (OSError, ValueError): pass
    return total


# ------------------------------------------------------------------ spec, design, manifest
def known_relax_flags():
    txt = open(os.path.join(SCRIPTS, 'yarn_model', 'relax_job.py')).read()
    return set(re.findall(r"add_argument\('(--[A-Za-z0-9-]+)'", txt))

def flag_of(name): return '--E' if name == 'E' else '--' + name.replace('_', '-')

def load_spec(name):
    p = os.path.join(cdir(name), 'spec.json')
    spec = rjson(p)
    if spec is None: sys.exit('cannot read %s' % p)
    errs = []
    for k in ('name', 'fabric', 'relax', 'vary', 'design', 'timeout_s', 'thresholds'):
        if k not in spec: errs.append("spec: missing key '%s'" % k)
    if errs: sys.exit('\n'.join(errs))
    if spec['name'] != name: errs.append("spec: name '%s' does not match the folder '%s'" % (spec['name'], name))
    fab = spec['fabric']
    if not (('matrix' in fab) ^ ('matrix_file' in fab)): errs.append('spec.fabric: give exactly one of matrix, matrix_file')
    for k in ('ne', 'cpi', 'wpi', 'tiles'):
        if k not in fab: errs.append("spec.fabric: missing '%s'" % k)
    flags = known_relax_flags()
    for k in spec['relax']:
        if k == 'kc_ratio': continue
        if flag_of(k) not in flags or flag_of(k) in RESERVED_RELAX: errs.append("spec.relax: '%s' is not an option of relax_job.py" % k)
    for need in ('E', 'mu'):
        if need not in spec['relax']: errs.append("spec.relax: '%s' is required (relax_job.py needs it)" % need)
    if ('kc_ratio' in spec['relax']) == ('k_contact' in spec['relax']): errs.append('spec.relax: give exactly one of kc_ratio (x EA/R, as the web UI) or k_contact')
    names = set()
    for v in spec['vary']:
        for k in ('name', 'min', 'max', 'scale', 'integer'):
            if k not in v: errs.append("spec.vary: entry %s lacks '%s'" % (v.get('name', '?'), k))
        n = v.get('name')
        if n in names: errs.append("spec.vary: '%s' twice" % n)
        names.add(n)
        if n not in BUILD_PARAMS and n != 'kc_ratio' and (flag_of(n or '') not in flags or flag_of(n or '') in RESERVED_RELAX): errs.append("spec.vary: '%s' is not a build or relax option" % n)
        if v.get('scale') not in ('linear', 'log'): errs.append("spec.vary: '%s' scale must be linear or log" % n)
        if v.get('scale') == 'log' and not (v.get('min', 0) > 0): errs.append("spec.vary: '%s' log scale needs min > 0" % n)
        if isinstance(v.get('min'), (int, float)) and isinstance(v.get('max'), (int, float)) and v['min'] >= v['max']: errs.append("spec.vary: '%s' needs min < max" % n)
    d = spec['design']
    if d.get('type') == 'latin_hypercube':
        if not isinstance(d.get('n_runs'), int) or d['n_runs'] < 1 or 'seed' not in d: errs.append('spec.design: latin_hypercube needs integer n_runs >= 1 and seed')
        if not spec['vary']: errs.append('spec.design: latin_hypercube needs at least one parameter in vary')
    elif d.get('type') == 'explicit':
        if not isinstance(d.get('runs'), list) or not d['runs']: errs.append('spec.design: explicit needs a non-empty list "runs" of parameter objects')
        else:
            for i, r in enumerate(d['runs']):
                for k in r:
                    if k not in BUILD_PARAMS and k != 'kc_ratio' and (flag_of(k) not in flags or flag_of(k) in RESERVED_RELAX): errs.append("spec.design.runs[%d]: '%s' is not a build or relax option" % (i, k))
    else: errs.append('spec.design.type must be latin_hypercube or explicit')
    th = spec['thresholds']
    for k in ('fold_min_turn_rad', 'wavelength_min_power_fraction'):
        if not isinstance(th.get(k), (int, float)): errs.append("spec.thresholds: '%s' (number) is required" % k)
    if not isinstance(spec['timeout_s'], (int, float)) or spec['timeout_s'] <= 0: errs.append('spec.timeout_s must be a positive number of seconds')
    if errs: sys.exit('spec problems:\n  ' + '\n  '.join(errs))
    return spec

def spec_hash(name): return hashlib.sha1(open(os.path.join(cdir(name), 'spec.json'), 'rb').read()).hexdigest()

def sig6(x): return float('%.6g' % x)

def design_points(spec):
    d = spec['design']
    if d['type'] == 'explicit': return [dict(r) for r in d['runs']]
    n, rng, vs = d['n_runs'], random.Random(d['seed']), spec['vary']
    cols = []
    for v in vs:
        perm = list(range(n)); rng.shuffle(perm); cols.append([(perm[i] + rng.random()) / n for i in range(n)])      # one point in each of n equal strata per parameter
    pts = []
    for i in range(n):
        p = {}
        for v, c in zip(vs, cols):
            u = c[i]
            if v['integer'] and v['scale'] == 'linear':          # equal strata per integer value: round() gave the two end values half the weight of the others (2026-11-26)
                p[v['name']] = min(int(v['max']), int(math.floor(v['min'] + u * (v['max'] - v['min'] + 1)))); continue
            x = v['min'] + u * (v['max'] - v['min']) if v['scale'] == 'linear' else math.exp(math.log(v['min']) + u * (math.log(v['max']) - math.log(v['min'])))
            p[v['name']] = int(round(x)) if v['integer'] else sig6(x)
        pts.append(p)
    return pts

def fabric_matrix(spec):
    f = spec['fabric']
    if 'matrix' in f: return f['matrix'].strip()
    p = os.path.join(CAMPAIGNS, f['matrix_file'])
    return open(p).read().strip()

def yarn_R(ne): return 25.4 / (28.0 * math.sqrt(ne)) / 2        # same formula as build_knit.py (physical mode)

def merged(spec, params):
    """(build settings, relax settings) for one run: fixed values overridden by the run's parameters."""
    b = {k: spec['fabric'][k] for k in BUILD_PARAMS}; r = dict(spec['relax'])
    for k, v in params.items():
        if k in BUILD_PARAMS: b[k] = v
        else: r[k] = v
    return b, r

def k_contact(r, R):
    """Same rule as webui/main.py _sim_cmd: kc = kc_ratio x EA / R, EA = E pi R^2 unless --ea or a stretch law is given."""
    if 'k_contact' in r: return float(r['k_contact'])
    if r.get('ea'): EA = float(r['ea'])
    elif r.get('stretch_law'):
        s, cn = r['stretch_law'].split(',')[0].split(':'); EA = 0.01 * float(cn) / float(s)
    else: EA = float(r['E']) * math.pi * R * R
    return float(r['kc_ratio']) * EA / R

def build_cmd(spec, name, rid, b, out):
    return [sys.executable, '-u', os.path.join(SCRIPTS, 'yarn_model', 'build_knit.py'), '--matrix', fabric_matrix(spec), '--tiles', str(int(b['tiles'])), '--ne', repr(float(b['ne'])), '--cpi', repr(float(b['cpi'])),
            '--wpi', repr(float(b['wpi'])), '--out', out, '--name', '%s-%s' % (name, rid), '--id', '%s-%s' % (name, rid), '--no-tube']

def relax_cmd(r, R, folder):
    cmd = [sys.executable, '-u', os.path.join(SCRIPTS, 'yarn_model', 'relax_job.py'), '--folder', folder, '--k-contact', repr(k_contact(r, R))]
    for k in sorted(r):
        if k in ('kc_ratio', 'k_contact'): continue
        v = r[k]
        if isinstance(v, bool):
            if v: cmd.append(flag_of(k))
        else: cmd += [flag_of(k), repr(v) if isinstance(v, float) else str(v)]
    return cmd


# ------------------------------------------------------------------ plan / status
def environment():
    import importlib.metadata as im
    def ver(p):
        try: return im.version(p)
        except Exception: return ''
    commit = dirty = ''
    try:
        commit = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--', 'scripts'], capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception: pass
    return dict(hostname=socket.gethostname(), python=sys.version.split()[0], jax=ver('jax'), jaxlib=ver('jaxlib'), scipy=ver('scipy'), numpy=ver('numpy'), git_commit=commit, scripts_uncommitted_changes=dirty)

def cmd_plan(a):
    name = a.name; spec = load_spec(name); mp = os.path.join(cdir(name), 'manifest.json')
    if os.path.exists(mp): sys.exit('manifest.json already exists for %s; plan never overwrites it (make a new campaign, or move the old folder aside yourself)' % name)
    pts = design_points(spec)
    runs = [dict(run='%04d' % (i + 1), params=p) for i, p in enumerate(pts)]
    wjson(mp, dict(name=name, created=now(), spec_sha1=spec_hash(name), n_runs=len(runs), runs=runs))
    for r in runs:
        os.makedirs(rdir(name, r['run']), exist_ok=True)
        wjson(os.path.join(rdir(name, r['run']), 'status.json'), dict(run=r['run'], status='queued', params=r['params'], stage='', attempts=[], interrupted_count=0, sim_index_start=0,
                                                                    exit_code=None, wall_s=None, reason='', history=[dict(t=now(), event='queued')]))
    say('planned %d runs in %s' % (len(runs), mp))
    for r in runs: print('  %s  %s' % (r['run'], json.dumps(r['params'])))

def load_status(name, rid): return rjson(os.path.join(rdir(name, rid), 'status.json'))

def save_status(name, st): wjson(os.path.join(rdir(name, st['run']), 'status.json'), st)

def event(st, ev, **kw): st['history'].append(dict(t=now(), event=ev, **kw))

def load_manifest(name):
    m = rjson(os.path.join(cdir(name), 'manifest.json'))
    if m is None: sys.exit('no manifest.json for %s: run  campaign.py plan %s  first' % (name, name))
    return m

def fmt(x):
    if isinstance(x, float): return '' if not math.isfinite(x) else repr(x)
    return '' if x is None else str(x)

def write_results(name):
    m = load_manifest(name); pcols = []
    for r in m['runs']:
        for k in r['params']:
            if k not in pcols: pcols.append(k)
    head = ['run', 'status', 'attempts', 'interrupted'] + pcols + METRIC_COLS; rows = []
    for r in m['runs']:
        st = load_status(name, r['run']) or {}
        met = rjson(os.path.join(rdir(name, r['run']), 'metrics.json'), {})
        row = [r['run'], st.get('status', ''), len(st.get('attempts', [])), st.get('interrupted_count', 0)] + [fmt(r['params'].get(k)) for k in pcols] + [fmt(met.get(k, '')) for k in METRIC_COLS]
        rows.append(row)
    p = os.path.join(cdir(name), 'results.csv'); tmp = '%s.tmp.%d' % (p, os.getpid())
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    return head, rows

def cmd_status(a):
    name = a.name; m = load_manifest(name); counts = {}
    for r in m['runs']:
        s = (load_status(name, r['run']) or {}).get('status', '?'); counts[s] = counts.get(s, 0) + 1
    print('campaign %s: %d runs  %s' % (name, len(m['runs']), '  '.join('%s %d' % kv for kv in sorted(counts.items()))))
    print('STOP requested' if os.path.exists(os.path.join(cdir(name), 'STOP')) else 'no STOP file')
    head, rows = write_results(name)
    show = ['run', 'status', 'attempts', 'interrupted'] + [h for h in head[4:] if h not in METRIC_COLS] + ['exit_code', 'relax_state', 'rounds_used', 'E_final', 'rel_dE_last', 'min_dist_over_2R_end', 'fold_direction', 'wall_s']
    idx = [head.index(h) for h in show]; tab = [show] + [[str(r[i]) if len(str(r[i])) < 12 else str(r[i])[:12] for i in idx] for r in rows]
    w = [max(len(t[j]) for t in tab) for j in range(len(show))]
    for i, t in enumerate(tab):
        print('  '.join(c.ljust(w[j]) for j, c in enumerate(t)))
        if i == 0: print('  '.join('-' * x for x in w))

def cmd_stop(a):
    p = os.path.join(cdir(a.name), 'STOP')
    if not os.path.isdir(cdir(a.name)): sys.exit('no such campaign: %s' % a.name)
    with open(p, 'w') as f: f.write(now() + '\n')
    say('STOP file created: %s  (the runner finishes the run in progress, then exits; nothing is killed)' % p)


# ------------------------------------------------------------------ metrics
def last_line(path, n=1):
    try:
        lines = [l.strip() for l in open(path, errors='replace') if l.strip()]
        return ' | '.join(lines[-n:])
    except OSError: return ''

def fold_metrics(axis_csv, meta, th):
    """Fold metrics of a relaxed fabric (relaxed_axis.csv). Curvatures from fold_metric.fit (global quadratic fit z = a x^2 + b y^2 + c xy + d x + e y + f). Amplitude and wavelength are computed here, on
    the vertex heights averaged in a grid of one bin per stitch (columns x rows). Cells that cannot be computed stay empty."""
    out = {}
    try:
        import numpy as np
        sys.path.insert(0, os.path.join(SCRIPTS, 'fold_analysis'))
        from fold_metric import fit
        P = np.loadtxt(axis_csv, delimiter=',', skiprows=1)[:, 1:]
    except Exception: return out
    if P.ndim != 2 or len(P) < 20 or not np.all(np.isfinite(P)): return out
    r = fit(P); kx, ky = float(r['kx']), float(r['ky']); Lx, Ly = float(r['Lx']), float(r['Ly'])
    out.update(fold_kx_per_mm=kx, fold_ky_per_mm=ky, fold_turn_x_rad=abs(kx) * Lx, fold_turn_y_rad=abs(ky) * Ly)
    tx, ty = abs(kx) * Lx >= th['fold_min_turn_rad'], abs(ky) * Ly >= th['fold_min_turn_rad']
    out['fold_direction'] = 'both' if tx and ty else 'course' if tx else 'wale' if ty else 'none'        # 'course' = curved along the courses (x), 'wale' = along the wales (y)
    mat, tiles = meta.get('matrix'), meta.get('tiles')
    if not mat or not tiles or Lx <= 0 or Ly <= 0: return out
    nx, ny = len(mat[0]) * tiles, len(mat) * tiles
    x, y, z = P[:, 0], P[:, 1], P[:, 2]; xc, yc = x - x.mean(), y - y.mean()
    def residual(basis):
        A = np.c_[tuple(basis)]; c = np.linalg.lstsq(A, z, rcond=None)[0]; return z - A @ c
    one = np.ones_like(xc)
    res_plane = residual([xc, yc, one]); res_quad = residual([xc * xc, yc * yc, xc * yc, xc, yc, one])           # same basis as fold_metric.fit
    xe = np.linspace(x.min(), x.max(), nx + 1); ye = np.linspace(y.min(), y.max(), ny + 1)
    cnt, _, _ = np.histogram2d(y, x, bins=[ye, xe])
    if (cnt == 0).mean() > 0.10: return out                                                    # too many empty cells: no amplitude or wavelength rather than a guess
    def grid(v):
        s, _, _ = np.histogram2d(y, x, bins=[ye, xe], weights=v); g = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan); return g
    gp, gq = grid(res_plane), grid(res_quad)
    out['fold_amplitude_mm'] = float(np.nanmax(gp) - np.nanmin(gp))                              # relief left after removing a plane (curl and pleats)
    out['fold_local_amplitude_mm'] = float(np.nanmax(gq) - np.nanmin(gq))                        # relief left after removing the global quadratic (pleats only)
    g = np.nan_to_num(gq - np.nanmean(gq), nan=0.0)
    P2 = np.abs(np.fft.fft2(g)) ** 2; tot = P2.sum()
    if tot > 1e-12:
        for ax, n, L, wl, fr in ((0, nx, Lx, 'wavelength_course_mm', 'wl_course_power_fraction'), (1, ny, Ly, 'wavelength_wale_mm', 'wl_wale_power_fraction')):
            pw = P2.sum(axis=ax)[1:n // 2 + 1]                                                    # power by spatial frequency k = 1 .. n/2 along x (ax 0) or y (ax 1); k = 0 excluded
            if len(pw) == 0 or pw.sum() <= 0: continue
            k = int(np.argmax(pw)) + 1; frac = float(pw[k - 1] / pw.sum()); out[fr] = frac
            if frac >= th['wavelength_min_power_fraction']: out[wl] = L / k
    return out

def collect_metrics(name, rid, spec, st):
    rd = rdir(name, rid); m = {}
    met = rjson(os.path.join(rd, 'meta.json'), {}) or {}
    sims = (met.get('simulations') or [])[st.get('sim_index_start', 0):]
    rounds = {}
    for s in sims:
        for r in s.get('rounds') or []: rounds[r['round']] = r
    if rounds:
        ks = sorted(rounds); last = rounds[ks[-1]]
        m.update(rounds_used=ks[-1], E_final=last['E'], E_stretch=last.get('stretch', ''), E_bend=last.get('bend', ''), E_contact=last.get('contact', ''))
        if len(ks) >= 2 and last['E']: m['rel_dE_last'] = (last['E'] - rounds[ks[-2]]['E']) / abs(last['E'])
        if 'min_dist_over_2R' in last: m['min_dist_over_2R_end'] = last['min_dist_over_2R']
    if sims:
        s0, s1 = sims[0], sims[-1]
        if s0.get('start'):
            m['E_start'] = s0['start'].get('total', ''); m['min_dist_over_2R_start'] = s0['start'].get('min_dist_over_2R', '')
        if s1.get('end'): m['min_dist_over_2R_end'] = s1['end'].get('min_dist_over_2R', m.get('min_dist_over_2R_end', ''))
        m['relax_state'] = s1.get('status', ''); m['outer_max'] = (s1.get('params') or {}).get('outer', '')
    sj = rjson(os.path.join(rd, 'state.json'), {}) or {}; summ = sj.get('summary') or {}
    if summ:
        m['settled_by'] = summ.get('settled_by', ''); dims = summ.get('dims') or {}
        m.update(shrink_x=dims.get('shrink_x', ''), shrink_y=dims.get('shrink_y', ''), z_range_over_R_end=dims.get('z_range_over_R_end', ''))
        wl = [abs(v) for v in (summ.get('wale_length_change') or []) if isinstance(v, (int, float))]
        if wl: m['rest_len_change_max_abs_wale'] = max(wl)
    try:
        for l in open(os.path.join(rd, 'relax.log'), errors='replace'):
            g = re.search(r'rest-length change per wale: .* total ([-+0-9.eE]+)\s*$', l)
            if g: m['rest_len_change_total'] = float(g.group(1))
    except (OSError, ValueError): pass
    ax = os.path.join(rd, 'relaxed_axis.csv')
    if os.path.exists(ax):
        try: m['vertices'] = sum(1 for _ in open(ax)) - 1
        except OSError: pass
        m.update(fold_metrics(ax, met, spec['thresholds']))
    m['exit_code'] = st.get('exit_code', ''); m['wall_s'] = st.get('wall_s', ''); m['build_s'] = st.get('build_s', ''); m['reason'] = st.get('reason', '')
    return m


# ------------------------------------------------------------------ running one run
class Ctx:
    def __init__(self, name, spec, manifest, poll_s):
        self.name, self.spec, self.manifest, self.poll_s = name, spec, manifest, poll_s
        self.env = environment()
        self.stop_path = os.path.join(cdir(name), 'STOP')

    def stop_requested(self): return os.path.exists(self.stop_path)

def finalize(ctx, st, t_note=''):
    """Called when an attempt has ended (exit file present) or the run failed before launch: set the final status, write metrics.json and results.csv."""
    name, rid = ctx.name, st['run']
    atts = st['attempts']
    ok = st.get('exit_code') in ACCEPTED_EXIT
    st['status'] = 'done' if ok else 'failed'
    if not ok and not st.get('reason'): st['reason'] = last_line(os.path.join(rdir(name, rid), 'relax.log')) or 'no reason recorded'
    st['wall_s'] = round(sum((a.get('t_end', a['t_start']) - a['t_start']) for a in atts if a.get('t_start')), 1) if atts else None
    event(st, st['status'], exit_code=st.get('exit_code'), reason=st.get('reason', ''))
    save_status(name, st)
    wjson(os.path.join(rdir(name, rid), 'metrics.json'), collect_metrics(name, rid, ctx.spec, st))
    write_results(name)
    say('run %s %s (exit %s, %s s)%s' % (rid, st['status'], st.get('exit_code'), st['wall_s'], (' - ' + st['reason']) if st.get('reason') else ''))

def read_exit(rd, n):
    try:
        rc, t = open(os.path.join(rd, 'exit.%d' % n)).read().split()[:2]; return int(rc), float(t)
    except (OSError, ValueError): return None

def monitor(ctx, st, att):
    """Wait for the attempt's wrapper to finish. Returns 'finished' (exit file present), 'interrupted' (wrapper gone without an exit file) or 'killed' (timeout or RAM limit, by this runner)."""
    name, rid, rd = ctx.name, st['run'], rdir(ctx.name, st['run']); spec = ctx.spec; rss_gb = spec.get('max_rss_gb')
    done_before = sum((a.get('t_end', a['t_start']) - a['t_start']) for a in st['attempts'][:-1] if a.get('t_start'))
    while True:
        ex = read_exit(rd, att['n'])
        if ex: att['t_end'] = ex[1]; st['exit_code'] = ex[0]; return 'finished'
        if not alive(att): return 'interrupted'
        elapsed = done_before + (time.time() - att['t_start'])
        if elapsed > spec['timeout_s']:
            say('run %s exceeded the timeout of %s s: sending SIGTERM to its own process group %d' % (rid, spec['timeout_s'], att['pid']))
            st['reason'] = 'timeout after %.0f s (limit %s s)' % (elapsed, spec['timeout_s']); return kill_own(att, 'timeout')
        if rss_gb:
            rss = descendants_rss_kb(att['pid']) / 1048576.0
            if rss > rss_gb:
                say('run %s uses %.2f GB (limit %s GB): sending SIGTERM to its own process group %d' % (rid, rss, rss_gb, att['pid']))
                st['reason'] = 'RAM %.2f GB over the limit of %s GB' % (rss, rss_gb); return kill_own(att, 'ram')
        time.sleep(2)

def kill_own(att, why):
    """SIGTERM to the process group of this run's own wrapper (setsid made it a group of its own), then wait up to 60 s for it to end; SIGKILL only if it ignores SIGTERM."""
    try: os.killpg(att['pid'], signal.SIGTERM)
    except OSError: pass
    for _ in range(60):
        if not alive(att): break
        time.sleep(1)
    else:
        try: os.killpg(att['pid'], signal.SIGKILL)
        except OSError: pass
    att['t_end'] = time.time(); att['killed'] = why; return 'killed'

def launch(ctx, st, build, relax, R, resume):
    name, rid = ctx.name, st['run']; rd = rdir(name, rid)
    cmd = relax_cmd(relax, R, rd) + (['--resume'] if resume else [])
    n = len(st['attempts']) + 1
    log = open(os.path.join(rd, 'relax.log'), 'a'); log.write('\n=== attempt %d at %s, resume=%s ===\n%s\n' % (n, now(), resume, ' '.join(cmd))); log.flush()
    wrapper = 'D="$1"; N="$2"; shift 2; "$@" >> "$D/relax.log" 2>&1; rc=$?; printf "%s %s\\n" "$rc" "$(date +%s)" > "$D/exit.$N.tmp"; mv "$D/exit.$N.tmp" "$D/exit.$N"'
    p = subprocess.Popen(['/bin/sh', '-c', wrapper, 'sh', rd, str(n)] + cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=SCRIPTS, start_new_session=True, close_fds=True)
    pi = proc_info(p.pid)
    att = dict(n=n, pid=p.pid, boot_id=boot_id(), start_ticks=pi[0] if pi else '', t_start=time.time(), started=now(), resume=bool(resume), cmd=cmd)
    st['attempts'].append(att); st['stage'] = 'relax'; save_status(name, st)
    return att, p

def do_build(ctx, st, build):
    name, rid = ctx.name, st['run']; rd = rdir(name, rid)
    mp = os.path.join(rd, 'meta.json')
    if (rjson(mp) or {}).get('structure'): return True                                   # complete build from an earlier attempt (meta.json is written last, atomically)
    st['stage'] = 'build'; event(st, 'build'); save_status(name, st)
    t0 = time.time(); bl = open(os.path.join(rd, 'build.log'), 'a'); cmd = build_cmd(ctx.spec, name, rid, build, rd)
    bl.write('=== build at %s ===\n%s\n' % (now(), ' '.join(cmd))); bl.flush()
    try: rc = subprocess.run(cmd, stdout=bl, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=SCRIPTS, timeout=1800).returncode
    except subprocess.TimeoutExpired: rc = -1
    st['build_s'] = round(time.time() - t0, 1)
    if rc != 0 or not (rjson(mp) or {}).get('structure'):
        st['exit_code'] = rc; st['reason'] = 'build failed (exit %s): %s' % (rc, last_line(os.path.join(rd, 'build.log'), 2)); st['attempts'].append(dict(n=len(st['attempts']) + 1, stage='build', t_start=t0, t_end=time.time()))
        return False
    return True

def run_one(ctx, rid, adopt=False):
    name = ctx.name; st = load_status(name, rid); rd = rdir(name, rid)
    run = next(r for r in ctx.manifest['runs'] if r['run'] == rid)
    build, relax = merged(ctx.spec, run['params'])
    if not adopt:
        st['status'] = 'running'; st['env'] = ctx.env; event(st, 'start'); save_status(name, st)
        if not do_build(ctx, st, build):
            st['exit_code'] = st.get('exit_code', -1); return finalize(ctx, st)
        meta = rjson(os.path.join(rd, 'meta.json')) or {}; R = meta.get('R') or yarn_R(build['ne'])
        ck = os.path.exists(os.path.join(rd, 'checkpoint.npz'))
        if not ck: st['sim_index_start'] = len(meta.get('simulations') or [])       # a from-scratch attempt: earlier partial entries in meta.json are not part of this run's result
        att, _p = launch(ctx, st, build, relax, R, resume=ck)
        say('run %s started (attempt %d, pid %d, %s) params %s' % (rid, att['n'], att['pid'], 'resumed from checkpoint' if ck else 'from scratch', json.dumps(run['params'])))
        proc = _p
    else:
        att = st['attempts'][-1]; proc = None
        say('run %s: adopting the job still running from before (pid %d)' % (rid, att['pid']))
    res = monitor(ctx, st, att)
    if proc is not None:
        try: proc.wait(timeout=5)
        except Exception: pass
    if res == 'finished': return finalize(ctx, st)
    if res == 'killed':
        st['exit_code'] = st.get('exit_code') if st.get('exit_code') is not None else -15
        return finalize(ctx, st)
    return mark_interrupted(ctx, st, 'wrapper process disappeared without an exit file (power cut or external kill)')

def mark_interrupted(ctx, st, why):
    st['interrupted_count'] = st.get('interrupted_count', 0) + 1
    if st['attempts']: st['attempts'][-1]['interrupted'] = now()
    event(st, 'interrupted', why=why)
    if st['interrupted_count'] > ctx.spec.get('max_interrupts', 3):
        st['status'] = 'failed'; st['reason'] = 'interrupted %d times' % st['interrupted_count']; st['exit_code'] = -1; event(st, 'failed', reason=st['reason'])
        save_status(ctx.name, st); write_results(ctx.name); say('run %s failed: %s' % (st['run'], st['reason'])); return
    st['status'] = 'queued'; st['stage'] = ''; event(st, 'requeued', checkpoint=os.path.exists(os.path.join(rdir(ctx.name, st['run']), 'checkpoint.npz')))
    save_status(ctx.name, st); write_results(ctx.name)
    say('run %s was interrupted (%s): requeued, %s' % (st['run'], why, 'will resume from checkpoint.npz' if os.path.exists(os.path.join(rdir(ctx.name, st['run']), 'checkpoint.npz')) else 'no checkpoint yet, will restart from scratch'))

def reconcile(ctx):
    """On runner start: look at runs marked running. Returns the ids whose job is still alive (to be adopted); finished ones are finalized, vanished ones are marked interrupted and requeued."""
    adopt = []
    for r in ctx.manifest['runs']:
        st = load_status(ctx.name, r['run'])
        if not st or st['status'] != 'running': continue
        att = st['attempts'][-1] if st['attempts'] else None
        ex = read_exit(rdir(ctx.name, r['run']), att['n']) if att and 'n' in att and att.get('pid') else None
        if ex: att['t_end'] = ex[1]; st['exit_code'] = ex[0]; finalize(ctx, st)
        elif att and alive(att): adopt.append(r['run'])
        else: mark_interrupted(ctx, st, 'runner started and found the run marked running without a live process (power cut or crash)')
    return adopt

def wait_until_free(ctx):
    """True when the machine is free to start a run; False if STOP was requested while waiting."""
    noted = False
    while True:
        if ctx.stop_requested(): return False
        why = busy_reason()
        if not why: return True
        if not noted:
            stale = [s for s in stale_stage_files() if not any(os.path.dirname(s) in ' '.join(a) for _, a in cmdlines())]
            if stale: say('note: %d job_stage.json file(s) under webui/results have no live process (stale after a power cut?); ignored' % len(stale))
            noted = True
        say('waiting: another job is running (%s); next check in %d s' % (why, ctx.poll_s))
        for _ in range(int(ctx.poll_s)):
            if ctx.stop_requested(): return False
            time.sleep(1)

def cmd_run(a):
    name = a.name; spec = load_spec(name); manifest = load_manifest(name)
    if manifest.get('spec_sha1') != spec_hash(name) and not a.ignore_spec_change: sys.exit('spec.json changed after the plan was made (hash differs). Refusing to run; use --ignore-spec-change if the change was harmless.')
    if a.dry_run:
        n = 0
        for r in manifest['runs']:
            st = load_status(name, r['run']) or {}
            if st.get('status', 'queued') != 'queued': continue
            b, rl = merged(spec, r['params']); R = yarn_R(b['ne']); n += 1
            print('run %s  params %s' % (r['run'], json.dumps(r['params'])))
            print('   build: ' + ' '.join(build_cmd(spec, name, r['run'], b, rdir(name, r['run']))))
            print('   relax: ' + ' '.join(relax_cmd(rl, R, rdir(name, r['run']))))
        print('%d run(s) would be executed, one after another; timeout %s s each; nothing was started.' % (n, spec['timeout_s'])); return
    if a.detach:
        os.makedirs(LOG_DIR, exist_ok=True); lp = os.path.join(LOG_DIR, 'campaign_%s.log' % name)
        cmd = [sys.executable, os.path.abspath(__file__), 'run', name, '--poll-s', str(a.poll_s)] + (['--ignore-spec-change'] if a.ignore_spec_change else [])
        p = subprocess.Popen(cmd, stdout=open(lp, 'a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
        print('runner started (own session, SIGHUP ignored): pid %d, log %s' % (p.pid, lp)); return
    import fcntl
    lockp = os.path.join(cdir(name), 'runner.lock'); lock = open(lockp, 'a+')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: sys.exit('another runner already holds %s (flock); only one runner per campaign' % lockp)
    pi = proc_info(os.getpid()); lock.seek(0); lock.truncate(); lock.write(json.dumps(dict(pid=os.getpid(), boot_id=boot_id(), start_ticks=pi[0] if pi else '', started=now(), host=socket.gethostname()))); lock.flush()
    if os.path.exists(os.path.join(cdir(name), 'STOP')):
        os.replace(os.path.join(cdir(name), 'STOP'), os.path.join(cdir(name), 'STOP.stale.%s' % time.strftime('%Y%m%d-%H%M%S'))); say('an old STOP file was found and moved aside (renamed STOP.stale.*)')
    ctx = Ctx(name, spec, manifest, a.poll_s)
    say('runner started for %s (pid %d, %s)' % (name, os.getpid(), json.dumps(ctx.env)))
    for rid in reconcile(ctx): run_one(ctx, rid, adopt=True)
    while True:
        if ctx.stop_requested(): break
        nxt = next((r['run'] for r in ctx.manifest['runs'] if (load_status(name, r['run']) or {}).get('status') == 'queued'), None)
        if nxt is None: say('no queued runs left'); break
        if not wait_until_free(ctx): break
        run_one(ctx, nxt)
    if ctx.stop_requested():
        os.replace(ctx.stop_path, os.path.join(cdir(name), 'STOP.honored.%s' % time.strftime('%Y%m%d-%H%M%S'))); say('STOP honoured between runs; runner exits (STOP file renamed STOP.honored.*)')
    write_results(name); say('runner finished')

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('shell-plan', 'shell-enqueue', 'shell-run', 'shell-status', 'shell-control', 'shell-freeze', 'shell-approve'):          # pure shell campaigns: scripts/campaigns/shell_campaign.py
        import shell_campaign; return shell_campaign.cli(sys.argv[1], sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('plan', 'status', 'stop', 'run'): sp = sub.add_parser(c); sp.add_argument('name')
    r = sub.choices['run']; r.add_argument('--dry-run', action='store_true'); r.add_argument('--detach', action='store_true'); r.add_argument('--poll-s', type=int, default=60); r.add_argument('--ignore-spec-change', action='store_true')
    a = ap.parse_args()
    {'plan': cmd_plan, 'status': cmd_status, 'stop': cmd_stop, 'run': cmd_run}[a.cmd](a)

if __name__ == '__main__':
    main()
