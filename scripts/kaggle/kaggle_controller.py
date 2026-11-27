#!/usr/bin/env python3
"""kaggle_controller.py - runs ON THE SERVER: keeps Kaggle sessions working on a pure-shell campaign through the shared queue with leases (shell_campaign.Camp.claim / import_result), 2026-11-27.

    python3 scripts/kaggle/kaggle_controller.py <campaign> [--once] [--poll-s 300]

Config  <campaign>/kaggle/config.json  (written by the web page; defaults below): enabled, slots [{name, gpu}], budget_s per session (CPU / GPU), est_s_per_stitch (CPU / GPU).
State   <campaign>/kaggle/state.json   one entry per slot: idle | running (kernel slug, version time, batch ids).
Loop (every poll): for every slot
  running  -> ask Kaggle for the kernel status; when it is complete / error / cancelled (or older than budget + 3 h): download the output, import every run its manifest lists as done (checked by
              import_result: settings, frozen-code provenance), mark failed runs in the index, release every other lease of the batch (those runs go back to the queue).
  idle     -> if enabled and the campaign's control is not 'stop' and runs are queued: claim a batch that fits the session budget (estimated time from the stitch count), push the worker kernel.
Network outages: every Kaggle call may fail; nothing is changed then and the same step is tried at the next poll. A lease expires by itself (budget + 3 h), so a lost session never blocks runs for good.
Power cut on the server: the state file is written atomically; on restart a 'running' slot is polled again and its output downloaded. Credentials: kaggle_cli (.env, never printed).
The frozen code is uploaded once per campaign as the private dataset <user>/auxetic-<campaign>-code-<8 hex of the manifest>; the worker runs exactly those files (provenance is checked on import)."""
import argparse, json, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kaggle_cli as kc
sys.path.insert(0, os.path.join(kc.ROOT, 'scripts', 'campaigns'))          # shell_campaign.py and campaign.py (scriptdirs.py is a tests/ helper, not on the server path: found at the first real start)
import shell_campaign as sc, campaign as cm          # noqa: E402

DEFAULT = dict(enabled=False, slots=[dict(name='k1', gpu=True), dict(name='k2', gpu=False)], budget_s=dict(cpu=10.5 * 3600, gpu=8.0 * 3600), est_s_per_stitch=dict(cpu=0.12, gpu=0.12),
               lease_margin_s=3 * 3600, max_batch=dict(gpu=300, cpu=80),          # GPU (2 x T4) measured 2026-11-27: 80 runs in 67 min (median 103 s per lane)
               notes='UNSOURCED operating choices (2026-11-27): Kaggle session limits taken as 12 h (CPU) and 9 h (GPU) from Kaggle documentation, NOT verified here, so the budgets leave 1.5 h / 1 h spare; '
                     'est_s_per_stitch from the server benchmark (774 s for 3360 stitches = 0.23 s) halved for Kaggle CPU (the probe ran 2x faster), refined from finished runs')

def say(*a): print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)

class Ctl:
    def __init__(s, name):
        s.c = sc.Camp(name); s.name = name
        if not s.c.exists(): raise SystemExit('no such shell campaign: %s' % name)
        s.d = os.path.join(s.c.d, 'kaggle'); os.makedirs(s.d, exist_ok=True); s.user = kc.username()
        man = s.c.manifest() or {}; h = __import__('hashlib').sha1(json.dumps(man.get('files') or {}, sort_keys=True).encode()).hexdigest()[:8]
        s.ds = ('auxetic-%s-code-%s' % (name, h)).lower()          # one dataset per frozen code: a re-created campaign never runs older files (provenance would refuse every import)
    def config(s):
        cf = dict(DEFAULT); cf.update(cm.rjson(os.path.join(s.d, 'config.json'), {}) or {}); return cf
    def state(s): return cm.rjson(os.path.join(s.d, 'state.json'), {}) or {}
    def save(s, st): cm.wjson(os.path.join(s.d, 'state.json'), st)
    def log(s, **kw): sc.append_line(os.path.join(s.d, 'events.jsonl'), dict(t=cm.now(), **kw))
    def worker(s, slot): return 'kaggle:' + slot

    def ensure_dataset(s):
        rc, o = kc.run(['datasets', 'status', '%s/%s' % (s.user, s.ds)])
        if rc == 0 and 'ready' in o.lower(): return True
        dd = os.path.join(s.d, 'dataset'); os.makedirs(dd, exist_ok=True)
        man = s.c.manifest() or {}
        for f in list((man.get('files') or {}).keys()) + ['MANIFEST.json']: shutil.copy2(os.path.join(s.c.code_dir(), f), os.path.join(dd, f))          # only the frozen files (the folder may also hold __pycache__)
        json.dump(dict(title=s.ds, id='%s/%s' % (s.user, s.ds), licenses=[dict(name='CC0-1.0')]), open(os.path.join(dd, 'dataset-metadata.json'), 'w'))
        rc, o = kc.run(['datasets', 'create', '-p', dd, '-q'], timeout=900); say('dataset create:', rc, o.strip()[-200:]); return False          # ready at a later poll

    def per_stitch(s, gpu, cf):
        """Median seconds per stitch of this campaign's finished Kaggle runs on that hardware (at least 3), else the configured value."""
        q = {x['id']: x['settings'] for x in s.c.queue_items()}; v = []
        for rid, x in s.c.index_last().items():
            if x.get('status') == 'done' and str(x.get('worker', '')).startswith('kaggle') and isinstance(x.get('wall_s'), (int, float)) and rid in q and (str(x.get('device', 'cpu')) != 'cpu') == gpu:
                st = q[rid]; v.append(x['wall_s'] / (len(st['matrix']) * len(st['matrix'][0]) * st['tiles'] ** 2))
        return sorted(v)[len(v) // 2] if len(v) >= 3 else cf['est_s_per_stitch']['gpu' if gpu else 'cpu']

    def estimate(s, item, gpu, cf, ps=None):
        st = item['settings']; n = len(st['matrix']) * len(st['matrix'][0]) * st['tiles'] ** 2; return n * (ps if ps is not None else s.per_stitch(gpu, cf))

    def start(s, slot, cf):
        gpu = bool(slot.get('gpu')); budget = cf['budget_s']['gpu' if gpu else 'cpu']; lanes = 2 if gpu else 1; pend = s.c.pending()
        if not pend: return None
        chosen, tot = [], 0.0; ps = s.per_stitch(gpu, cf)
        mb = cf['max_batch']; mb = mb.get('gpu' if gpu else 'cpu', 80) if isinstance(mb, dict) else mb
        for it in pend[:mb]:
            e = s.estimate(it, gpu, cf, ps)
            if tot + e > budget * lanes * 0.9 and chosen: break
            chosen.append(it['id']); tot += e
        got = s.c.claim(          # claim takes from the head of the queue: the chosen runs are that head (a run started meanwhile by another worker is simply not included)
s.worker(slot['name']), len(chosen), budget + cf['lease_margin_s'], 'kaggle %s' % slot['name'])
        if not got: return None
        thr = (s.c.spec_().get('thresholds') or {}).get('fold_min_turn_rad', 0.2)
        batch = [dict(id=x['id'], settings=dict(x['settings'], fold_min_turn_rad=thr), est_s=round(s.estimate(x, gpu, cf, ps))) for x in got]
        slug = 'auxetic-%s-%s-%s' % (s.name, slot['name'], time.strftime('%m%d%H%M%S') + '%03d' % (int(time.time() * 1000) % 1000)); kd = os.path.join(s.d, 'kernels', slug); os.makedirs(kd, exist_ok=True)          # one kernel per session: a later push never replaces output not yet downloaded (2026-11-27)
        tpl = open(os.path.join(HERE, 'worker_kernel.py')).read().replace('__BATCH__', 'json.loads(%r)' % json.dumps(batch)).replace('__BUDGET_S__', repr(float(budget))).replace('__CAMPAIGN__', json.dumps(s.name))          # BATCH as a JSON string: JSON text is not Python (true, null; bug found by the test)
        open(os.path.join(kd, 'worker.py'), 'w').write(tpl)
        json.dump(dict(id='%s/%s' % (s.user, slug), title=slug, code_file='worker.py', language='python', kernel_type='script', is_private=True, enable_gpu=gpu, enable_internet=False,
                       dataset_sources=['%s/%s' % (s.user, s.ds)], competition_sources=[], kernel_sources=[]), open(os.path.join(kd, 'kernel-metadata.json'), 'w'))
        rc, o = kc.run(['kernels', 'push', '-p', kd], timeout=600)
        if rc != 0 or 'successfully' not in o.lower():
            s.c.release([x['id'] for x in got], s.worker(slot['name']), 'push failed'); say('push failed for', slot['name'], o.strip()[-300:]); s.log(event='push_failed', slot=slot['name'], out=o.strip()[-300:]); return None
        ent = dict(status='running', slot=slot['name'], slug=slug, gpu=gpu, ids=[x['id'] for x in got], t_push=time.time(), t_push_iso=cm.now(), budget_s=budget)
        say('pushed %s: %d runs (est %.1f h on %d lane(s))' % (slug, len(got), tot / 3600, lanes)); s.log(event='pushed', slot=slot['name'], ids=ent['ids'], gpu=gpu); return ent

    def renew(s, name, ent, cf):
        """Extend the leases of a session that Kaggle still reports running or queued (to now + lease margin), so the server never takes a run that is still being worked on."""
        with sc.xlock(os.path.join(s.c.d, 'queue.lock')):
            for rid in ent['ids']:
                p = os.path.join(s.c.leases, rid + '.json'); x = cm.rjson(p)
                if isinstance(x, dict) and x.get('state') == 'held' and x.get('worker') == s.worker(name) and float(x.get('expires', 0)) < time.time() + cf['lease_margin_s'] / 2:
                    x['expires'] = time.time() + cf['lease_margin_s']; x['renewed'] = cm.now(); cm.wjson(p, x)

    def finish(s, name, ent, cf, why, timeout=3600):
        od = os.path.join(s.d, 'downloads', '%s_%d' % (name, int(ent['t_push']))); os.makedirs(od, exist_ok=True)
        if os.path.isfile(os.path.join(od, 'COMPLETE')): rc, o = 0, 'downloaded on the workstation (fetch_on_workstation.py)'          # the server's own internet is slow / often down (2026-11-27)
        else: rc, o = kc.run(['kernels', 'output', '%s/%s' % (s.user, ent['slug']), '-p', od], timeout=timeout)          # no -o: files already here are skipped, so a retry resumes (one file at a time is slow: 31 runs in 20 min, 2026-11-27)
        tarp = os.path.join(od, 'out.tar')
        if rc == 0 and os.path.isfile(tarp) and not os.path.isdir(os.path.join(od, 'out')):          # workers since 2026-11-27 pack the run folders into one file
            import tarfile
            with tarfile.open(tarp) as tf:
                try: tf.extractall(od, filter='data')          # refuses links and paths outside od
                except TypeError: tf.extractall(od)          # Python before 3.12
        if rc != 0:
            say('download failed for', name, o.strip()[-200:])
            if why != 'too old': return False          # try again next poll; an overdue session is closed without output (its runs go back to the queue; review 2026-11-27)
        man = cm.rjson(os.path.join(od, 'manifest.json'), {}) or {}; runs = man.get('runs') or {}; nd = nf = 0
        for rid in ent['ids']:
            r = runs.get(rid) or {}; src = os.path.join(od, 'out', rid)
            if r.get('status') == 'done':
                try: res = s.c.import_result(rid, src, s.worker(name), wall_s=r.get('wall_s'), extra=dict(device=r.get('device'), kaggle_slug=ent['slug'])); nd += res == 'done'
                except ValueError as e:
                    nf += 1; sc.append_line(s.c.index, dict(id=rid, status='failed', t=cm.now(), reason='kaggle import refused: %s' % e, worker=s.worker(name))); say('import refused', rid, e)
            elif r.get('status') == 'failed' and r.get('rc') != 'timeout':          # a timeout on Kaggle (slower hardware than estimated) is not a failure of the run: it goes back to the queue (review 2026-11-27)
                nf += 1; tail = ''
                try: tail = open(os.path.join(src, 'shell.log'), errors='replace').read()[-300:]
                except OSError: pass
                sc.append_line(s.c.index, dict(id=rid, status='failed', t=cm.now(), reason='kaggle run failed (rc %s): %s' % (r.get('rc'), tail.strip().splitlines()[-1] if tail.strip() else ''), worker=s.worker(name), wall_s=r.get('wall_s')))
        back = s.c.release(ent['ids'], s.worker(name), 'session ended: ' + why); s.c.write_results()
        walls = [(r['wall_s'], rid) for rid, r in runs.items() if r.get('status') == 'done']
        say('%s finished (%s): %d imported, %d failed, %d back to the queue' % (name, why, nd, nf, len(back))); s.log(event='finished', slot=name, why=why, imported=nd, failed=nf, released=back, walls=walls[:200])
        return True

    def tick(s):
        cf = s.config(); st = s.state(); slots = {x['name']: x for x in cf['slots']}
        if not s.user: say('no KAGGLE_USERNAME in .env'); return
        ds_ok = (not cf.get('enabled')) or s.ensure_dataset()          # a Kaggle outage must not stop finishing and importing sessions (bug found 2026-11-27): only new pushes wait
        aw = st.setdefault('_awaiting', {})          # finished on Kaggle, output not downloaded yet (server internet down; the workstation fetches it): leases renewed, download retried every 30 min
        for slug, ent in list(aw.items()):
            name = ent.get('slot') or slug.split('-')[2]
            if os.path.isfile(os.path.join(s.d, 'downloads', '%s_%d' % (name, int(ent['t_push'])), 'COMPLETE')) or time.time() - ent.get('t_try', 0) > 1800:
                ent['t_try'] = time.time(); s.save(st)
                if s.finish(name, ent, cf, 'complete (after waiting for the download)', timeout=900): aw.pop(slug); s.save(st); continue
            if time.time() - ent.get('t_done', time.time()) > cf.get('await_max_s', 4 * 86400):          # never downloaded within 4 days (UNSOURCED): give its runs back to the queue
                if s.finish(name, ent, cf, 'too old'): aw.pop(slug); s.save(st)
                continue
            s.renew(name, ent, cf)
        for name, slot in slots.items():
            ent = st.get(name) or dict(status='idle')
            if ent.get('status') == 'running':
                rc, o = kc.run(['kernels', 'status', '%s/%s' % (s.user, ent['slug'])]); low = o.lower(); age = time.time() - ent['t_push']
                alive = rc == 0 and any(w in low for w in ('running', 'queued'))
                if alive and age < cf.get('hard_limit_s', 13 * 3600):          # still working on Kaggle (it may have waited in Kaggle's queue): keep its leases from expiring (review 2026-11-27)
                    s.renew(name, ent, cf); continue
                old = age > (cf.get('hard_limit_s', 13 * 3600) if alive else ent['budget_s'] + cf['lease_margin_s'])
                if rc != 0 and not old: continue          # network trouble: ask again next time
                done = any(w in low for w in ('complete', 'error', 'cancel'))
                if 'complete' in low and not s.finish(name, ent, cf, 'complete', timeout=900):          # finished but the download failed: park it and free the slot for a new session
                    ent = dict(ent, slot=name, status='awaiting_download', t_done=time.time(), t_try=time.time()); aw[ent['slug']] = ent
                    say('%s complete on Kaggle; download pending (slot freed): %s' % (name, ent['slug'])); s.log(event='awaiting_download', slot=name, slug=ent['slug'], session_h=round((time.time() - ent['t_push']) / 3600, 2), gpu=ent.get('gpu'))
                elif 'complete' in low: s.log(event='session_hours', slot=name, slug=ent['slug'], session_h=round((time.time() - ent['t_push']) / 3600, 2), gpu=ent.get('gpu'))
                elif not (done or old) or not s.finish(name, ent, cf, 'error/cancel' if done else 'too old'): continue
                ent = dict(status='idle', last=ent['slug'], t_last=cm.now()); st[name] = ent; s.save(st)          # finished: the same tick may start the next batch
            if ds_ok and cf.get('enabled') and s.c.get_control() != 'stop' and not slot.get('retire'):          # retire: finish and import, no new session (K 2026-11-27: Kaggle GPU only)
                new = s.start(slot, cf)
                if new: st[name] = new; s.save(st)
        st['_tick'] = cm.now(); s.save(st)

def main():
    a = argparse.ArgumentParser(); a.add_argument('campaign'); a.add_argument('--once', action='store_true'); a.add_argument('--poll-s', type=float, default=300); g = a.parse_args()
    c = Ctl(g.campaign); say('kaggle controller for %s (user %s)' % (g.campaign, 'set' if c.user else 'MISSING'))
    if not g.once:          # one controller per campaign (flock, released by the kernel when the process dies, also after a power cut); controller.json tells the page that it is alive
        lk = cm.pc.try_lock(os.path.join(c.d, 'controller.lock'))          # Linux flock (as before) / Windows byte lock (2026-11-27)
        if lk is None: sys.exit('another kaggle controller already runs for %s' % g.campaign)
        pi = cm.proc_info(os.getpid()); info = dict(pid=os.getpid(), boot_id=cm.boot_id(), start_ticks=pi[0] if pi else '', started=cm.now(), poll_s=g.poll_s)
    watch = [os.path.abspath(__file__), os.path.join(HERE, 'kaggle_cli.py'), os.path.join(HERE, 'worker_kernel.py'), sc.__file__]
    mt = {f: os.path.getmtime(f) for f in watch if os.path.exists(f)}
    while True:
        if not g.once and any(os.path.exists(f) and os.path.getmtime(f) != t for f, t in mt.items()):          # a git pull changed the code: restart in place (the lock fd is close-on-exec, so it is released)
            say('code changed on disk: restarting the controller with the new code'); os.execv(sys.executable, [sys.executable, '-u', os.path.abspath(__file__)] + sys.argv[1:])
        if not g.once: cm.wjson(os.path.join(c.d, 'controller.json'), dict(info, t=cm.now(), t_epoch=time.time()))
        try: c.tick()
        except Exception as e: say('tick error (retried next poll):', repr(e))
        if g.once: break
        time.sleep(g.poll_s)

if __name__ == '__main__':
    main()
