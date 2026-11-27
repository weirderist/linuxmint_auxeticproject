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
sys.path.insert(0, os.path.join(kc.ROOT, 'scripts')); import scriptdirs          # noqa: E402
import shell_campaign as sc, campaign as cm          # noqa: E402

DEFAULT = dict(enabled=False, slots=[dict(name='k1', gpu=True), dict(name='k2', gpu=False)], budget_s=dict(cpu=10.5 * 3600, gpu=8.0 * 3600), est_s_per_stitch=dict(cpu=0.12, gpu=0.12),
               lease_margin_s=3 * 3600, max_batch=80,
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
        for f in os.listdir(s.c.code_dir()): shutil.copy2(os.path.join(s.c.code_dir(), f), os.path.join(dd, f))
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
        for it in pend[:cf['max_batch']]:
            e = s.estimate(it, gpu, cf, ps)
            if tot + e > budget * lanes * 0.9 and chosen: break
            chosen.append(it['id']); tot += e
        got = s.c.claim(          # claim takes from the head of the queue: the chosen runs are that head (a run started meanwhile by another worker is simply not included)
s.worker(slot['name']), len(chosen), budget + cf['lease_margin_s'], 'kaggle %s' % slot['name'])
        if not got: return None
        thr = (s.c.spec_().get('thresholds') or {}).get('fold_min_turn_rad', 0.2)
        batch = [dict(id=x['id'], settings=dict(x['settings'], fold_min_turn_rad=thr), est_s=round(s.estimate(x, gpu, cf, ps))) for x in got]
        slug = 'auxetic-%s-%s' % (s.name, slot['name']); kd = os.path.join(s.d, 'kernels', slug); os.makedirs(kd, exist_ok=True)
        tpl = open(os.path.join(HERE, 'worker_kernel.py')).read().replace('__BATCH__', 'json.loads(%r)' % json.dumps(batch)).replace('__BUDGET_S__', repr(float(budget))).replace('__CAMPAIGN__', json.dumps(s.name))          # BATCH as a JSON string: JSON text is not Python (true, null; bug found by the test)
        open(os.path.join(kd, 'worker.py'), 'w').write(tpl)
        json.dump(dict(id='%s/%s' % (s.user, slug), title=slug, code_file='worker.py', language='python', kernel_type='script', is_private=True, enable_gpu=gpu, enable_internet=False,
                       dataset_sources=['%s/%s' % (s.user, s.ds)], competition_sources=[], kernel_sources=[]), open(os.path.join(kd, 'kernel-metadata.json'), 'w'))
        rc, o = kc.run(['kernels', 'push', '-p', kd], timeout=600)
        if rc != 0 or 'successfully' not in o.lower():
            s.c.release([x['id'] for x in got], s.worker(slot['name']), 'push failed'); say('push failed for', slot['name'], o.strip()[-300:]); s.log(event='push_failed', slot=slot['name'], out=o.strip()[-300:]); return None
        ent = dict(status='running', slug=slug, gpu=gpu, ids=[x['id'] for x in got], t_push=time.time(), t_push_iso=cm.now(), budget_s=budget)
        say('pushed %s: %d runs (est %.1f h on %d lane(s))' % (slug, len(got), tot / 3600, lanes)); s.log(event='pushed', slot=slot['name'], ids=ent['ids'], gpu=gpu); return ent

    def finish(s, name, ent, cf, why):
        od = os.path.join(s.d, 'downloads', '%s_%d' % (name, int(ent['t_push']))); os.makedirs(od, exist_ok=True)
        rc, o = kc.run(['kernels', 'output', '%s/%s' % (s.user, ent['slug']), '-p', od, '-o'], timeout=1800)
        if rc != 0: say('download failed for', name, o.strip()[-200:]); return False          # try again next poll
        man = cm.rjson(os.path.join(od, 'manifest.json'), {}) or {}; runs = man.get('runs') or {}; nd = nf = 0
        for rid in ent['ids']:
            r = runs.get(rid) or {}; src = os.path.join(od, 'out', rid)
            if r.get('status') == 'done':
                try: res = s.c.import_result(rid, src, s.worker(name), wall_s=r.get('wall_s'), extra=dict(device=r.get('device'), kaggle_slug=ent['slug'])); nd += res == 'done'
                except ValueError as e:
                    nf += 1; sc.append_line(s.c.index, dict(id=rid, status='failed', t=cm.now(), reason='kaggle import refused: %s' % e, worker=s.worker(name))); say('import refused', rid, e)
            elif r.get('status') == 'failed':
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
        if cf.get('enabled') and not s.ensure_dataset(): return
        for name, slot in slots.items():
            ent = st.get(name) or dict(status='idle')
            if ent.get('status') == 'running':
                rc, o = kc.run(['kernels', 'status', '%s/%s' % (s.user, ent['slug'])]); low = o.lower(); old = time.time() - ent['t_push'] > ent['budget_s'] + cf['lease_margin_s']
                if rc != 0 and not old: continue          # network trouble: ask again next time
                done = any(w in low for w in ('complete', 'error', 'cancel'))
                if not (done or old) or not s.finish(name, ent, cf, 'complete' if 'complete' in low else ('error/cancel' if done else 'too old')): continue
                ent = dict(status='idle', last=ent['slug'], t_last=cm.now()); st[name] = ent; s.save(st)          # finished: the same tick may start the next batch
            if cf.get('enabled') and s.c.get_control() != 'stop':
                new = s.start(slot, cf)
                if new: st[name] = new; s.save(st)
        st['_tick'] = cm.now(); s.save(st)

def main():
    a = argparse.ArgumentParser(); a.add_argument('campaign'); a.add_argument('--once', action='store_true'); a.add_argument('--poll-s', type=float, default=300); g = a.parse_args()
    c = Ctl(g.campaign); say('kaggle controller for %s (user %s)' % (g.campaign, 'set' if c.user else 'MISSING'))
    while True:
        try: c.tick()
        except Exception as e: say('tick error (retried next poll):', repr(e))
        if g.once: break
        time.sleep(g.poll_s)

if __name__ == '__main__':
    main()
