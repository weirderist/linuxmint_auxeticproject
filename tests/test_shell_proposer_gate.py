#!/usr/bin/env python3
"""Tests of the proposer gate and the hard cap in scripts/shell_proposer.py (decision by K, 23 Nov 2026) on a SYNTHETIC seeded campaign: no solver, numpy only, temporary folders.
Terms (defined in scripts/shell_proposer.py): proposer run = tier 5; control run = tier 1 or a 'control' record; original run = neither; originals finished = original runs whose last index status is 'done';
proposer gate = open at PROPOSER_MIN_ORIGINALS_FINISHED (800) originals finished; hard cap = at most PROPOSER_MAX_TOTAL (100, UNSOURCED) proposer runs ever created.
What this file does NOT test: that the RUNNER skips queued tier-5 entries while the gate is closed. The runner (Camp.pending and serve in scripts/shell_campaign.py) is frozen code of campaign shell1
and was not changed; the INFO line below shows that the unchanged runner still puts queued tier-5 entries before the filler.
Run from the repository root: python tests/test_shell_proposer_gate.py"""
import hashlib, inspect, os, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='shellgate_'); SR = os.path.join(TMP, 'shell_runs')
os.environ.update(SHELL_RUNS_DIR=SR, CAMPAIGN_LOG_DIR=os.path.join(TMP, 'logs'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import shell_campaign as sc, shell_proposer as pr, campaign as cm
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)
def digest(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

check('constants: PROPOSER_MIN_ORIGINALS_FINISHED = 800 (K, 23 Nov 2026), PROPOSER_MAX_TOTAL = 100 (UNSOURCED)', pr.PROPOSER_MIN_ORIGINALS_FINISHED == 800 and pr.PROPOSER_MAX_TOTAL == 100)
sb = inspect.signature(pr.select_batch).parameters
check('the proposer stays target-blind: select_batch keeps length scale 1.4 and noise 1e-2', sb['ls'].default == 1.4 and sb['noise'].default == 1e-2)

# ---- one seeded campaign (11 seeds, 2 controls, 91 variants, 720 filler = 822 original runs), copied for every case
r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'campaign.py'), 'shell-plan', 'base', '--policy', os.path.join(ROOT, 'campaigns', 'shell_policy.json'), '--filler', '720'], capture_output=True, text=True, env=dict(os.environ))
base = sc.Camp('base'); Q = base.queue_items(); ORIG = [x['id'] for x in Q if pr.is_original_run(x)]; CTRL = [x['id'] for x in Q if pr.is_control_run(x)]
check('synthetic campaign: 822 original runs, 2 control runs (tier 1), no proposer runs', r.returncode == 0 and len(ORIG) == 822 and len(CTRL) == 2 and all(x['tier'] == 1 for x in Q if x['id'] in CTRL) and not any(pr.is_proposer_run(x) for x in Q), (r.stdout + r.stderr)[-200:])

def fresh(name):
    shutil.copytree(base.d, os.path.join(SR, name)); return sc.Camp(name)
def mark(c, ids, status='done'):
    for rid in ids: sc.append_line(c.index, dict(id=rid, status=status, t=cm.now(), metrics={}, wall_s=1.0))
def add_proposer_entries(c, n, done=False):
    """Append n tier-5 entries the way old proposer runs look in queue.jsonl (test fixture only; the real queue is never written by a test)."""
    st = Q[0]['settings']; ids = []
    for k in range(n):
        rid = c.next_id(); sc.append_line(c.queue, dict(id=rid, t=cm.now(), source='proposer:gp', settings=st, sha1='fixture%04d' % k, tier=5, label='fixture')); ids.append(rid)
    if done: mark(c, ids)
    return ids
def new_tier5(c, before): return [x for x in c.queue_items()[before:] if x['tier'] == 5]

# ---- gate closed at 799 originals, open at 800
c = fresh('g799'); mark(c, ORIG[:799]); qd = digest(c.queue); res = pr.propose(c, 3)
check('799 originals finished: gate closed, nothing proposed, queue.jsonl byte-identical', pr.originals_finished(c) == 799 and res['status'] == 'gate closed' and res['proposed'] == [] and digest(c.queue) == qd, res.get('note', ''))
c = fresh('g800'); mark(c, ORIG[:800]); n0 = len(c.queue_items()); head = open(c.queue, 'rb').read(); res = pr.propose(c, 3); new = new_tier5(c, n0)
check('800 originals finished: gate open, 3 proposals at tier 5 through the existing refill', pr.originals_finished(c) == 800 and res['status'] == 'proposed' and len(new) == 3 and all(x['source'] == 'proposer:gp' for x in new), res['status'])
check('the proposals are appended only: every earlier byte of queue.jsonl is unchanged', open(c.queue, 'rb').read()[:len(head)] == head)
check('existing refill rule kept: with 2 or more proposals pending, nothing more is added', pr.propose(c, 3)['status'] == 'enough proposals queued')

# ---- what does not count
c = fresh('gfail'); mark(c, ORIG[:799]); mark(c, ORIG[799:802], 'failed'); mark(c, ORIG[802:804], 'timeout'); mark(c, ORIG[804:806], 'needs_approval'); mark(c, ORIG[806:808], 'rejected')
check('failed, timed-out, needs_approval (interrupted 3 times) and rejected originals do not count: 799, gate closed', pr.originals_finished(c) == 799 and pr.propose(c, 3)['status'] == 'gate closed')
c = fresh('gredo'); mark(c, ORIG[:800]); mark(c, ORIG[:1], 'failed')
check('the last index line wins: a run marked done and then failed does not count (799, gate closed)', pr.originals_finished(c) == 799 and pr.propose(c, 3)['status'] == 'gate closed')
c = fresh('gctrl'); mark(c, ORIG[:799]); mark(c, CTRL)
check('control runs are excluded: 799 originals + 2 finished controls = 799, gate closed', pr.originals_finished(c) == 799 and pr.propose(c, 3)['status'] == 'gate closed')
c = fresh('gprop'); mark(c, ORIG[:799]); add_proposer_entries(c, 200, done=True)
check('proposer runs are excluded: 799 originals + 200 finished tier-5 runs = 799, gate closed', pr.originals_finished(c) == 799 and pr.propose(c, 3)['status'] == 'gate closed')

# ---- already-queued tier-5 entries while the gate is closed
c = fresh('gqueued'); mark(c, ORIG[:500]); mark(c, CTRL); queued = add_proposer_entries(c, 4); qd, idd = digest(c.queue), digest(c.index)
outs = [pr.propose(c, 3) for _ in range(3)]
p = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'shell_proposer.py'), 'gqueued', '--once'], capture_output=True, text=True, env=dict(os.environ))
check('gate closed with 4 tier-5 entries queued: the proposer adds nothing (3 calls and the --once command line)', all(o['status'] == 'gate closed' and o['proposed'] == [] for o in outs) and p.returncode == 0 and 'gate closed' in p.stdout, p.stdout.strip()[-120:])
check('the queued tier-5 entries are untouched on disk: queue.jsonl and index.jsonl byte-identical, all 4 still present and pending', digest(c.queue) == qd and digest(c.index) == idd and set(queued) <= {x['id'] for x in c.pending()})
first = c.pending()[0]
print('INFO  not tested (frozen code, unchanged): the runner takes pending()[0], which is now %s at tier %s; the frozen runner WOULD dispatch the queued tier-5 entries while the gate is closed' % (first['id'], first['tier']))

# ---- hard cap
c = fresh('gcap200'); mark(c, ORIG[:810]); add_proposer_entries(c, 200, done=True); n0 = len(c.queue_items()); res = pr.propose(c, 3)
check('gate open but 200 proposer runs already exist (cap 100): cap reached, nothing proposed', res['status'] == 'cap reached' and res['proposed'] == [] and len(c.queue_items()) == n0, res.get('note', ''))
c = fresh('gcap100'); mark(c, ORIG[:810]); add_proposer_entries(c, 100, done=True); res = pr.propose(c, 3)
check('exactly 100 proposer runs (finished): cap reached', res['status'] == 'cap reached')
c = fresh('gcap98'); mark(c, ORIG[:810]); add_proposer_entries(c, 98, done=True); n0 = len(c.queue_items()); res = pr.propose(c, 3); new = new_tier5(c, n0)
check('98 proposer runs: only 2 more are created, so the total stops at the cap of 100', res['status'] == 'proposed' and len(new) == 2 and pr.proposer_runs_created(c) == 100, str(len(new)))
check('after that the cap holds: the next call adds nothing', pr.propose(c, 3)['status'] in ('cap reached', 'enough proposals queued') and pr.proposer_runs_created(c) == 100)

# ---- fail safe: an unreadable count keeps the gate closed
c = fresh('gunread1'); mark(c, ORIG[:810]); os.rename(c.index, c.index + '.moved'); os.makedirs(c.index); n0 = len(c.queue_items()); res = pr.propose(c, 3)
check('index.jsonl unreadable (replaced by a folder in this test copy) with 810 originals finished: gate closed, nothing added', pr.originals_finished(c) is None and res['status'] == 'gate closed (count unreadable)' and len(c.queue_items()) == n0, res['status'])
c = fresh('gunread2'); mark(c, ORIG[:810])
def boom(): raise OSError('simulated read error')
c.index_last = boom; n0 = len(c.queue_items()); res = pr.propose(c, 3)
check('reading the index raises an error: gate closed, nothing added', pr.originals_finished(c) is None and res['status'] == 'gate closed (count unreadable)' and len(c.queue_items()) == n0, res['status'])
c = fresh('gunread3'); mark(c, ORIG[:810]); os.rename(c.queue, c.queue + '.moved'); res = pr.propose(c, 3)
check('queue.jsonl missing: both counts unreadable, gate closed, nothing created', pr.originals_finished(c) is None and pr.proposer_runs_created(c) is None and res['status'] == 'gate closed (count unreadable)' and not os.path.exists(c.queue))

print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
