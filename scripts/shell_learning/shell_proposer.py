#!/usr/bin/env python3
"""shell_proposer.py - Stage 6: a model-based proposer for the pure-shell campaign. A separate process (numpy only, low priority) that ADDS runs to the queue ahead of the filler (tier 5 < 9), only through
Camp.enqueue, so the approved policy is enforced in code: ESSENTIAL settings are never touched, CPI/WPI stay inside the approved ratio range at constant density, the pattern is a slight edit of a seed.
Phase 1 (while the proposer gate is closed, see PROPOSER_MIN_ORIGINALS_FINISHED below; and in any case before MIN_RESULTS = 100 finished runs): nothing proposed - the seeds, variants and the deterministic
filler are the space-filling random sampling. After the gate opens, the total number of proposer runs in the campaign is capped at PROPOSER_MAX_TOTAL.
Phase 2: Gaussian-process uncertainty sampling on standardised pattern features + ln(CPI/WPI). What propose() really does (docstring corrected 2026-11-26): it uses the GP posterior standard deviation
with a FIXED RBF length scale 1.4 and noise 1e-2 (UNSOURCED), which does not depend on any outcome, so it is a space-filling design in feature space; gp_fit (hyper-parameters by marginal likelihood on a grid)
and PROP_TARGETS are used only by evaluate(). Proposals that are queued or running count as known points, so a later poll does not propose near-copies of them. Candidates are
slight edits of the seeds with a random ratio (tiles 4 only); the batch is chosen greedily by the largest predictive standard deviation (uncertainty sampling; the posterior variance does not depend on the unknown
outcomes, so adding a chosen point lowers the variance around it: batch selection without waiting for results). Objective: where a measured reference exists (only Pattern 1 at the success settings, which has no free
setting left), nothing is tuned; elsewhere the aim is to explore the fold space and record descriptors. evaluate() judges the proposer against random sampling on a fixed held-out test of finished runs.
Usage: python3 shell_proposer.py <campaign> [--once] [--dry-run] [--batch 3] [--poll-s 600]"""
import json, math, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_S = os.path.dirname(HERE)
if os.path.basename(_S) == 'scripts': sys.path.extend(os.path.join(_S, _d) for _d in sorted(os.listdir(_S)) if _d not in ('legacy', '__pycache__') and os.path.isdir(os.path.join(_S, _d)))          # the other folders of scripts/ (not in a flat frozen copy)
import shell_features as sf, shell_policy as sp

# ------------------------------------------------------------------ proposer gate and hard cap: the two constants K edits are here and nowhere else
# Definitions (used in this file and in tests/test_shell_proposer_gate.py):
#   proposer run        a queue entry of tier PROPOSER_TIER (5), the tier at which this script enqueues. An entry without a tier counts as tier 5, as in Camp.pending().
#   control run         a queue entry of tier CONTROL_TIER (1) or one that carries a 'control' record (the 2 iteration-cap controls of the policy).
#   original run        a queue entry that is neither a proposer run nor a control run (seeds tier 0, variants tier 2, filler tier 9).
#   originals finished  the number of original runs whose last line in index.jsonl has status 'done'. Failed, timed-out, rejected, quarantined, interrupted and queued runs do not count.
#   proposer gate       open when originals finished >= PROPOSER_MIN_ORIGINALS_FINISHED, otherwise closed. Fail safe: if the count cannot be read, the gate is closed.
#   hard cap            the number of proposer runs ever created in the campaign (every tier-5 entry in the append-only queue.jsonl, whatever its status) may not exceed PROPOSER_MAX_TOTAL.
# While the gate is closed or the cap is reached, propose() adds nothing. It never edits or removes an existing queue entry.
PROPOSER_MIN_ORIGINALS_FINISHED = 800    # Source: decision by K, 23 Nov 2026.
PROPOSER_MAX_TOTAL = 100                 # UNSOURCED default, to be adjusted by K.
PROPOSER_TIER = 5; CONTROL_TIER = 1      # tier numbers of shell_campaign.Camp.enqueue and shell_policy.build_plan (not new constants)

MIN_RESULTS = 100
PROP_FEATURES = ('knit_fraction', 'boundary_density', 'sections_per_100', 'tm_wale_share', 'tm_course_share', 'axis_direction', 'dilution_wale', 'log_ratio')
PROP_TARGETS = ('thickness_rel', 'shrink_y', 'fold_turn_x_rad')
LS_GRID = (0.7, 1.4, 2.8, 5.6); NOISE_GRID = (1e-3, 1e-2, 1e-1)


def rbf(A, B, ls):
    d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1); return np.exp(-0.5 * d / ls ** 2)


def gp_fit(X, y):
    """Best (length scale, noise) by log marginal likelihood on the grid; returns a model dict."""
    ym, ys = y.mean(), y.std() or 1.0; z = (y - ym) / ys; best = None
    for ls in LS_GRID:
        for nz in NOISE_GRID:
            K = rbf(X, X, ls) + nz * np.eye(len(X))
            try: L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError: continue
            a = np.linalg.solve(L.T, np.linalg.solve(L, z)); ll = -0.5 * z @ a - np.log(np.diag(L)).sum()
            if best is None or ll > best[0]: best = (ll, ls, nz, L, a)
    _, ls, nz, L, a = best; return dict(X=X, ls=ls, noise=nz, L=L, alpha=a, ym=ym, ys=ys)


def gp_predict(m, Xs):
    Ks = rbf(Xs, m['X'], m['ls']); mean = Ks @ m['alpha'] * m['ys'] + m['ym']; v = np.linalg.solve(m['L'], Ks.T); var = np.maximum(1.0 - (v ** 2).sum(0), 1e-12) * m['ys'] ** 2
    return mean, np.sqrt(var)


def posterior_std(X, Xs, ls, noise):
    """Predictive std (unit prior variance) at Xs given training inputs X only (independent of y)."""
    K = rbf(X, X, ls) + noise * np.eye(len(X)); Ks = rbf(Xs, X, ls); v = np.linalg.solve(np.linalg.cholesky(K), Ks.T); return np.sqrt(np.maximum(1.0 - (v ** 2).sum(0), 1e-12))


def design(rows, keys=PROP_FEATURES):
    X = np.array([[r['features'][k] for k in keys] for r in rows], float); mu = X.mean(0); sd = X.std(0); sd[sd < 1e-12] = 1.0; return (X - mu) / sd, mu, sd


def select_batch(Xtrain, Xcand, batch, ls=1.4, noise=1e-2):
    """Greedy batch by the largest predictive std; every chosen candidate is added to the training inputs (kriging believer for the variance)."""
    chosen = []; X = Xtrain.copy(); avail = list(range(len(Xcand)))
    for _ in range(min(batch, len(avail))):
        sd = posterior_std(X, Xcand[avail], ls, noise); j = avail[int(np.argmax(sd))]; chosen.append(j); avail.remove(j); X = np.vstack([X, Xcand[j][None, :]])
    return chosen


def evaluate(rows, repeats=6, seed=0, target='thickness_rel', start=10, budgets=(20, 40, 60, 80)):
    """Fixed test: 25 % of the finished runs are held out (fixed split by seed); from the rest, a training set grows to each budget either by random choice or by uncertainty sampling (start = 10 random runs).
    Returns the mean held-out RMSE of a GP trained on the chosen runs, for both, per budget."""
    rows = [r for r in rows if r['targets'].get(target) is not None]; rng = np.random.default_rng(seed); order = rng.permutation(len(rows)); nt = max(1, len(rows) // 4)
    test = [rows[i] for i in order[:nt]]; pool = [rows[i] for i in order[nt:]]; Xall, mu, sd = design(rows); Xp = ((np.array([[r['features'][k] for k in PROP_FEATURES] for r in pool]) - mu) / sd)
    Xt = ((np.array([[r['features'][k] for k in PROP_FEATURES] for r in test]) - mu) / sd); yp = np.array([r['targets'][target] for r in pool]); yt = np.array([r['targets'][target] for r in test]); out = {b: dict(random=[], gp=[]) for b in budgets}
    for rep in range(repeats):
        r2 = np.random.default_rng(seed + 100 + rep); perm = r2.permutation(len(pool))
        for b in budgets:
            if b > len(pool): continue
            ir = perm[:b]; m = gp_fit(Xp[ir], yp[ir]); out[b]['random'].append(float(np.sqrt(np.mean((gp_predict(m, Xt)[0] - yt) ** 2))))
            chosen = list(perm[:start])
            while len(chosen) < b:
                rest = [i for i in range(len(pool)) if i not in chosen]; sd_ = posterior_std(Xp[chosen], Xp[rest], 1.4, 1e-2); chosen.append(rest[int(np.argmax(sd_))])
            m = gp_fit(Xp[chosen], yp[chosen]); out[b]['gp'].append(float(np.sqrt(np.mean((gp_predict(m, Xt)[0] - yt) ** 2))))
    return {b: dict(random_rmse=float(np.mean(v['random'])), gp_rmse=float(np.mean(v['gp'])), repeats=len(v['random'])) for b, v in out.items() if v['random']}


def candidates(policy, seeds, rows, rng, n=400, tiles=4, bases=None):
    """Policy-valid candidates. Policy version 1 (shell1): slight edits of a seed, ratio log-uniform in the approved range at constant density, the given tiles. Policy version 2 (shell2,
    2026-11-27): the same mix as the campaign plan (policy 'mix': random blocks, pixel-random, resized seed edits), stitch density uniform in density_range (density is not a proposer feature:
    the shell barely depends on it at fixed R, measured 2026-11-21), tiles = shell_policy.tiles_for (the largest within the stitch cap). Returns a list of (settings, features)."""
    rg = policy['ranges']; lo, hi = rg['ratio_cpi_over_wpi']; out = []; tries = 0; seen = set(); v2 = 'density_range' in rg
    mix = sorted((policy.get('mix') or {'edits': 1.0}).items())
    while len(out) < n and tries < n * 20:
        tries += 1; kind = 'edits'
        if v2:
            u = rng.random(); acc = 0.0
            for k, w in mix:
                acc += w
                if u <= acc: kind = k; break
        if kind == 'edits':
            base = seeds[rng.randrange(len(seeds))]; B = sp.resize_matrix(base['matrix'], rng, rg['matrix_side'][0], rg['matrix_side'][1], rg['max_resize']) if v2 else base['matrix']
            M = sp.edit_matrix(B, rng, policy); bid = base['id']
        else:
            sl, sh = rg['matrix_side']; M = (sp.random_blocks if kind == 'blocks' else sp.random_pixels)(rng, rng.randint(sl, sh), rng.randint(sl, sh)); bid = 'random:' + kind
        if M is None: continue
        dens = rng.uniform(*rg['density_range']) if v2 else rg['density_cpi_x_wpi']; t = sp.tiles_for(policy, M) if v2 else tiles
        if t is None: continue
        R = math.exp(math.log(lo) + rng.random() * (math.log(hi) - math.log(lo))); cpi, wpi = sp.cpi_wpi(R, dens); s = sp._setting(policy, M, cpi, wpi, t)
        if len(M) * len(M[0]) * t * t > rg['max_stitches'] or (v2 and sp.violations(s, policy)): continue
        h = sp._sha(s)
        if h in seen: continue
        seen.add(h); out.append((s, sf.features(M, t, cpi, wpi)))
        if bases is not None: bases.append(bid)          # the seed this candidate edits (recorded in the label: rule checks grouped by seed family need it)
    return out


def is_proposer_run(q): return q.get('tier', PROPOSER_TIER) == PROPOSER_TIER
def is_control_run(q): return q.get('tier', PROPOSER_TIER) == CONTROL_TIER or bool(q.get('control'))
def is_original_run(q): return not is_proposer_run(q) and not is_control_run(q)


def _readable(*paths):
    """True only if every file exists and can be opened for reading (a missing, unreadable or directory path gives False)."""
    try:
        for p in paths: open(p, 'rb').close()
        return True
    except OSError: return False


def originals_finished(camp):
    """Originals finished (definition above), or None if queue.jsonl or index.jsonl cannot be read. Read only."""
    try:
        if not _readable(camp.queue, camp.index): return None
        q = {x['id']: x for x in camp.queue_items()}
        return sum(1 for rid, x in camp.index_last().items() if x.get('status') == 'done' and rid in q and is_original_run(q[rid]))
    except Exception: return None


def proposer_runs_created(camp):
    """Number of proposer runs ever created (every tier-5 entry in queue.jsonl), or None if queue.jsonl cannot be read. Read only."""
    try:
        if not _readable(camp.queue): return None
        return sum(1 for x in camp.queue_items() if is_proposer_run(x))
    except Exception: return None


def gate(camp):
    """State of the proposer gate and the hard cap. 'open' is True only if the gate is open AND the cap leaves room; 'room' = proposals that may still be created."""
    ov = (camp.spec_().get('proposer') or {}) if hasattr(camp, 'spec_') else {}          # per-campaign values (K 2026-11-27: shell2 gate 1000 originals, cap 300); absent = the defaults above
    lo = int(ov.get('min_originals_finished', PROPOSER_MIN_ORIGINALS_FINISHED)); cap = int(ov.get('max_total', PROPOSER_MAX_TOTAL))
    n = originals_finished(camp); p = proposer_runs_created(camp); out = dict(originals_finished=n, min_originals_finished=lo, proposer_runs=p, max_total=cap, open=False, room=0)
    if n is None or p is None: out.update(status='gate closed (count unreadable)', note='the finished count could not be read: fail safe, nothing is proposed'); return out
    if n < lo: out.update(status='gate closed', note='%d original runs finished, the proposer starts at %d' % (n, lo)); return out
    if p >= cap: out.update(status='cap reached', note='%d proposer runs exist, the cap is %d' % (p, cap)); return out
    out.update(open=True, room=cap - p, status='gate open', note='%d original runs finished; %d of %d proposer runs used' % (n, p, cap)); return out


def propose(camp, batch=3, dry=False, seed=None):
    import shell_campaign as sc
    pol = camp.policy()
    if not pol: return dict(status='no policy', proposed=[])
    g = gate(camp)
    if not g['open']: return dict(status=g['status'], note=g['note'], proposed=[], gate=g)
    rows = sf.dataset(camp)
    if len(rows) < MIN_RESULTS: return dict(status='waiting', note='%d finished runs, the model starts at %d' % (len(rows), MIN_RESULTS), proposed=[])
    if sum(1 for x in camp.pending() if x.get('tier') == 5) >= 2: return dict(status='enough proposals queued', proposed=[])
    plan = camp.spec_().get('plan') or {}; base = os.path.dirname(plan['seeds_file']); sj = json.load(open(plan['seeds_file'], encoding='utf8'))
    seeds = [dict(id=x['id'], matrix=sc.matrix_from_text(open(os.path.join(base, x['file']), encoding='utf8').read())) for x in sj['seeds']]
    qn = len(camp.queue_items()); bases = []
    rng = random.Random(seed if seed is not None else len(rows) * 100003 + qn); cand = candidates(pol, seeds, rows, rng, n=max(400, 4 * batch), bases=bases)          # the queue length also seeds it: an unchanged run count gave the same candidates again
    Xtr, mu, sd = design(rows); Xc = (np.array([[f[k] for k in PROP_FEATURES] for _, f in cand]) - mu) / sd
    pend = [x['settings'] for x in camp.pending() if x.get('tier', PROPOSER_TIER) == PROPOSER_TIER]          # queued / running proposals are known points too (2026-11-26)
    if pend: Xtr = np.vstack([Xtr, (np.array([[sf.features(s['matrix'], s['tiles'], s['cpi'], s['wpi'])[k] for k in PROP_FEATURES] for s in pend]) - mu) / sd])
    chosen = select_batch(Xtr, Xc, min(batch, g['room'])); ids = []
    for j in chosen:
        s, f = cand[j]
        if not dry: ids.append(camp.enqueue(s, source='proposer:gp', tier=5, label='GP uncertainty sampling, seed %s, R %.3f' % (bases[j] if j < len(bases) else '?', s['cpi'] / s['wpi'])))
        else: ids.append(None)
    return dict(status='proposed' if not dry else 'dry run', proposed=ids, n_finished=len(rows), batch=batch)


def main():
    import shell_campaign as sc
    a = sys.argv[1:]; camp = sc.Camp(a[0]); once = '--once' in a; dry = '--dry-run' in a; batch = int(a[a.index('--batch') + 1]) if '--batch' in a else 3; poll = float(a[a.index('--poll-s') + 1]) if '--poll-s' in a else 600.0
    if not camp.exists(): sys.exit('no such shell campaign: %s' % a[0])
    try: os.nice(19)
    except (AttributeError, OSError): pass
    while True:
        r = propose(camp, batch, dry); print('%s proposer: %s %s' % (time.strftime('%F %T'), r['status'], r.get('proposed', '')), flush=True)
        if once: break
        time.sleep(poll)


if __name__ == '__main__':
    main()
