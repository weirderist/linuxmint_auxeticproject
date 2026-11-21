#!/usr/bin/env python3
"""shell_proposer.py - Stage 6: a model-based proposer for the pure-shell campaign. A separate process (numpy only, low priority) that ADDS runs to the queue ahead of the filler (tier 5 < 9), only through
Camp.enqueue, so the approved policy is enforced in code: ESSENTIAL settings are never touched, CPI/WPI stay inside the approved ratio range at constant density, the pattern is a slight edit of a seed.
Phase 1 (before MIN_RESULTS = 100 finished runs): nothing proposed - the seeds, variants and the deterministic filler are the space-filling random sampling.
Phase 2: a Gaussian process (RBF kernel, hyper-parameters by marginal likelihood on a grid) per target (thickness_rel, shrink_y, fold_turn_x_rad) on standardised pattern features + ln(CPI/WPI). Candidates are
slight edits of the seeds with a random ratio; the batch is chosen greedily by the largest mean predictive standard deviation (uncertainty sampling; the posterior variance does not depend on the unknown
outcomes, so adding a chosen point lowers the variance around it: batch selection without waiting for results). Objective: where a measured reference exists (only Pattern 1 at the success settings, which has no free
setting left), nothing is tuned; elsewhere the aim is to explore the fold space and record descriptors. evaluate() judges the proposer against random sampling on a fixed held-out test of finished runs.
Usage: python3 shell_proposer.py <campaign> [--once] [--dry-run] [--batch 3] [--poll-s 600]"""
import json, math, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import shell_features as sf, shell_policy as sp

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


def candidates(policy, seeds, rows, rng, n=400, tiles=4):
    """Policy-valid candidates: slight edits of a seed, ratio log-uniform in the approved range at constant density. Returns a list of (settings, features)."""
    rg = policy['ranges']; lo, hi = rg['ratio_cpi_over_wpi']; out = []; tries = 0; seen = set()
    while len(out) < n and tries < n * 20:
        tries += 1; base = seeds[rng.randrange(len(seeds))]; M = sp.edit_matrix(base['matrix'], rng, policy)
        if M is None: continue
        R = math.exp(math.log(lo) + rng.random() * (math.log(hi) - math.log(lo))); cpi, wpi = sp.cpi_wpi(R, rg['density_cpi_x_wpi']); s = sp._setting(policy, M, cpi, wpi, tiles)
        if len(M) * len(M[0]) * tiles * tiles > rg['max_stitches']: continue
        h = sp._sha(s)
        if h in seen: continue
        seen.add(h); out.append((s, sf.features(M, tiles, cpi, wpi)))
    return out


def propose(camp, batch=3, dry=False, seed=None):
    import shell_campaign as sc
    pol = camp.policy(); rows = sf.dataset(camp)
    if not pol: return dict(status='no policy', proposed=[])
    if len(rows) < MIN_RESULTS: return dict(status='waiting', note='%d finished runs, the model starts at %d' % (len(rows), MIN_RESULTS), proposed=[])
    if sum(1 for x in camp.pending() if x.get('tier') == 5) >= 2: return dict(status='enough proposals queued', proposed=[])
    plan = camp.spec_().get('plan') or {}; base = os.path.dirname(plan['seeds_file']); sj = json.load(open(plan['seeds_file'], encoding='utf8'))
    seeds = [dict(id=x['id'], matrix=sc.matrix_from_text(open(os.path.join(base, x['file']), encoding='utf8').read())) for x in sj['seeds']]
    rng = random.Random(seed if seed is not None else len(rows)); cand = candidates(pol, seeds, rows, rng)
    Xtr, mu, sd = design(rows); Xc = (np.array([[f[k] for k in PROP_FEATURES] for _, f in cand]) - mu) / sd
    chosen = select_batch(Xtr, Xc, batch); ids = []
    for j in chosen:
        s, f = cand[j]
        if not dry: ids.append(camp.enqueue(s, source='proposer:gp', tier=5, label='GP uncertainty sampling, R %.3f' % (s['cpi'] / s['wpi'])))
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
