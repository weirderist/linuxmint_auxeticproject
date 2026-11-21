#!/usr/bin/env python3
"""shell_proposer_eval.py - the FIXED test of the shell proposer against random picking. Offline, read-only on the campaign (numpy only); writes <campaign>/analysis/proposer_eval.json.

Question: does the proposer's selection rule (shell_proposer.select_batch, imported unchanged: greedy batch by the largest GP predictive std, length scale 1.4, noise 1e-2, kriging believer for the variance) give
a better model per run spent than picking runs at random?
Procedure on the finished, non-control runs of a campaign (default shell1; nothing is computed from fewer than MIN_RUNS = 100):
  1. held-out set: every run whose sha1(run id + split seed) falls in the lowest test_frac (default 25 %) is held out. The split depends only on the run id, so a run never changes sides as the campaign grows.
     The held-out runs are never chosen and are used only to measure the prediction error.
  2. the rest is the pool. Repeat R = 30 times: draw a random seed subset (start = 10 runs) from the pool; grow it to each budget (start + 10, start + 20, ... below the pool size) in two ways,
     (a) the proposer's rule, batch 3 (its default in propose()), (b) random picks; both start from the SAME seed subset (paired comparison).
  3. at every budget a GP (shell_proposer.gp_fit: RBF kernel, hyper-parameters by marginal likelihood on its grid) is fitted on the chosen runs for each target of the proposer (PROP_TARGETS) and its
     RMSE on the held-out runs is recorded. Features: shell_proposer.PROP_FEATURES standardised over all runs (features only, no targets).
  4. per budget: mean and standard error (over the R repeats) of the RMSE for both ways, per target; and the paired difference (proposer - random) in units of the held-out standard deviation of each target.
  5. verdict: D_r = mean over budgets and targets of the normalised paired difference in repeat r. 'better than random' if mean(D) + 2 SE < 0, 'worse than random' if mean(D) - 2 SE > 0, otherwise
     'no different from random'. 2 SE is the conventional ~95 % criterion (normal approximation), a choice, not a sourced threshold.
Limits (stated in the output): the SE describes the spread over seed subsets and random picks for THIS set of runs and THIS held-out split, not the spread over new campaigns; the proposer's real candidates
are new pattern edits, here it may only pick among runs that already exist (pool-based test, the standard offline form of an active-learning test).
Usage: python3 shell_proposer_eval.py [campaign] [--repeats 30] [--start 10] [--step 10] [--batch 3] [--test-frac 0.25] [--split-seed 0] [--seed 0] [--min-runs 100] [--out PATH]"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import shell_features as sf
import shell_proposer as pr                       # imported, never modified: select_batch, gp_fit, gp_predict, PROP_FEATURES, PROP_TARGETS

MIN_RUNS = 100
VERDICTS = ('better than random', 'no different from random', 'worse than random')


def proposer_rule(Xsel, Xrest, k, rng):
    """The proposer's own selection rule, called exactly as propose() calls it (default length scale and noise)."""
    return pr.select_batch(Xsel, Xrest, k)


def random_rule(Xsel, Xrest, k, rng):
    return [int(j) for j in rng.choice(len(Xrest), size=min(k, len(Xrest)), replace=False)]


def is_test(run_id, frac, split_seed=0):
    h = hashlib.sha1(('%s|%d' % (run_id, split_seed)).encode()).hexdigest(); return int(h[:12], 16) / float(16 ** 12) < frac


def rmse_gp(Xtr, ytr, Xte, yte):
    m = pr.gp_fit(Xtr, ytr); return float(np.sqrt(np.mean((pr.gp_predict(m, Xte)[0] - yte) ** 2)))


def curve(Xp, Yp, Xt, Yt, start_idx, budgets, rule, batch, rng):
    """Grow the chosen set from start_idx with `rule`; held-out RMSE per target at every budget. Returns {budget: [rmse per target]}."""
    chosen = list(start_idx); out = {}
    for b in budgets:
        while len(chosen) < b:
            cs = set(chosen); rest = [i for i in range(len(Xp)) if i not in cs]
            pick = rule(Xp[chosen], Xp[rest], min(batch, b - len(chosen)), rng); chosen += [rest[j] for j in pick]
        out[b] = [rmse_gp(Xp[chosen], Yp[chosen, t], Xt, Yt[:, t]) for t in range(Yp.shape[1])]
    return out


def evaluate_rows(rows, repeats=30, start=10, step=10, batch=3, test_frac=0.25, split_seed=0, seed=0, targets=pr.PROP_TARGETS, rule_a=proposer_rule, rule_b=random_rule, min_runs=MIN_RUNS):
    """rows: shell_features.dataset() rows (controls already removed). Returns the result dict (see the module docstring)."""
    rows = [r for r in rows if all(r['targets'].get(t) is not None and math.isfinite(r['targets'][t]) for t in targets) and all(math.isfinite(r['features'][k]) for k in pr.PROP_FEATURES)]
    out = dict(n_runs=len(rows), min_runs=min_runs, targets=list(targets), features=list(pr.PROP_FEATURES))
    if len(rows) < min_runs: out.update(status='waiting', note='%d finished runs with all targets; the test needs %d' % (len(rows), min_runs)); return out
    test = [r for r in rows if is_test(r['run'], test_frac, split_seed)]; pool = [r for r in rows if not is_test(r['run'], test_frac, split_seed)]
    Xall, mu, sd = pr.design(rows); X = lambda rr: (np.array([[r['features'][k] for k in pr.PROP_FEATURES] for r in rr], float) - mu) / sd
    Y = lambda rr: np.array([[r['targets'][t] for t in targets] for r in rr], float)
    Xp, Yp, Xt, Yt = X(pool), Y(pool), X(test), Y(test); sdt = Yt.std(0); sdt[sdt < 1e-12] = 1.0
    budgets = list(range(start + step, len(pool), step))
    if len(test) < 5 or not budgets: out.update(status='too few runs', note='held-out %d, pool %d: no budget between start %d and the pool size' % (len(test), len(pool), start)); return out
    A = np.zeros((repeats, len(budgets), len(targets))); B = np.zeros_like(A); rng = np.random.default_rng(seed)
    for r in range(repeats):
        s0 = [int(i) for i in rng.choice(len(pool), size=start, replace=False)]
        ca = curve(Xp, Yp, Xt, Yt, s0, budgets, rule_a, batch, np.random.default_rng(rng.integers(2 ** 31)))
        cb = curve(Xp, Yp, Xt, Yt, s0, budgets, rule_b, batch, np.random.default_rng(rng.integers(2 ** 31)))
        A[r] = [ca[b] for b in budgets]; B[r] = [cb[b] for b in budgets]
    se = lambda v: float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float('nan')
    D = ((A - B) / sdt[None, None, :]).mean(axis=(1, 2)); md, sdd = float(D.mean()), se(D)
    verdict = VERDICTS[0] if md + 2 * sdd < 0 else VERDICTS[2] if md - 2 * sdd > 0 else VERDICTS[1]
    per = []
    for i, b in enumerate(budgets):
        dn = ((A[:, i, :] - B[:, i, :]) / sdt[None, :]).mean(1)
        per.append(dict(budget=b, proposer={t: dict(mean_rmse=float(A[:, i, k].mean()), se=se(A[:, i, k])) for k, t in enumerate(targets)},
                        random={t: dict(mean_rmse=float(B[:, i, k].mean()), se=se(B[:, i, k])) for k, t in enumerate(targets)},
                        diff_norm=dict(mean=float(dn.mean()), se=se(dn))))
    out.update(status='ok', n_test=len(test), n_pool=len(pool), repeats=repeats, start=start, step=step, batch=batch, test_frac=test_frac, split_seed=split_seed, seed=seed, budgets=budgets,
               heldout_target_std={t: float(sdt[k]) for k, t in enumerate(targets)}, per_budget=per, mean_diff_norm=md, se_diff_norm=sdd,
               verdict='proposer selection is %s (mean normalised RMSE difference %+.4f, SE %.4f, %d repeats; negative = proposer better)' % (verdict, md, sdd, repeats), verdict_short=verdict,
               method='pool-based test: rule (a) shell_proposer.select_batch imported unchanged (batch %d), rule (b) random; same random seed subset of %d runs per repeat; GP of shell_proposer.gp_fit for both; '
                      'held-out by sha1 of the run id' % (batch, start),
               limits='SE = spread over seed subsets and random picks for these runs and this split, not over new campaigns; the real proposer picks among new pattern edits, here only among existing runs; '
                      'the 2 SE criterion is a conventional choice')
    return out


def run(camp, out_path=None, **kw):
    rows = [r for r in sf.dataset(camp) if not r['control']]
    res = dict(campaign=camp.name, generated=time.strftime('%Y-%m-%dT%H:%M:%S')); res.update(evaluate_rows(rows, **kw))
    path = out_path or os.path.join(camp.d, 'analysis', 'proposer_eval.json'); os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(res, f, indent=1)
    os.replace(tmp, path); return res, path


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0]); a.add_argument('campaign', nargs='?', default='shell1')
    for k, t, d in (('--repeats', int, 30), ('--start', int, 10), ('--step', int, 10), ('--batch', int, 3), ('--test-frac', float, 0.25), ('--split-seed', int, 0), ('--seed', int, 0), ('--min-runs', int, MIN_RUNS)): a.add_argument(k, type=t, default=d)
    a.add_argument('--out', default=None); g = a.parse_args()
    import shell_campaign as sc
    camp = sc.Camp(g.campaign)
    if not camp.exists(): sys.exit('no such shell campaign: %s' % g.campaign)
    try: os.nice(19)
    except (AttributeError, OSError): pass
    res, path = run(camp, g.out, repeats=g.repeats, start=g.start, step=g.step, batch=g.batch, test_frac=g.test_frac, split_seed=g.split_seed, seed=g.seed, min_runs=g.min_runs)
    print('%s: %s' % (path, res.get('verdict') or res.get('note') or res['status']))


if __name__ == '__main__':
    main()
