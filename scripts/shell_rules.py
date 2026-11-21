#!/usr/bin/env python3
"""shell_rules.py - Stage 7: framing of rules from the saved shell runs. A separate offline job (numpy only), run at low priority; it only READS the campaign and writes analysis/rules.json (+ a numbered copy).

For every target (shell_features.TARGETS: scale-free fold descriptors) it regresses the target on the pattern features (shell_features.FEATURES):
  1. sparse linear regression (lasso, coordinate descent on standardised features), lambda chosen by 5-fold cross-validation with the one-standard-error rule; reported: standardised coefficients, how often each
     feature is selected in 40 bootstrap refits (stability), cross-validated R^2 against the mean predictor, number of runs;
  2. a shallow regression tree (depth <= 3, at least 10 runs per leaf) as a cross-check, with its cross-validated R^2 and the rules as text.
Symbolic regression is NOT done (it needs an extra package: K decides). Nothing is reported from fewer than MIN_RUNS = 100 finished runs (a rule from fewer is not shown: status 'waiting').
Every rule carries the label: learned from the shell, not confirmed on real fabric. Controls and rejected runs are excluded. The thresholds in the features are UNSOURCED (shell_features.py).
Usage: python3 shell_rules.py <campaign> [--once] [--every N] [--poll-s 300]   (the loop lowers its priority with os.nice(19))"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import shell_features as sf

MIN_RUNS = 100
LABEL = 'learned from the shell, not confirmed on real fabric'


def standardise(X):
    m = X.mean(0); s = X.std(0); s[s < 1e-12] = 1.0; return (X - m) / s, m, s


def lasso(Z, y, lam, iters=300):
    n, p = Z.shape; b = np.zeros(p); r = y - y.mean(); col = (Z ** 2).sum(0) / n
    for _ in range(iters):
        old = b.copy()
        for j in range(p):
            if col[j] < 1e-12: continue
            rj = r + Z[:, j] * b[j]; rho = Z[:, j] @ rj / n; b[j] = np.sign(rho) * max(abs(rho) - lam, 0.0) / col[j]; r = rj - Z[:, j] * b[j]
        if np.abs(b - old).max() < 1e-7: break
    return b


def folds(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n); return [idx[i::k] for i in range(k)]


def cv_lasso(Z, y, lams, k=5, seed=0):
    mse = np.zeros((len(lams), k))
    for f, te in enumerate(folds(len(y), k, seed)):
        tr = np.setdiff1d(np.arange(len(y)), te); Zt, mt, st = standardise(Z[tr]); ym = y[tr].mean()
        for i, lam in enumerate(lams): b = lasso(Zt, y[tr], lam); mse[i, f] = np.mean(((Z[te] - mt) / st @ b + ym - y[te]) ** 2)
    return mse.mean(1), mse.std(1) / math.sqrt(k)


def baseline_mse(y, k=5, seed=0):
    out = []
    for te in folds(len(y), k, seed): tr = np.setdiff1d(np.arange(len(y)), te); out.append(np.mean((y[tr].mean() - y[te]) ** 2))
    return float(np.mean(out))


def fit_tree(X, y, depth=3, min_leaf=10):
    if depth == 0 or len(y) < 2 * min_leaf or np.ptp(y) < 1e-12: return dict(leaf=float(y.mean()), n=int(len(y)))
    best = None
    for j in range(X.shape[1]):
        for t in np.unique(np.quantile(X[:, j], np.linspace(0.1, 0.9, 9))):
            L = X[:, j] <= t
            if L.sum() < min_leaf or (~L).sum() < min_leaf: continue
            sse = ((y[L] - y[L].mean()) ** 2).sum() + ((y[~L] - y[~L].mean()) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, j, float(t))
    if best is None or best[0] >= ((y - y.mean()) ** 2).sum() - 1e-12: return dict(leaf=float(y.mean()), n=int(len(y)))
    _, j, t = best; L = X[:, j] <= t
    return dict(feature=j, threshold=t, left=fit_tree(X[L], y[L], depth - 1, min_leaf), right=fit_tree(X[~L], y[~L], depth - 1, min_leaf), n=int(len(y)))


def predict_tree(tr, x):
    while 'leaf' not in tr: tr = tr['left'] if x[tr['feature']] <= tr['threshold'] else tr['right']
    return tr['leaf']


def tree_rules(tr, names, prefix=''):
    if 'leaf' in tr: return ['%smean %.4g (%d runs)' % (prefix and prefix + ' then ', tr['leaf'], tr['n'])]
    f, t = names[tr['feature']], tr['threshold']
    return tree_rules(tr['left'], names, (prefix + ' and ' if prefix else 'if ') + '%s <= %.4g' % (f, t)) + tree_rules(tr['right'], names, (prefix + ' and ' if prefix else 'if ') + '%s > %.4g' % (f, t))


def analyse_target(rows, target, boots=40, seed=0):
    names = list(sf.FEATURES); pts = [(r['features'], r['targets'].get(target)) for r in rows if r['targets'].get(target) is not None and all(math.isfinite(v) for v in r['features'].values())]
    n = len(pts); out = dict(target=target, n_runs=n)
    if n < MIN_RUNS: out.update(status='waiting', note='fewer than %d finished runs with this descriptor: no rule is reported' % MIN_RUNS); return out
    X = np.array([[f[k] for k in names] for f, _ in pts]); y = np.array([v for _, v in pts], float); Z, m, s = standardise(X); base = baseline_mse(y)
    lams = np.geomspace(max(np.abs(Z.T @ (y - y.mean())).max() / n, 1e-9), 1e-3 * max(np.abs(Z.T @ (y - y.mean())).max() / n, 1e-9), 12); cvm, cvse = cv_lasso(X, y, lams)
    i0 = int(np.argmin(cvm)); i1 = next(i for i in range(len(lams)) if cvm[i] <= cvm[i0] + cvse[i0]); lam = lams[i1]; b = lasso(Z, y, lam); rng = np.random.default_rng(seed); sel = np.zeros(len(names))
    for _ in range(boots): ix = rng.integers(0, n, n); Zb, _, _ = standardise(X[ix]); sel += lasso(Zb, y[ix], lam) != 0
    coef = [dict(feature=names[j], coefficient_std=float(b[j]), per_unit=float(b[j] / s[j]), stability=float(sel[j] / boots)) for j in np.argsort(-np.abs(b)) if b[j] != 0]
    tmse = []
    for te in folds(n, 5, 1): tr = np.setdiff1d(np.arange(n), te); t = fit_tree(X[tr], y[tr]); tmse.append(np.mean([(predict_tree(t, X[i]) - y[i]) ** 2 for i in te]))
    tree = fit_tree(X, y)
    out.update(status='ok', label=LABEL, lasso=dict(lambda_=float(lam), intercept=float(y.mean()), coefficients=coef, cv_mse=float(cvm[i1]), baseline_cv_mse=base, cv_r2=float(1 - cvm[i1] / base), support=n, n_selected=len(coef)),
               tree=dict(rules=tree_rules(tree, names), cv_mse=float(np.mean(tmse)), cv_r2=float(1 - np.mean(tmse) / base), support=n), target_mean=float(y.mean()), target_std=float(y.std()))
    return out


def run(camp):
    rows = [r for r in sf.dataset(camp) if not r['control']]; res = dict(campaign=camp.name, generated=time.strftime('%Y-%m-%dT%H:%M:%S'), n_finished=len(rows), min_runs=MIN_RUNS, label=LABEL,
                                                                      method='lasso (CV, 1-SE rule, bootstrap stability) and a depth-3 tree; no symbolic regression (needs an extra package, K decides)', features=list(sf.FEATURES),
                                                                      targets=[analyse_target(rows, t) for t in sf.TARGETS])
    d = os.path.join(camp.d, 'analysis'); os.makedirs(d, exist_ok=True)
    for path in (os.path.join(d, 'rules_%06d.json' % len(rows)), os.path.join(d, 'rules.json')):
        tmp = path + '.tmp'; json.dump(res, open(tmp, 'w'), indent=1); os.replace(tmp, path)
    return res


def main():
    import shell_campaign as sc
    a = sys.argv[1:]; name = a[0]; once = '--once' in a; every = int(a[a.index('--every') + 1]) if '--every' in a else 25; poll = float(a[a.index('--poll-s') + 1]) if '--poll-s' in a else 300.0
    camp = sc.Camp(name)
    if not camp.exists(): sys.exit('no such shell campaign: %s' % name)
    try: os.nice(19)
    except (AttributeError, OSError): pass
    last = -10 ** 9
    while True:
        n = sum(1 for x in camp.index_last().values() if x.get('status') == 'done')
        if n - last >= every or once:
            res = run(camp); last = n; print('%s rules: %d finished runs, %s' % (time.strftime('%F %T'), n, ', '.join('%s %s' % (t['target'], t['status']) for t in res['targets'])), flush=True)
        if once: break
        time.sleep(poll)


if __name__ == '__main__':
    main()
