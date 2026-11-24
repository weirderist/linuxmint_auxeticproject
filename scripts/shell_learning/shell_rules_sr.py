#!/usr/bin/env python3
"""shell_rules_sr.py - open-ended rule search on the pure-shell runs, with a feature registry. Offline, read-only on the campaign, numpy only; writes <campaign>/analysis/rules_sr.json.
A separate job from shell_rules.py (the existing linear rules); it imports shell_rules and shell_features unchanged and uses the linear rules as the baseline a new rule must beat.

(a) Feature generator, from the TILED matrix T (rows = courses, columns = wales; digits 1 and 2) and the run settings:
    base       every feature of shell_features.features (the existing linear-rule features) and ratio = CPI / WPI;
    motifs     frequencies of every 2x2 window (m2x2_<row>/<row>), every 3x1 window = 3 courses in one wale (m3x1_<d>/<d>/<d>, along the wale) and every 1x3 window = 3 wales in one course
               (m1x3_<ddd>, along the course); rows are written in matrix row order;
    runs       run lengths of equal digits along the wales (run_wale_*) and along the courses (run_course_*): mean, stitch-weighted mean (sum L^2 / sum L), longest / line length, stitch-weighted mean per digit;
    sections   4-connected regions of one digit: mean and stitch-weighted mean width and height (stitches), stitch-weighted mean ln(height / width), stitch-weighted distance of the region centre from the sheet
               centre (/ half diagonal), share of stitches in regions that touch the sheet edge; centre_minus_edge_d1 = share of digit 1 in the central half box minus that outside it;
    axes       per wale (column) and per course (row): the majority digit and its share; maj_switch = share of neighbouring lines whose majority digit differs (alternation), tm_group_mean = mean number
               of neighbouring strong-majority lines (share >= shell_features.MAJORITY) of the same digit (Pattern 1: 2, the two-wale TM axis of K's report), tm_group_alt = share of successive such groups of
               opposite digit; diagonals (both directions, length >= 4): mean majority share and share of strong-majority diagonals (the diagonal TM axis of K's report, Batch B Sample 5);
    settings   every numeric setting of a run (except seed and test hooks) as set_<key>: a new setting such as a tightness parameter is picked up as soon as runs carry it; one that is constant over all runs
               (or absent from all) is dropped;
    user       functions feat_* in scripts/shell_learning/shell_features_user.py (or --user-features PATH), see that file.
(b) Derived terms of the TOP_K base features most correlated with the target on the selection set: squares, products, both ratios (only when the denominator keeps one sign and |denominator| >= 1 % of its
    largest value: UNSOURCED guard against blow-ups) and logs (only of positive features); degree at most 2.
(c) see (a) user.
(d) Selection: stability selection (Meinshausen and Buehlmann 2010, J. R. Stat. Soc. B 72(4) 417-473): BOOTS random half-subsamples of the selection set; on each, a lasso path (FISTA, standardised
    columns) with a complexity-weighted penalty lambda x w_j (w = 1 base, 1.5 log, 2 square / product / ratio: UNSOURCED weights), stopped at the first lambda selecting >= Q terms; selection probability =
    share of subsamples that select a term. Stable = probability >= PI_THR (0.75, inside the 0.6-0.9 range recommended by that paper). Their bound on the expected number of falsely selected terms,
    q^2 / ((2 PI_THR - 1) p), is reported. The rule = ordinary least squares of the target on the stable terms (selection set), written as a formula in the original units.
(e) Honesty guards. The held-out set (runs whose sha1(run id + split seed) is in the lowest TEST_FRAC; a run never changes sides as the campaign grows) is never used for any choice: not for the top
    features, the standardisation, the selection or the fit. A permutation test repeats the WHOLE selection (top features, derived terms, stability selection) PERMS times with the targets shuffled
    within the selection set and counts the stable terms that appear by chance; FDR estimate = mean chance count / observed count. A rule is REPORTED only if all of these hold:
      enough_runs   >= MIN_RUNS (100, shell_rules.MIN_RUNS) finished runs carry the target;
      stable        at least one stable term;
      support       a term enters the rule only if it varies (differs from its most common value) in >= MIN_SUPPORT selection runs (20, UNSOURCED); stable terms without that support are removed
                    and listed under stable_terms_without_support; the guard fails if no supported stable term is left;
      beats_linear  on the held-out runs the rule's squared error is lower than that of the existing linear rules (shell_rules' lasso on shell_features.FEATURES, CV 1-SE rule, fitted on the same selection
                    set), with the 95 % bootstrap interval of the mean paired improvement above 0;
      permutation   FDR estimate <= MAX_FDR (0.25, UNSOURCED).
(f) Output analysis/rules_sr.json: per target the status ('rule', 'no rule' with the failed guards, 'waiting'), the readable formula, held-out errors (rule, linear rules, mean), stability of every term,
    permutation false-discovery estimate, and the label: learned from the shell, not confirmed on real fabric.
Usage: python3 shell_rules_sr.py [campaign] [--targets a,b] [--boots 100] [--perms 30] [--user-features PATH] [--out PATH] [--seed 0]   (lowers its own priority with os.nice(19))"""
import argparse, hashlib, importlib.util, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_S = os.path.dirname(HERE)
if os.path.basename(_S) == 'scripts': sys.path.extend(os.path.join(_S, _d) for _d in sorted(os.listdir(_S)) if _d not in ('legacy', '__pycache__') and os.path.isdir(os.path.join(_S, _d)))          # the other folders of scripts/ (not in a flat frozen copy)
import shell_features as sf
import shell_rules as rl

LABEL = rl.LABEL
MIN_RUNS = rl.MIN_RUNS
PARAMS = dict(test_frac=0.25, split_seed=0, top_k=8, boots=100, q=8, pi_thr=0.75, n_lams=25, lam_min_ratio=0.01, fista_iters=150, perms=30, min_support=20, max_fdr=0.25, ci_boot=2000, seed=0)
PARAM_SOURCES = dict(pi_thr='Meinshausen and Buehlmann 2010 recommend 0.6-0.9', q='UNSOURCED', top_k='UNSOURCED', boots='UNSOURCED (the paper uses 100)', min_support='UNSOURCED', max_fdr='UNSOURCED',
                     test_frac='UNSOURCED', complexity_weights='UNSOURCED', ratio_guard='UNSOURCED')
CPLX = dict(base=1.0, log=1.5, square=2.0, product=2.0, ratio=2.0)              # UNSOURCED complexity weights (penalty factors)
SKIP_SETTINGS = ('seed',)
USER_FILE = os.path.join(HERE, 'shell_features_user.py')


# ---------------------------------------------------------------- (a) feature generator
def motif_freq(T, h, w, prefix):
    B = (T == 1).astype(np.int64); R, C = B.shape
    if R < h or C < w: return {}
    code = np.zeros((R - h + 1, C - w + 1), np.int64)
    for i in range(h):
        for j in range(w): code = code * 2 + B[i:R - h + 1 + i, j:C - w + 1 + j]
    cnt = np.bincount(code.ravel(), minlength=2 ** (h * w)); tot = float(code.size); out = {}
    for k in range(2 ** (h * w)):
        d = ['1' if (k >> (h * w - 1 - t)) & 1 else '2' for t in range(h * w)]
        out['%s_%s' % (prefix, '/'.join(''.join(d[i * w:(i + 1) * w]) for i in range(h)))] = cnt[k] / tot
    return out


def run_stats(lines, prefix):
    """lines: 2-D array, one line per row. Run lengths of equal digits along each line."""
    L, V = [], []
    for v in lines:
        cut = np.flatnonzero(np.diff(v) != 0) + 1; st = np.r_[0, cut]; en = np.r_[cut, len(v)]; L.append(en - st); V.append(v[st])
    L = np.concatenate(L).astype(float); V = np.concatenate(V); wm = lambda m: float((L[m] ** 2).sum() / L[m].sum()) if m.any() else 0.0
    return {prefix + '_mean': float(L.mean()), prefix + '_wmean': wm(np.ones(len(L), bool)), prefix + '_max_rel': float(L.max() / lines.shape[1]), prefix + '_wmean_d1': wm(V == 1), prefix + '_wmean_d2': wm(V == 2)}


def labels4(T):
    R, C = T.shape; lab = -np.ones((R, C), np.int64); n = 0
    for r in range(R):
        for c in range(C):
            if lab[r, c] >= 0: continue
            v = T[r, c]; stack = [(r, c)]; lab[r, c] = n
            while stack:
                i, j = stack.pop()
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= a < R and 0 <= b < C and lab[a, b] < 0 and T[a, b] == v: lab[a, b] = n; stack.append((a, b))
            n += 1
    return lab, n


def section_stats(T):
    R, C = T.shape; lab, n = labels4(T); l = lab.ravel(); rr, cc = np.indices(T.shape); rr = rr.ravel(); cc = cc.ravel()
    size = np.bincount(l, minlength=n).astype(float)
    rmin = np.full(n, R); rmax = np.full(n, -1); cmin = np.full(n, C); cmax = np.full(n, -1)
    np.minimum.at(rmin, l, rr); np.maximum.at(rmax, l, rr); np.minimum.at(cmin, l, cc); np.maximum.at(cmax, l, cc)
    hgt = (rmax - rmin + 1).astype(float); wid = (cmax - cmin + 1).astype(float); cy = np.bincount(l, rr, n) / size; cx = np.bincount(l, cc, n) / size
    dist = np.hypot(cy - (R - 1) / 2.0, cx - (C - 1) / 2.0) / math.hypot(R / 2.0, C / 2.0); edge = (rmin == 0) | (cmin == 0) | (rmax == R - 1) | (cmax == C - 1); sw = size / size.sum()
    k = (T == 1); cen = np.zeros_like(k); cen[R // 4:R - R // 4, C // 4:C - C // 4] = True
    return dict(sec_width_mean=float(wid.mean()), sec_height_mean=float(hgt.mean()), sec_width_wmean=float((sw * wid).sum()), sec_height_wmean=float((sw * hgt).sum()),
                sec_log_aspect_wmean=float((sw * np.log(hgt / wid)).sum()), sec_centre_dist_wmean=float((sw * dist).sum()), sec_edge_stitch_frac=float(size[edge].sum() / size.sum()),
                centre_minus_edge_d1=float(k[cen].mean() - k[~cen].mean()) if cen.any() and (~cen).any() else 0.0)


def axis_stats(lines, prefix):
    s1 = (lines == 1).mean(1); share = np.maximum(s1, 1 - s1); maj = np.where(s1 > 0.5, 1, np.where(s1 < 0.5, 2, 0)); tm = (share >= sf.MAJORITY) & (maj > 0)
    pairs = [(maj[i], maj[i + 1]) for i in range(len(maj) - 1) if maj[i] and maj[i + 1]]
    groups, types, cur, prev = [], [], 0, None
    for i in range(len(maj)):
        if tm[i] and cur and maj[i] == prev: cur += 1
        else:
            if cur: groups.append(cur); types.append(prev)
            cur, prev = (1, maj[i]) if tm[i] else (0, None)
    if cur: groups.append(cur); types.append(prev)
    alt = [types[i] != types[i + 1] for i in range(len(types) - 1)]
    return {prefix + '_maj_switch': float(np.mean([a != b for a, b in pairs])) if pairs else 0.0, prefix + '_tm_group_mean': float(np.mean(groups)) if groups else 0.0,
            prefix + '_tm_group_alt': float(np.mean(alt)) if alt else 0.0}


def diag_stats(T, min_len=4):
    sh = []
    for M in (T, np.fliplr(T)):
        for o in range(-T.shape[0] + 1, T.shape[1]):
            d = np.diagonal(M, o)
            if len(d) >= min_len: s = (d == 1).mean(); sh.append(max(s, 1 - s))
    sh = np.array(sh) if sh else np.zeros(1)
    return dict(diag_majority_mean=float(sh.mean()), tm_diag_share=float((sh >= sf.MAJORITY).mean()))


def load_user(path):
    """(list of (name, function), SETTING_DEFAULTS, error text) from the plug-in file; a missing file gives no features."""
    if not path or not os.path.isfile(path): return [], {}, ''
    try:
        spec = importlib.util.spec_from_file_location('shell_features_user_plugin', path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    except Exception as e: return [], {}, 'plug-in %s not loaded: %s' % (path, e)
    fs = sorted((n[5:], f) for n, f in vars(mod).items() if n.startswith('feat_') and callable(f))
    return fs, dict(getattr(mod, 'SETTING_DEFAULTS', {}) or {}), ''


def pattern_features(s, user=()):
    """All generated features of one run's settings (dict). Non-finite or failing user features come back as NaN (dropped later)."""
    M, t = s['matrix'], int(s['tiles']); T = sf.tile(M, t); f = dict(sf.features(M, t, s['cpi'], s['wpi'])); f['ratio'] = s['cpi'] / s['wpi']
    f.update(motif_freq(T, 2, 2, 'm2x2')); f.update(motif_freq(T, 3, 1, 'm3x1')); f.update(motif_freq(T, 1, 3, 'm1x3'))
    f.update(run_stats(T.T, 'run_wale')); f.update(run_stats(T, 'run_course')); f.update(section_stats(T)); f.update(axis_stats(T.T, 'wale')); f.update(axis_stats(T, 'course')); f.update(diag_stats(T))
    ctx = dict(matrix=M, tiles=t, cpi=s['cpi'], wpi=s['wpi'], settings=s)
    for name, fn in user:
        try:
            v = fn(T, ctx)
            if isinstance(v, dict):
                for k, x in v.items(): f['user_%s_%s' % (name, k)] = float(x)
            else: f['user_' + name] = float(v)
        except Exception: f['user_' + name] = float('nan')
    return f


def setting_features(rows, user_defaults):
    """set_<key> for every numeric setting (bools, matrix, seed and test hooks excluded). Partly present settings are filled from user SETTING_DEFAULTS or shell_campaign.DEFAULTS, else dropped."""
    try:
        import shell_campaign as sc; dflt = dict(sc.DEFAULTS)
    except Exception: dflt = {}
    dflt.update(user_defaults); num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    keys = sorted({k for r in rows for k, v in r['settings'].items() if num(v) and k not in SKIP_SETTINGS and not k.startswith('stub')}); cols, dropped = {}, []
    for k in keys:
        have = [num(r['settings'].get(k)) for r in rows]
        if not all(have) and not num(dflt.get(k)): dropped.append('set_%s (present in %d of %d runs, no default)' % (k, sum(have), len(rows))); continue
        cols['set_' + k] = np.array([float(r['settings'][k]) if h else float(dflt[k]) for r, h in zip(rows, have)])
    return cols, dropped


def feature_table(rows, user_path=USER_FILE):
    """(X (n x p), names, notes) of all base features for the rows. Columns with any non-finite value are dropped and listed."""
    user, udef, uerr = load_user(user_path); per = [pattern_features(r['settings'], user) for r in rows]
    names = sorted({k for f in per for k in f}); cols = {k: np.array([f.get(k, float('nan')) for f in per], float) for k in names}
    sc_cols, dropped = setting_features(rows, udef); cols.update(sc_cols); out_n, out_c = [], []
    for k in sorted(cols):
        if not np.all(np.isfinite(cols[k])): dropped.append('%s (non-finite or failed for some runs)' % k); continue
        out_n.append(k); out_c.append(cols[k])
    X = np.stack(out_c, 1) if out_c else np.zeros((len(rows), 0))
    return X, out_n, dict(dropped_features=dropped, user_features=[n for n, _ in user], user_error=uerr)


# ---------------------------------------------------------------- (b) derived terms
def derived(Xall, names, top):
    cols, nm, cx = [], [], []
    for a in top:
        xa = Xall[:, a]; cols.append(xa ** 2); nm.append('%s^2' % names[a]); cx.append(CPLX['square'])
        if xa.min() > 0: cols.append(np.log(xa)); nm.append('log(%s)' % names[a]); cx.append(CPLX['log'])
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            cols.append(Xall[:, a] * Xall[:, b]); nm.append('%s * %s' % (names[a], names[b])); cx.append(CPLX['product'])
            for p, d in ((a, b), (b, a)):
                den = Xall[:, d]
                if (np.all(den > 0) or np.all(den < 0)) and np.abs(den).min() >= 0.01 * np.abs(den).max(): cols.append(Xall[:, p] / den); nm.append('%s / %s' % (names[p], names[d])); cx.append(CPLX['ratio'])
    return (np.stack(cols, 1) if cols else np.zeros((len(Xall), 0))), nm, cx


def clean(X, names, cx, sel):
    """Drop columns constant or non-finite on the selection rows, and exact duplicates (|corr| > 1 - 1e-9; the simpler / first is kept)."""
    keep = [j for j in range(X.shape[1]) if np.all(np.isfinite(X[:, j])) and X[sel, j].std() > 1e-12 * max(1.0, np.abs(X[sel, j]).max())]
    order = sorted(keep, key=lambda j: (cx[j], j)); Z = X[sel][:, order]; Z = (Z - Z.mean(0)) / Z.std(0); R = np.abs(Z.T @ Z) / len(Z); kept = []
    for k in range(len(order)):
        if not kept or R[k, kept].max() < 1 - 1e-9: kept.append(k)
    out = sorted(order[k] for k in kept); return X[:, out], [names[j] for j in out], [cx[j] for j in out]


# ---------------------------------------------------------------- (d) stability selection
def stability(Z, y, w, P, rng):
    """Selection probability of every column of Z (standardised) for y. Batched FISTA lasso over BOOTS half-subsamples, penalty lambda x w_j, first lambda with >= q terms."""
    n, p = Z.shape; m = n // 2; B = P['boots']; q = min(P['q'], p)
    idx = np.array([rng.choice(n, m, replace=False) for _ in range(B)]); Zb = Z[idx]; yb = y[idx]
    Zb = Zb - Zb.mean(1, keepdims=True); yb = yb - yb.mean(1, keepdims=True)
    G = np.matmul(Zb.transpose(0, 2, 1), Zb) / m; c = np.matmul(Zb.transpose(0, 2, 1), yb[..., None])[..., 0] / m
    L = np.linalg.eigvalsh(G)[:, -1] + 1e-12; lmax = (np.abs(c) / w[None, :]).max(1)
    b = np.zeros((B, p)); sel = np.zeros((B, p), bool); done = np.zeros(B, bool)
    for rk in np.geomspace(1.0, P['lam_min_ratio'], P['n_lams'])[1:]:
        thr = (lmax * rk / L)[:, None] * w[None, :]; x = b.copy(); v = b.copy(); t = 1.0
        for _ in range(P['fista_iters']):
            u = v - (np.matmul(G, v[..., None])[..., 0] - c) / L[:, None]; xn = np.sign(u) * np.maximum(np.abs(u) - thr, 0.0)
            tn = (1 + math.sqrt(1 + 4 * t * t)) / 2; v = xn + ((t - 1) / tn) * (xn - x); x, t = xn, tn
        b = x; nz = b != 0; hit = (~done) & (nz.sum(1) >= q); sel[hit] = nz[hit]; done |= hit
        if done.all(): break
    sel[~done] = (b != 0)[~done]
    return sel.mean(0), float(sel.sum(1).mean())


def select(Xb, nb, y_sel, sel, P, rng):
    """The whole selection stage on the selection rows only: top features, derived terms, cleaning, stability. Returns (X columns for ALL rows, names, complexities, probabilities, mean q)."""
    Zs = Xb[sel]; Zs = (Zs - Zs.mean(0)) / Zs.std(0); yc = y_sel - y_sel.mean(); cor = np.abs(Zs.T @ yc) / (len(yc) * (y_sel.std() or 1.0))
    top = [int(j) for j in np.argsort(-cor)[:P['top_k']]]; Xd, nd, cd = derived(Xb, nb, top)
    X, names, cx = clean(np.hstack([Xb, Xd]), nb + nd, [CPLX['base']] * len(nb) + cd, sel)
    Z = X[sel]; Z = (Z - Z.mean(0)) / Z.std(0); prob, qm = stability(Z, y_sel, np.array(cx), P, rng)
    return X, names, cx, prob, qm


# ---------------------------------------------------------------- (e) baseline and guards
def linear_baseline(F_sel, y_sel, F_test):
    """The existing linear rules (shell_rules.analyse_target's lasso: 5-fold CV, 1-SE rule) fitted on the selection rows only; predictions for the held-out rows."""
    Z, m, s = rl.standardise(F_sel); n = len(y_sel); g = max(np.abs(Z.T @ (y_sel - y_sel.mean())).max() / n, 1e-9); lams = np.geomspace(g, 1e-3 * g, 12); cvm, cvse = rl.cv_lasso(F_sel, y_sel, lams)
    i0 = int(np.argmin(cvm)); i1 = next(i for i in range(len(lams)) if cvm[i] <= cvm[i0] + cvse[i0]); b = rl.lasso(Z, y_sel, lams[i1])
    return (F_test - m) / s @ b + y_sel.mean()


def support(col):
    v = np.round(col, 12); u, c = np.unique(v, return_counts=True); return int(len(v) - c.max())


def is_test(run_id, frac, split_seed=0):
    h = hashlib.sha1(('%s|%d' % (run_id, split_seed)).encode()).hexdigest(); return int(h[:12], 16) / float(16 ** 12) < frac


def search_target(rows, target, X0, names0, P=None, min_runs=MIN_RUNS):
    P = dict(PARAMS, **(P or {})); rng = np.random.default_rng(P['seed'])
    ok = [i for i, r in enumerate(rows) if r['targets'].get(target) is not None and math.isfinite(r['targets'][target])]; out = dict(target=target, n_runs=len(ok), label=LABEL)
    if len(ok) < min_runs: out.update(status='waiting', note='fewer than %d finished runs with this descriptor: no rule is reported' % min_runs); return out
    rr = [rows[i] for i in ok]; Xb = X0[ok]; y = np.array([r['targets'][target] for r in rr], float); test = np.array([is_test(r['run'], P['test_frac'], P['split_seed']) for r in rr]); sel = ~test
    Xb, nb, _ = clean(Xb, list(names0), [CPLX['base']] * len(names0), sel)
    X, names, cx, prob, qm = select(Xb, nb, y[sel], sel, P, rng); p = len(names); stable_all = [j for j in range(p) if prob[j] >= P['pi_thr']]
    stable = [j for j in stable_all if support(X[sel][:, j]) >= P['min_support']]                 # the support guard is per term: an unsupported stable term is removed (and listed), never reported
    out['stable_terms_without_support'] = [dict(term=names[j], probability=float(prob[j]), supporting_runs=support(X[sel][:, j])) for j in stable_all if j not in stable]
    perm_counts = [int((select(Xb, nb, rng.permutation(y[sel]), sel, P, rng)[3] >= P['pi_thr']).sum()) for _ in range(P['perms'])]
    mean_chance = float(np.mean(perm_counts)) if perm_counts else None; fdr = (min(1.0, mean_chance / len(stable_all)) if stable_all else None) if mean_chance is not None else None
    topj = [int(j) for j in np.argsort(-prob)[:15]]
    out.update(n_selection=int(sel.sum()), n_heldout=int(test.sum()), n_candidate_terms=p, mean_terms_per_subsample=qm, mb_bound_expected_false_terms=float(qm ** 2 / ((2 * P['pi_thr'] - 1) * p)),
               top_selection_probabilities=[dict(term=names[j], probability=float(prob[j]), complexity=cx[j]) for j in topj],
               permutation=dict(perms=P['perms'], stable_terms_by_chance=perm_counts, mean_stable_terms_by_chance=mean_chance, share_of_perms_with_any=float(np.mean([c > 0 for c in perm_counts])) if perm_counts else None,
                                fdr_estimate=fdr))
    guards = dict(enough_runs=True, stable=bool(stable_all), support=bool(stable))
    yt = y[test]; mse_mean = float(np.mean((yt - y[sel].mean()) ** 2)); Fsf = np.array([[r['features'][k] for k in sf.FEATURES] for r in rr], float); pl = linear_baseline(Fsf[sel], y[sel], Fsf[test])
    mse_lin = float(np.mean((yt - pl) ** 2)); out.update(heldout_mse_mean=mse_mean, heldout_mse_linear_rules=mse_lin, heldout_r2_linear_rules=1 - mse_lin / mse_mean if mse_mean > 0 else None)
    if stable:
        A = np.c_[np.ones(int(sel.sum())), X[sel][:, stable]]; coef = np.linalg.lstsq(A, y[sel], rcond=None)[0]; pr_ = np.c_[np.ones(int(test.sum())), X[test][:, stable]] @ coef
        mse_rule = float(np.mean((yt - pr_) ** 2)); d = (yt - pl) ** 2 - (yt - pr_) ** 2; bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(P['ci_boot'])]); ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        sup = [support(X[sel][:, j]) for j in stable]
        guards.update(beats_linear=ci[0] > 0, permutation=fdr is not None and fdr <= P['max_fdr'])
        formula = '%s = %.4g' % (target, coef[0]) + ''.join(' %s %.4g * %s' % ('-' if c < 0 else '+', abs(c), '(%s)' % names[j] if ' ' in names[j] else names[j]) for c, j in zip(coef[1:], stable))
        out['rule'] = dict(formula=formula, intercept=float(coef[0]), terms=[dict(term=names[j], coefficient=float(c), stability=float(prob[j]), complexity=cx[j], supporting_runs=s_) for c, j, s_ in zip(coef[1:], stable, sup)],
                           complexity=float(sum(cx[j] for j in stable)), heldout_mse=mse_rule, heldout_r2=1 - mse_rule / mse_mean if mse_mean > 0 else None, improvement_over_linear_ci95=ci,
                           new_terms=[names[j] for j in stable if names[j] not in sf.FEATURES],
                           only_existing_features=all(names[j] in sf.FEATURES for j in stable))
        if out['rule']['only_existing_features']:          # it can beat the linear rules only by an unshrunk refit of the same features: said plainly, not hidden
            out['rule']['note'] = 'uses only features of the existing linear rules; it beats them on the held-out runs through an unshrunk least-squares refit, not through a new term'
    else: guards.update(beats_linear=False, permutation=False)
    failed = [k for k, v in guards.items() if not v]; out.update(guards=guards, status='rule' if not failed else 'no rule', failed_guards=failed)
    if failed and 'rule' in out: out['rule']['reported'] = False; out['candidate'] = out.pop('rule')            # kept for inspection, clearly not a reported rule
    return out


def run_rows(rows, targets=sf.TARGETS, P=None, user_path=USER_FILE, min_runs=MIN_RUNS):
    rows = [r for r in rows if not r.get('control')]; X0, names0, notes = feature_table(rows, user_path) if rows else (np.zeros((0, 0)), [], {})
    P_ = dict(PARAMS, **(P or {}))
    return dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S'), n_finished=len(rows), min_runs=min_runs, label=LABEL, params=P_, param_sources=PARAM_SOURCES, complexity_weights=CPLX, n_base_features=len(names0),
                base_features=names0, notes=notes, method='feature registry + degree-2 derived terms + complexity-weighted stability selection (lasso); OLS refit on stable terms; held-out by run id; permutation test',
                targets=[search_target(rows, t, X0, names0, P_, min_runs) for t in targets])


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0]); a.add_argument('campaign', nargs='?', default='shell1'); a.add_argument('--targets', default=','.join(sf.TARGETS))
    a.add_argument('--boots', type=int, default=PARAMS['boots']); a.add_argument('--perms', type=int, default=PARAMS['perms']); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--user-features', default=USER_FILE); a.add_argument('--out', default=None); g = a.parse_args()
    import shell_campaign as sc
    camp = sc.Camp(g.campaign)
    if not camp.exists(): sys.exit('no such shell campaign: %s' % g.campaign)
    try: os.nice(19)
    except (AttributeError, OSError): pass
    res = dict(campaign=camp.name); res.update(run_rows(sf.dataset(camp), [t for t in g.targets.split(',') if t], dict(boots=g.boots, perms=g.perms, seed=g.seed), g.user_features))
    path = g.out or os.path.join(camp.d, 'analysis', 'rules_sr.json'); os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(res, f, indent=1)
    os.replace(tmp, path)
    for t in res['targets']: print('%-16s %-8s %s' % (t['target'], t['status'], (t.get('rule') or {}).get('formula', ', '.join(t.get('failed_guards', [])) or t.get('note', ''))))
    print('written %s' % path)


if __name__ == '__main__':
    main()
