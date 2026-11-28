#!/usr/bin/env python3
"""shell_rules_model.py - the shell rules in two formats (K 2026-11-28): one for people (K and the judges) to read, one for the yarn model to read and use. numpy only.

Input: the result of shell_rules.run (analysis/rules.json): per target a sparse linear rule (lasso on standardised pattern features, cross-validated) and a depth-3 tree.
Output, next to rules.json in <campaign>/analysis/:
  rules_readable.md   plain sentences per fold property: what raises it, what lowers it, how much of the run-to-run variation the rule explains (cross-validated R^2), from how many runs,
                      plus the tree as if/then lines. Every rule carries the label 'learned from the shell, not confirmed on real fabric'.
  rules_model.json    schema shell_rules_model_v1: per target  intercept + sum_j per_unit_j * (feature_j - mean_j),  the feature definitions are shell_features.features(matrix, tiles, cpi, wpi)
                      (the same code the rules were fitted with). Only targets with status 'ok' are written. predict() below is the reader; the yarn pipeline imports it:
                        import shell_rules_model as srm; m = srm.load(path); srm.predict(m, matrix, tiles, cpi, wpi) -> {target: value}
What the yarn model does with the predicted fold (for example the curvature drive or the loop-length grade) is NOT decided here (tasks 47 and 59: K chooses which fields the fold drives).
Usage: python3 shell_rules_model.py <rules.json> [--matrix "1 2;2 1" --tiles 4 --cpi 22 --wpi 16]   (writes the two files next to rules.json; with --matrix also prints a prediction)"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import shell_features as sf

SCHEMA = 'shell_rules_model_v1'
# plain names of the fold properties (targets of shell_features.targets) and of the pattern features (shell_features docstring)
TARGET_TEXT = dict(thickness_rel='fold height (thickness / flat size)', shrink_x='shrink along the courses (x)', shrink_y='shrink along the wales (y)',
                   fold_turn_x_rad='fold turn along the courses (rad)', fold_turn_y_rad='fold turn along the wales (rad)', wave_amp_rel='wave amplitude (/ flat width)')
FEATURE_TEXT = dict(knit_fraction='share of knit loops', boundary_density='how broken up the pattern is (knit/purl borders per neighbour pair)', sections_per_100='number of one-type regions per 100 stitches',
                    mean_section_frac='mean region size', largest_section_frac='largest region size', wale_majority_mean='how one-type the wales are', course_majority_mean='how one-type the courses are',
                    tm_wale_share='share of wales that are axes (>= 75 % one type)', tm_course_share='share of courses that are axes', axis_direction='wale axes minus course axes',
                    dilution_wale='opposite loops inside wale axes', dilution_course='opposite loops inside course axes', log_ratio='ln(CPI / WPI)', aspect='rows / columns of the sheet', tiles='number of tiles')


def export(res, originals_only=None):
    """(model dict, readable markdown text) from a shell_rules result."""
    model = dict(schema=SCHEMA, campaign=res.get('campaign'), generated=res.get('generated'), n_finished=res.get('n_finished'), originals_only=res.get('originals_only', originals_only), label=res.get('label'),
                 features='shell_features.features(matrix, tiles, cpi, wpi)', formula='value = intercept + sum(per_unit * (feature - mean))', targets={})
    md = ['# Shell rules: %s' % res.get('campaign'), '', '%s. %d finished runs%s, generated %s.' % ((res.get('label') or '').capitalize(), res.get('n_finished') or 0,
          ' (GP-chosen runs left out)' if model['originals_only'] else '', res.get('generated')), '',
          'R2 = share of the run-to-run variation the rule predicts on runs it was not fitted on (cross-validated); 0 = no better than the average, 1 = perfect.', '']
    for t in res.get('targets', []):
        name = t['target']; md.append('## ' + TARGET_TEXT.get(name, name))
        if t.get('status') != 'ok': md += ['Not enough runs yet (%s runs).' % t.get('n_runs'), '']; continue
        L = t['lasso']; terms = [c for c in L['coefficients'] if 'mean' in c]
        model['targets'][name] = dict(text=TARGET_TEXT.get(name, name), intercept=L['intercept'], cv_r2=L['cv_r2'], n_runs=L['support'],
                                      terms=[dict(feature=c['feature'], per_unit=c['per_unit'], mean=c['mean'], stability=c['stability']) for c in terms])
        up = [c for c in terms if c['coefficient_std'] > 0]; dn = [c for c in terms if c['coefficient_std'] < 0]
        say = lambda cs: ', '.join('%s (%s%.2g per unit, stable in %d %% of refits)' % (FEATURE_TEXT.get(c['feature'], c['feature']), '+' if c['per_unit'] > 0 else '', c['per_unit'], round(100 * c['stability'])) for c in cs)
        md.append('Rule explains R2 = %.2f, from %d runs; average %.4g.' % (L['cv_r2'], L['support'], t.get('target_mean', L['intercept'])))
        md.append('- Goes **up** with: ' + (say(up) or 'nothing'))
        md.append('- Goes **down** with: ' + (say(dn) or 'nothing'))
        if L['cv_r2'] < 0.2: md.append('- Weak rule: it predicts little of the variation; do not rely on it.')
        tr = t.get('tree') or {}
        if tr.get('rules'): md += ['- Tree (R2 %.2f):' % tr.get('cv_r2', float('nan'))] + ['  - ' + r for r in tr['rules']]
        md.append('')
    return model, '\n'.join(md) + '\n'


def write(res, d, originals_only=None):
    model, md = export(res, originals_only)
    for name, text in (('rules_model.json', json.dumps(model, indent=1)), ('rules_readable.md', md)):
        p = os.path.join(d, name); tmp = p + '.tmp'
        with open(tmp, 'w', encoding='utf8') as f: f.write(text)
        os.replace(tmp, p)
    return model


def load(path):
    m = json.load(open(path, encoding='utf8'))
    if m.get('schema') != SCHEMA: raise ValueError('not a %s file: %s' % (SCHEMA, path))
    return m


def predict(model, matrix, tiles, cpi, wpi):
    """{target: predicted value} for a pattern, from the same feature code the rules were fitted with."""
    f = sf.features(matrix, tiles, cpi, wpi)
    return {t: r['intercept'] + sum(c['per_unit'] * (f[c['feature']] - c['mean']) for c in r['terms']) for t, r in model['targets'].items()}


def main():
    a = sys.argv[1:]
    if not a: sys.exit(__doc__)
    res = json.load(open(a[0], encoding='utf8')); model = write(res, os.path.dirname(os.path.abspath(a[0])))
    print('wrote rules_model.json (%d targets) and rules_readable.md' % len(model['targets']))
    if '--matrix' in a:
        g = lambda k, d: a[a.index(k) + 1] if k in a else d
        M = [[int(v) for v in row.split()] for row in g('--matrix', '').split(';')]
        for t, v in predict(model, M, int(g('--tiles', 4)), float(g('--cpi', 22)), float(g('--wpi', 16))).items(): print('%-18s %.4g' % (t, v))


if __name__ == '__main__':
    main()
