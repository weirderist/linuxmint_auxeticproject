#!/usr/bin/env python3
"""shell_features.py - features of a stitch pattern and targets from the shell descriptors, shared by the rule job (shell_rules.py) and the proposer (shell_proposer.py). numpy only.

Features of the TILED matrix T (rows x cols of digits 1 and 2) and of the fabric:
  knit_fraction            share of digit 1
  boundary_density         unequal horizontal and vertical neighbour pairs / all neighbour pairs (how fragmented the pattern is)
  sections_per_100         4-connected regions of one loop type per 100 stitches;  mean_section_frac  mean region size / area;  largest_section_frac  largest region / area
  wale_majority_mean       mean over wales (columns) of max(share of digit 1, share of digit 2);  course_majority_mean  the same over courses (rows)
  tm_wale_share            share of wales whose majority share is >= MAJORITY (0.75, UNSOURCED threshold: a 'transfer medium axis' candidate in K's report is a wale or course with a strong majority of one loop type)
  tm_course_share          the same for courses
  axis_direction           tm_wale_share - tm_course_share (positive: wale-wise axes, negative: course-wise)
  dilution_wale / dilution_course   mean over those axes of (1 - majority): how many opposite-type loops sit inside the axis (0 if there is no axis)
  log_ratio                ln(CPI / WPI)
  aspect                   rows / cols of the tiled sheet;  tiles
Every threshold is UNSOURCED; the features are this project's own definitions (K's report: TM axes form where one loop type is in strong majority, wale-wise folds sharpest, course-wise blunt, diagonal weakest)."""
import math
import numpy as np

MAJORITY = 0.75      # UNSOURCED
FEATURES = ('knit_fraction', 'boundary_density', 'sections_per_100', 'mean_section_frac', 'largest_section_frac', 'wale_majority_mean', 'course_majority_mean', 'tm_wale_share', 'tm_course_share',
            'axis_direction', 'dilution_wale', 'dilution_course', 'log_ratio', 'aspect', 'tiles')
TARGETS = ('thickness_rel', 'shrink_x', 'shrink_y', 'fold_turn_x_rad', 'fold_turn_y_rad', 'wave_amp_rel')


def tile(M, n): return np.tile(np.array(M), (n, n))


def sections(T):
    """Sizes of the 4-connected regions of equal value (iterative flood fill)."""
    R, C = T.shape; seen = np.zeros((R, C), bool); sizes = []
    for r in range(R):
        for c in range(C):
            if seen[r, c]: continue
            v = T[r, c]; stack = [(r, c)]; seen[r, c] = True; n = 0
            while stack:
                i, j = stack.pop(); n += 1
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= a < R and 0 <= b < C and not seen[a, b] and T[a, b] == v: seen[a, b] = True; stack.append((a, b))
            sizes.append(n)
    return sizes


def features(matrix, tiles, cpi, wpi):
    T = tile(matrix, tiles); R, C = T.shape; k = (T == 1)
    f = float(k.mean()); pairs = (R * (C - 1) + (R - 1) * C); unequal = int((T[:, 1:] != T[:, :-1]).sum() + (T[1:, :] != T[:-1, :]).sum())
    sz = sections(T); wm = np.maximum(k.mean(0), 1 - k.mean(0)); cm = np.maximum(k.mean(1), 1 - k.mean(1)); tw = wm >= MAJORITY; tc = cm >= MAJORITY
    return dict(knit_fraction=f, boundary_density=unequal / float(pairs), sections_per_100=100.0 * len(sz) / (R * C), mean_section_frac=float(np.mean(sz)) / (R * C), largest_section_frac=max(sz) / float(R * C),
                wale_majority_mean=float(wm.mean()), course_majority_mean=float(cm.mean()), tm_wale_share=float(tw.mean()), tm_course_share=float(tc.mean()), axis_direction=float(tw.mean() - tc.mean()),
                dilution_wale=float((1 - wm[tw]).mean()) if tw.any() else 0.0, dilution_course=float((1 - cm[tc]).mean()) if tc.any() else 0.0, log_ratio=math.log(cpi / wpi), aspect=R / float(C), tiles=float(tiles))


def targets(m):
    """Scale-free targets from a run's descriptors (shell_metrics.describe): None where missing."""
    g = lambda k: m.get(k)
    ok = all(g(k) is not None for k in ('thickness', 'flat_x', 'flat_y'))
    return dict(thickness_rel=(g('thickness') / math.sqrt(g('flat_x') * g('flat_y'))) if ok else None, shrink_x=g('shrink_x'), shrink_y=g('shrink_y'), fold_turn_x_rad=g('fold_turn_x_rad'), fold_turn_y_rad=g('fold_turn_y_rad'),
                wave_amp_rel=(g('wave_amp_median_mm') / g('flat_x')) if g('wave_amp_median_mm') is not None and g('flat_x') else None)


def dataset(camp):
    """(rows) of finished runs of a campaign: dict(run, tier, source, settings, features, targets). Read only."""
    q = {x['id']: x for x in camp.queue_items()}; out = []
    for rid, x in camp.index_last().items():
        if x.get('status') != 'done' or rid not in q: continue
        s = q[rid]['settings']; m = x.get('metrics') or {}; t = targets(m)
        out.append(dict(run=rid, tier=q[rid].get('tier', 5), source=q[rid].get('source', ''), control=bool(q[rid].get('control')), settings=s, features=features(s['matrix'], s['tiles'], s['cpi'], s['wpi']), targets=t))
    return out
