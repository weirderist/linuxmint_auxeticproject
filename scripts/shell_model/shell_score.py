#!/usr/bin/env python3
"""shell_score.py - similarity of a shell fold to a MEASURED reference, out of 100, only where such a reference exists. Pure Python; computed when a table is read, not inside the frozen worker.

The only measured reference is K's success mesh of Pattern 1 (tiles 3, CPI 22, WPI 16: the 13 Nov success), measured separately by K / Claude from his Blender mesh with this project's own wave definitions
(shell_metrics.describe): sheet about 26 x 58 mm in plan, about 14 mm thick, about 2.5 waves along the length, strongest in the middle (crumpled, disordered ends). These numbers are APPROXIMATE as given.
Score = 100 x the mean of five sub-scores in [0, 1]: plan_x, plan_y and thickness: 1 - |value - reference| / (tol x reference), floor 0 (tol = 0.25); waves: 1 - |wave_k - 2.5| / 1.0, floor 0; middle strongest:
min(1, mid_over_end). Equal weights, the tolerances and the form are UNSOURCED (display choices of this project). A run that is not the reference configuration has no score (None): for the other patterns the
real-fold text of K's report is stored beside the run for comparison by eye.
The mesh itself has not been analysed with these definitions yet (K's mesh file is needed): until then the reference values are the ones above."""

REFERENCE = dict(plan_x=26.0, plan_y=58.0, thickness=14.0, wave_k=2.5)          # mm, mm, mm, waves along the length
TOL = 0.25                                                                        # UNSOURCED relative tolerance
REFERENCE_SOURCES = ('seed:pattern1', 'control')                                  # Pattern 1, tiles 3, CPI 22, WPI 16 (the controls change only the iteration cap)


def is_reference_run(row):
    src = (row.get('source') or '')
    return (src == 'seed:pattern1' or src.startswith('control:')) and row.get('tiles') == 3 and row.get('cpi') == 22.0 and row.get('wpi') == 16.0


def parts(row):
    """Sub-scores in [0, 1] (None where a descriptor is missing)."""
    out = {}
    for k in ('plan_x', 'plan_y', 'thickness'):
        v = row.get(k); out[k] = None if v is None else max(0.0, 1.0 - abs(v - REFERENCE[k]) / (TOL * REFERENCE[k]))
    v = row.get('wave_k'); out['waves'] = None if v is None else max(0.0, 1.0 - abs(v - REFERENCE['wave_k']) / 1.0)
    v = row.get('mid_over_end'); out['middle_strongest'] = None if v is None else min(1.0, max(0.0, v))
    return out


def score(row):
    """Similarity out of 100, or None if the run is not the reference configuration or a descriptor is missing."""
    if row.get('status') != 'done' or not is_reference_run(row): return None
    p = parts(row)
    if any(v is None for v in p.values()): return None
    return round(100.0 * sum(p.values()) / len(p), 1)
