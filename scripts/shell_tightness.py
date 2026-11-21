#!/usr/bin/env python3
"""shell_tightness.py - nominal tightness quantities of a run, computed ON READ from the CPI and WPI recorded in the run's settings. Display only: it is not a shell input, it changes no
calculation, and nothing is written into any run folder. Pure Python (math only).

Definitions
  tex      = 590.5 / Ne: conversion of the English cotton count Ne to tex (grams per 1000 m of yarn); the same conversion as scripts/loop_grade.py.
  NE_FABRIC = 12: K's fabric (project data, campaigns/shell_policy.json 'no_effect': the shell does not use Ne). Pure-shell runs always use this constant; a yarn-level campaign run uses its own
             recorded Ne (its build setting), because there Ne is a real input.
  S        = CPI x WPI: nominal stitch density, stitches per square inch, from the INPUT CPI and WPI (not a measured value; no relaxation state is implied by the shell).
  TF       = sqrt(tex) / l: the tightness factor (chapters.docx 3.1; l = stitch length, the yarn length in one loop). The yarn-level runs record it in meta.json from the as-built loop length
             (loop_grade.tightness_record); that record is not touched here.
  The shell has no loop length. Munden's empirical relation S = K_s / l^2 would give l from S, so that TF = sqrt(tex S / K_s). No value of K_s (or of K_c, K_w) and no statement of its units
  or relaxation state was found in the project's sources (search of 2026-11-21: the 26 papers in 'Sources and stuff/Project/Sources/Auxetic Work' and chapters.docx; Wadekar et al. and Bueno et al.
  cite Munden only qualitatively). Therefore no absolute TF is computed. Instead:
  TF_rel   = sqrt(tex x S): proportional to TF under Munden's relation (TF = TF_rel / sqrt(K_s), with S and l in consistent units); the scale constant is UNSOURCED. Unit: tex^0.5 per inch.
"""
import math

NE_FABRIC = 12.0                 # K's fabric (project data); not a published value
TEX_PER_NE = 590.5               # tex = 590.5 / Ne (as scripts/loop_grade.py)
LABEL_S = 'nominal stitch density S = CPI x WPI (stitches per square inch, from the input CPI and WPI)'
LABEL_TF = 'TF_rel = sqrt(tex x CPI x WPI) (tex^0.5 per inch): proportional to the tightness factor TF = sqrt(tex)/l; scale constant UNSOURCED'
NOTE = ('Display only: computed from the recorded CPI and WPI, never a shell input. TF = sqrt(tex)/l needs the loop length l; the shell has none, and no sourced constant relating l to CPI x WPI '
        "(Munden's K_s) was found in the project sources, so only the proportional value TF_rel is shown.")
CONSTANT_NOTE = 'constant in this campaign (CPI x WPI fixed)'


def tex(ne=NE_FABRIC):
    ne = float(ne)
    if not (math.isfinite(ne) and ne > 0): raise ValueError('Ne must be a positive number')
    return TEX_PER_NE / ne


def stitch_density(cpi, wpi):
    """S = CPI x WPI (stitches per square inch); None if either is missing or not a positive finite number."""
    try: c, w = float(cpi), float(wpi)
    except (TypeError, ValueError): return None
    return c * w if (math.isfinite(c) and math.isfinite(w) and c > 0 and w > 0) else None


def tf_rel(cpi, wpi, ne=NE_FABRIC):
    """sqrt(tex x CPI x WPI); None where S is undefined."""
    s = stitch_density(cpi, wpi); return None if s is None else math.sqrt(tex(ne) * s)


def record(settings, ne=None):
    """The display values of one run from its settings dict (cpi, wpi). ne None = NE_FABRIC (pure shell); a yarn run passes its own Ne."""
    n = NE_FABRIC if ne is None else float(ne); s = stitch_density((settings or {}).get('cpi'), (settings or {}).get('wpi'))
    return dict(stitch_density=s, tf_rel=None if s is None else math.sqrt(tex(n) * s), ne=n, tex=tex(n))


def constant(values, rel_tol=0.0):
    """True if all defined values agree within rel_tol (relative spread (max - min) / mean); None if there is no defined value."""
    v = [float(x) for x in values if x is not None]
    if not v: return None
    m = sum(v) / len(v); return (max(v) - min(v)) <= rel_tol * abs(m) + 1e-12 * abs(m)


def campaign_info(values, policy=None):
    """Summary for a campaign: the labels, the note, and whether S is constant over its runs. A policy with 'density_cpi_x_wpi' fixes S by design (tolerance 'density_tolerance');
    without one, S counts as constant only if all runs agree to round-off (1e-12 relative, a numerical tolerance)."""
    rg = (policy or {}).get('ranges') or {}; fixed = rg.get('density_cpi_x_wpi'); tol = float(rg.get('density_tolerance', 0.0)) if fixed is not None else 0.0
    v = [float(x) for x in values if x is not None]
    if fixed is not None: const = (all(abs(x / float(fixed) - 1) <= tol for x in v) if v else None)          # the policy's own rule (shell_policy.violations): every run within tol of the fixed density
    else: const = constant(v, 0.0)
    return dict(label_stitch_density=LABEL_S, label_tf_rel=LABEL_TF, note=NOTE, constant=const, constant_note=CONSTANT_NOTE if const else '', policy_density=fixed, policy_tolerance=tol if fixed is not None else None,
                ne_fabric=NE_FABRIC)
