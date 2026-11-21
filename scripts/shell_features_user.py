#!/usr/bin/env python3
"""shell_features_user.py - K's own pattern features for the open-ended rule search (scripts/shell_rules_sr.py). Picked up automatically; nothing else needs changing.

How to add a feature: write a function whose name starts with feat_ . It receives
    T    the TILED stitch matrix as a numpy integer array (rows = courses, columns = wales; digit 1 or 2, as in the campaign matrix),
    ctx  a dict with 'matrix' (the untiled matrix, list of lists), 'tiles', 'cpi', 'wpi' and 'settings' (the run's full settings dict),
and returns ONE number (the feature is named user_<name without feat_>) or a dict of numbers (named user_<name>_<key>).
A feature that raises an error or returns a non-finite number for any run is dropped and listed under 'dropped_features' in analysis/rules_sr.json; it never stops the search.
A feature that is the same for every run is dropped automatically (it cannot explain anything).

SETTING_DEFAULTS: a numeric setting that only some runs carry (for example a setting added after a campaign started) is filled with the value given here for the runs that lack it;
without an entry here (or in shell_campaign.DEFAULTS) such a setting is dropped and listed.

The two examples below are switched off. To switch one on, remove the leading '# ' of its lines.
"""

SETTING_DEFAULTS = {}

# Example: share of courses that are entirely digit 2.
# def feat_purl_rows(T, ctx):
#     return float((T == 2).all(axis=1).mean())

# Prepared but OFF: the nominal tightness index TF_rel = sqrt(tex x CPI x WPI), Ne fixed at 12 (scripts/shell_tightness.py; proportional to the tightness factor, scale constant UNSOURCED).
# It is deliberately not a feature now: in shell1 CPI x WPI is fixed at 352 (campaigns/shell_policy.json), so it is constant, would be dropped as constant anyway, and it carries no information
# beyond CPI and WPI. Switch it on only for a campaign in which CPI x WPI varies. The existing linear rules (shell_rules.py, fixed feature list) are not affected either way; rules_sr.json then
# gains one candidate feature, user_tf_rel.
# def feat_tf_rel(T, ctx):
#     import shell_tightness
#     return shell_tightness.tf_rel(ctx['cpi'], ctx['wpi'])

# Also prepared but OFF: TF_c_rel = sqrt(tex) x CPI and TF_w_rel = sqrt(tex) x WPI (scripts/shell_tightness.py). Unlike TF_rel these change from run to run in shell1, but at constant CPI x WPI each is a fixed
# function of CPI / WPI alone (CPI = sqrt(352 R), WPI = sqrt(352 / R)), and log_ratio = ln(CPI / WPI) is already a feature, so they add no information to the search; switching them on only changes the names
# in which a rule can be written. They are most useful in a campaign in which CPI x WPI varies.
# def feat_tf_c_rel(T, ctx):
#     import shell_tightness
#     return shell_tightness.tf_c_rel(ctx['cpi'])
# def feat_tf_w_rel(T, ctx):
#     import shell_tightness
#     return shell_tightness.tf_w_rel(ctx['wpi'])
