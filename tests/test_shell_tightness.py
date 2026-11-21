#!/usr/bin/env python3
"""Tests of scripts/shell_tightness.py against hand-computed values (pure Python, no numpy). Run from the repository root: python tests/test_shell_tightness.py"""
import json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import shell_tightness as st
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)

# hand values: Ne 12 -> tex 590.5 / 12 = 49.208333...; CPI 22 x WPI 16 = 352; tex x S = 17321.3333...; sqrt = 131.6105 (131.61^2 = 17321.19, + 0.1412 / 263.22 = 131.610537)
check('tex of Ne 12 is 590.5 / 12 = 49.2083', abs(st.tex(12) - 49.208333333) < 1e-8)
check('tex of Ne 590.5 is exactly 1', st.tex(590.5) == 1.0)
check('S = CPI x WPI: 22 x 16 = 352', st.stitch_density(22, 16) == 352.0)
check('TF_rel for Ne 12, CPI 22, WPI 16 is 131.6105 (hand value)', abs(st.tf_rel(22, 16) - 131.610537) < 1e-5, '%.6f' % st.tf_rel(22, 16))
check('TF_rel for tex 1 (Ne 590.5), CPI 10, WPI 10 is sqrt(100) = 10', abs(st.tf_rel(10, 10, 590.5) - 10.0) < 1e-12)
check('TF_rel scales as sqrt(S): CPI and WPI doubled gives twice the value', abs(st.tf_rel(44, 32) / st.tf_rel(22, 16) - 2.0) < 1e-12)
check('same S, other ratio (R 1.6 at 352: CPI 23.73, WPI 14.83) gives the same TF_rel', abs(st.tf_rel(math.sqrt(352 * 1.6), math.sqrt(352 / 1.6)) - st.tf_rel(22, 16)) < 1e-9)
check('missing or invalid CPI/WPI gives None, never 0', st.stitch_density(None, 16) is None and st.tf_rel(22, 0) is None and st.tf_rel('x', 16) is None and st.record({})['tf_rel'] is None)
r = st.record(dict(cpi=22.0, wpi=16.0, ne=9.0))
check('a shell run uses NE_FABRIC = 12 whatever its settings say; a yarn run passes its own Ne', r['ne'] == 12.0 and abs(st.record(dict(cpi=22, wpi=16), ne=590.5)['tf_rel'] - math.sqrt(352)) < 1e-12)
pol = json.load(open(os.path.join(ROOT, 'campaigns', 'shell_policy.json')))
# TF_c_rel = sqrt(49.208333) x CPI = 7.014866 x 22 = 154.32705; TF_w_rel = 7.014866 x 16 = 112.23786; their product 17321.333 = tex x S (hand values)
check('TF_c_rel for Ne 12, CPI 22 is 154.32705 and TF_w_rel for WPI 16 is 112.23786 (hand values)', abs(st.tf_c_rel(22) - 154.32705) < 1e-4 and abs(st.tf_w_rel(16) - 112.23786) < 1e-4, '%.5f %.5f' % (st.tf_c_rel(22), st.tf_w_rel(16)))
check('TF_c_rel x TF_w_rel = tex x CPI x WPI = 17321.3333 (so sqrt of it is TF_rel)', abs(st.tf_c_rel(22) * st.tf_w_rel(16) - 17321.33333) < 1e-3 and abs(math.sqrt(st.tf_c_rel(22) * st.tf_w_rel(16)) - st.tf_rel(22, 16)) < 1e-9)
check('at constant S = 352 the two values change with R in opposite directions (R 0.7 and R 1.6), TF_rel does not', st.tf_c_rel(math.sqrt(352 * 0.7)) < st.tf_c_rel(22) < st.tf_c_rel(math.sqrt(352 * 1.6)) and st.tf_w_rel(math.sqrt(352 / 0.7)) > st.tf_w_rel(16) > st.tf_w_rel(math.sqrt(352 / 1.6)) and abs(st.tf_rel(math.sqrt(352 * 0.7), math.sqrt(352 / 0.7)) - st.tf_rel(22, 16)) < 1e-9)
check('record() carries tf_c_rel and tf_w_rel; missing CPI or WPI gives None for that value only', st.record(dict(cpi=22, wpi=16))['tf_c_rel'] == st.tf_c_rel(22) and st.record(dict(cpi=22))['tf_w_rel'] is None and st.record(dict(cpi=22))['tf_c_rel'] is not None and st.tf_c_rel(0) is None and st.tf_c_rel('x') is None)
check('the labels say UNSOURCED for k_c and k_w', 'UNSOURCED' in st.LABEL_TF_C and 'UNSOURCED' in st.LABEL_TF_W)
check('shell_policy.json fixes CPI x WPI at 352 with a 0.5 % tolerance (so S is constant in shell1)', pol['ranges']['density_cpi_x_wpi'] == 352.0 and pol['ranges']['density_tolerance'] == 0.005)
info = st.campaign_info([352.0, 352.0 * 1.004, 351.0, 23.7318 * 14.8324], pol)      # each within 0.5 % of 352, as shell_policy.violations checks; the last is a rounded R 1.6 pair
check('with that policy, values within 0.5 % count as constant and carry the note', info['constant'] is True and info['constant_note'] == 'constant in this campaign (CPI x WPI fixed)')
check('outside the tolerance, or without a policy and unequal values: not constant, no note', st.campaign_info([352.0, 354.0], pol)['constant'] is False and st.campaign_info([352.0, 352.5])['constant'] is False and st.campaign_info([352.0, 352.5])['constant_note'] == '')
check('no defined value: constant is None', st.campaign_info([None, None])['constant'] is None)
check('the labels say UNSOURCED and nominal', 'UNSOURCED' in st.LABEL_TF and 'nominal' in st.LABEL_S)
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
