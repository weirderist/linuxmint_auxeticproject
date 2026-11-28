#!/usr/bin/env python3
"""test_yarn_defaults.py - the standard yarn settings are ONE set (2026-11-28, K: "ensure consistency"): relax_job.py command-line defaults, the shell-fold path (shell_target.py,
shell_fold.py) and the Simulate page (webui/main.py CONFIG) all equal scripts/yarn_model/yarn_defaults.py STANDARD; no absolute contact stiffness is left in the fold path; the strain
report is right. No solver run. From the repository root: python tests/test_yarn_defaults.py"""
import argparse, os, re, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'scripts', 'yarn_model'), os.path.join(ROOT, 'scripts', 'shell_derived_target'), os.path.join(ROOT, 'webui')]
os.environ['AUX_AUTORESUME'] = '0'
import numpy as np
import yarn_defaults as YD
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)
S = YD.STANDARD

# relax_job: parse its own command line with only the required arguments and compare the defaults
import relax_job as rj
cap = {}
real_parse = argparse.ArgumentParser.parse_args
def grab(self, args=None, namespace=None):
    ns = real_parse(self, ['--folder', 'X', '--E', '500', '--mu', '0.49', '--k-contact', '1'], namespace); cap['g'] = ns; raise SystemExit(0)
argparse.ArgumentParser.parse_args = grab
try: rj.main()
except SystemExit: pass
argparse.ArgumentParser.parse_args = real_parse
g = cap['g']
check('relax_job defaults = STANDARD (perturb, outer, maxiter, ref-scale 1, stall rule on)', (g.perturb, g.outer, g.maxiter, g.ref_scale, g.stall_tol, g.stall_rounds, g.stall_move) ==
      (S['perturb'], S['outer'], S['maxiter'], S['ref_scale'], S['stall_tol'], S['stall_rounds'], S['stall_move']), str(vars(g))[:200])

import shell_fold as sfo, shell_target as sta, main
check('shell_fold defaults = STANDARD, contact as kc_ratio (no absolute k_contact)', sfo.RELAX_DEFAULTS == dict(E=S['E'], mu=S['mu'], kc_ratio=S['kc_ratio'], perturb=S['perturb'], curl=S['curl'], tension=S['tension_cN'],
      outer=S['outer'], maxiter=S['maxiter'], stall_tol=S['stall_tol']) and 'k_contact' not in sfo.RELAX_DEFAULTS)
src_fold = open(sfo.__file__, encoding='utf8').read(); src_tgt = open(sta.__file__, encoding='utf8').read()
check('shell_fold computes the contact from the build (YD.k_contact) and pre-clears the flat build (task 52 option b)', 'g.k_contact = YD.k_contact(' in src_fold and 'pre-clear (flat, contact only)' in src_fold and "A('--ref-scale', type=float, default=YD.STANDARD['ref_scale'])" in src_fold)
d = dict(zip(sta.DEFAULTS[0::2], sta.DEFAULTS[1::2]))
check('shell_target passes the STANDARD settings to relax_job and no absolute --k-contact in its defaults', '--k-contact' not in d and float(d['--ref-scale']) == S['ref_scale'] and float(d['--perturb']) == S['perturb'] and float(d['--curl']) == S['curl'] and int(d['--outer']) == S['outer'] and "'--k-contact', repr(kc)" in src_tgt)
C = main.CONFIG
check('Simulate page CONFIG = STANDARD', (C['yarn_E_MPa'], C['kc_ratio'], C['perturb'], C['outer'], C['maxiter'], C['stall_tol'], C['tension_cN']) == (S['E'], S['kc_ratio'], S['perturb'], S['outer'], S['maxiter'], S['stall_tol'], S['tension_cN']))
B = main.SimBody(E=500, mu=0.49, kc_ratio=30)
check('Simulate API defaults (fields omitted in a POST) = STANDARD: ref-scale, stall rule, outer, maxiter, perturb', (B.ref_scale, B.stall_tol, B.outer, B.maxiter, B.perturb) == (S['ref_scale'], S['stall_tol'], S['outer'], S['maxiter'], S['perturb']))
check('no absolute contact constant of 40 is left in the fold path', not re.search(r"k_contact\s*=\s*40|'--k-contact',\s*'40'", src_fold + src_tgt))

# contact rule and strain report
R = 0.1309347931912187
check('k_contact = 30 EA/R = 6170 N/mm for Ne 12 (E 500 MPa)', abs(YD.k_contact(500.0, R) - 30 * 500 * np.pi * R) < 1e-9 and abs(YD.k_contact(500.0, R) - 6170) < 5, '%.1f' % YD.k_contact(500.0, R))
y = types.SimpleNamespace(X=np.array([[0, 0, 0], [1.01, 0, 0], [2.0, 0, 0]], float), l0=np.array([1.0, 1.0]))
m, hi, lo = YD.max_strain([y])
check('max_strain: +1 % stretch and -1 % compression found, largest |strain| 1 %', abs(hi - 0.01) < 1e-12 and abs(lo + 0.01) < 1e-12 and abs(m - 0.01) < 1e-12, (m, hi, lo))
src_rj = open(rj.__file__, encoding='utf8').read()
check('relax_job reports the largest strain every round and at the end, with the flag', 'max_strain=YD.max_strain(sim.yarns)[0]' in src_rj and "d['summary']['yarn_strain']" in src_rj and 'VISIBLE STRETCH' in src_rj)
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)
