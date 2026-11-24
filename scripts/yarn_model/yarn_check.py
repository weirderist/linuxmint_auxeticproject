#!/usr/bin/env python3
"""yarn_check.py - virtual tensile and bending test of the yarn model, to run BEFORE a simulation. It shows what the solver's energy does with the yarn properties you entered,
so you can compare it with your own tensile / bending measurements. It does not test the knitted fabric.
Usage: python3 yarn_check.py --R <yarn radius, mm> --E <modulus N/mm^2> [--ea N] [--ei N_mm2] [--stretch-law 'strain:cN,...'] [--folder results/<id>]   (same meaning as in relax_job.py)
 1. Tensile: a straight chain of segments is stretched uniformly; the force is the derivative of the solver's stretch energy with respect to the chain length (central difference),
    printed in cN against the force the law says. They agree if the energy and the law are consistent (a check of the code, not of your yarn).
 2. Bending: a circular arc of chosen radius; the bending energy gives EI_eff = 2 E_bend / (kappa^2 L), compared with the EI you entered (differs only by discretisation).
 3. The ratio of the entered EI to a solid round rod of the same EA and radius (EA R^2 / 4). Spun yarns are far below 1 (0.0032 for acrylic, Ding et al.; Sperl et al. 2022 explore 1e-3..10).
Units: N and mm. Needs JAX, like relax_job.py.
"""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relax_state as rs, relax_solver as sv, relax_job as rj

def chain(n, xs, ys=None):
    """Single-yarn argument set for sv._energy: n vertices at positions xs (n,3). Contact and shell terms are zero."""
    seg = np.array([(i, i + 1) for i in range(n - 1)]); tri = np.array([(i - 1, i, i + 1) for i in range(1, n - 1)])
    l0 = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    return dict(seg=seg, tri=tri, l0=l0, lbar=np.array([0.5*(l0[i - 1] + l0[i]) for i in range(1, n - 1)]), kb0=np.zeros((n - 2, 3)),
                pairs=np.zeros((1, 2), int), pm=np.zeros(1), cell=np.zeros(n, int), tgt=np.zeros((1, 3)), cnt=np.ones(1))

def energy(x, c, l0, EA, EI):
    return float(sv._energy(np.asarray(x, float).ravel(), l0, c['seg'], c['tri'], c['lbar'], c['kb0'], c['pairs'], c['pm'], EA, EI, 0.0, 0.0, c['cell'], c['tgt'], c['cnt'], 0.0))

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--R', type=float); a.add_argument('--folder'); a.add_argument('--E', type=float, required=True)
    a.add_argument('--ea', type=float, default=0.0); a.add_argument('--ei', type=float, default=0.0); a.add_argument('--stretch-law', default='')
    a.add_argument('--stretch-scale', type=float, default=1.0); a.add_argument('--bend-scale', type=float, default=1.0)
    a.add_argument('--bend-radius', type=float, default=0.0, help='arc radius for the bending test, mm (default 20 R)')
    g = a.parse_args(); g.stretch_law = g.stretch_law or ''
    R = g.R if g.R else json.load(open(os.path.join(g.folder, 'meta.json')))['R'] if g.folder else None
    if not R: sys.exit('give --R (yarn radius, mm) or --folder')
    try: EA, EI = rj.yarn_inputs(g, R)
    except ValueError as e: sys.exit(str(e))
    print('yarn: R %.4g mm, EA %.5g N, EI %.5g N mm^2, stretch law %s' % (R, EA, EI, 'measured' if rs.LAW else 'Hooke'))
    print('EI against a solid round rod of the same EA and radius (EA R^2/4 = %.5g N mm^2): %.4g' % (EA*R*R/4, EI/(EA*R*R/4)))
    n = 12; l = 1.0; xs = np.c_[np.arange(n)*l, np.zeros(n), np.zeros(n)]; c = chain(n, xs); L0 = (n - 1)*l
    print('\ntensile test (%d segments of %.2g mm, uniform strain)\n  strain    force from the energy (cN)    force from the law (cN)' % (n - 1, l))
    top = (rj.parse_law(g.stretch_law)[-1][0]*1.5) if g.stretch_law else 0.08
    worst = 0.0
    for eps in np.linspace(top/8, top, 8):
        h = 1e-5*L0; f = []
        for d in (-h, h):
            x = xs*(1.0 + eps + d/L0); f.append(energy(x, c, c['l0'], EA, 0.0))
        F = (f[1] - f[0])/(2*h); Fl = float(rs.tension(1.0 + eps, 1.0, EA)); worst = max(worst, abs(F - Fl)/max(abs(Fl), 1e-30))
        print('  %.4f    %12.4f                  %12.4f' % (eps, 100*F, 100*Fl))
    print('largest relative difference %.2e (should be ~1e-6 or smaller)' % worst)
    rho = g.bend_radius or 20*R; phi = 0.05; m = 40; th = (np.arange(m) - (m - 1)/2)*phi
    xs = np.c_[rho*np.sin(th), rho*(1 - np.cos(th)), np.zeros(m)]; c = chain(m, xs); Eb = energy(xs, c, c['l0'], 0.0, EI); Larc = c['lbar'].sum()
    print('\nbending test (arc of radius %.4g mm, %.3g rad per vertex): E_bend %.5g N mm, EI_eff = 2 E L^-1 kappa^-2 = %.5g N mm^2 = %.4f x the entered EI' % (rho, phi, Eb, 2*Eb/(Larc/rho**2), 2*Eb/(Larc/rho**2)/EI))

if __name__ == '__main__': main()
