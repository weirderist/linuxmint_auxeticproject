#!/usr/bin/env python3
"""fold_metric.py - how curved is a fabric? Least-squares fit of z = a x^2 + b y^2 + c x y + d x + e y + f to all yarn-axis vertices of a CSV (course,x,y,z).
Prints the curvatures kx = 2a (along the courses, x) and ky = 2b (along the wales, y) in 1/mm, and the rise of the sheet at the middle of each edge relative to the middle
(the sag a rolled fabric would show), in mm and in units of the yarn diameter. A flat fabric gives ~0. Usage: python3 fold_metric.py file.csv [file2.csv ...]"""
import sys, numpy as np
def fit(P):
    x = P[:, 0] - P[:, 0].mean(); y = P[:, 1] - P[:, 1].mean(); z = P[:, 2]
    A = np.c_[x*x, y*y, x*y, x, y, np.ones_like(x)]; c = np.linalg.lstsq(A, z, rcond=None)[0]
    return dict(kx=2*c[0], ky=2*c[1], kxy=c[2], tilt=(c[3], c[4]), Lx=float(np.ptp(P[:, 0])), Ly=float(np.ptp(P[:, 1])))
if __name__ == '__main__':
    for f in sys.argv[1:]:
        P = np.loadtxt(f, delimiter=',', skiprows=1)[:, 1:]; r = fit(P)
        print('%s: kx %+.4f /mm, ky %+.4f /mm, twist %+.4f; edge rise x-direction %+.3f mm, y-direction %+.3f mm (sheet %.1f x %.1f mm)' % (f, r['kx'], r['ky'], r['kxy'], r['kx']*(r['Lx']/2)**2/2, r['ky']*(r['Ly']/2)**2/2, r['Lx'], r['Ly']))
