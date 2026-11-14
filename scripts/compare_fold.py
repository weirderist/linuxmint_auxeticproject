#!/usr/bin/env python3
"""compare_fold.py <result folder> - how much of the relaxed fold is the shell's. Reads shell.npz, relaxed_axis.csv, fab_axis.csv, meta.json, state.json.
Each relaxed yarn vertex is compared with the nearest shell grid vertex in (x, y) (approximate: the shell grid itself is folded in x, y). Prints z-ranges and
the rms / correlation of z, in yarn radii R."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
d = sys.argv[1]; R = json.load(open(os.path.join(d, 'meta.json')))['R']
rel = np.loadtxt(os.path.join(d, 'relaxed_axis.csv'), delimiter=',', skiprows=1)[:, 1:]; base = np.loadtxt(os.path.join(d, 'fab_axis.csv'), delimiter=',', skiprows=1)[:, 1:]
rng = lambda z: np.ptp(z)/R
print('z-range in R: baseline %.2f, relaxed %.2f' % (rng(base[:, 2]), rng(rel[:, 2])))
p = os.path.join(d, 'shell.npz')
if os.path.exists(p):
    X = np.load(p)['X'].reshape(-1, 3); print('shell z-range in R: %.2f' % rng(X[:, 2]))
    i = cKDTree(X[:, :2]).query(rel[:, :2])[1]; dz = rel[:, 2] - X[i, 2]
    print('relaxed z minus nearest shell z: rms %.2f R, correlation %.3f' % (np.sqrt((dz**2).mean())/R, np.corrcoef(rel[:, 2], X[i, 2])[0, 1]))
s = json.load(open(os.path.join(d, 'state.json'))); print('state', s['state'], 'end energy', (s.get('summary') or {}).get('end', {}).get('total'))
