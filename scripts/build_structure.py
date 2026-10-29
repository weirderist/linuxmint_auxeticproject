#!/usr/bin/env python3
"""Phase 1: node structure + per-course cubic spline for the nine-node loop.
Inputs: Ne, CPI, WPI, matrix (1 = front loop, 2 = back loop). No sweeping here.
Usage: python3 build_structure.py --ne 10 --cpi 14 --wpi 18 --matrix "1 2 1/2 1 2" --out outdir
       (rows separated by '/', row 1 = course 1; --tiles 25 = 25 x 25 repeats)
"""
import argparse, itertools, json, math, os
import numpy as np
from scipy.interpolate import CubicSpline

def params(ne, cpi, wpi):
    d = 25.4 / (28.0 * math.sqrt(ne))      # cotton diameter relation (1/(28 sqrt Ne) inch)
    c = 25.4 / cpi                          # K's mapping: c = 25.4/CPI
    w = 25.4 / wpi                          # K's mapping: w = 25.4/WPI
    L = c                                   # standard: L = c (L kept in the node formulae)
    return d, c, w, L

def base_nodes(c, w, d, L, t):
    """Nine nodes P1..P9 of one loop, local coordinates. t=1 front, t=2 back (z inverted)."""
    P = np.array([
        [0,               0,           d/2],
        [(c+2*d)/4,       (L-w)/2,     0],
        [c/4,             L/2,        -d/2],
        [(c-2*d)/4,       (L+w)/2,     0],
        [c/2,             L,           d/2],
        [(3*c+2*d)/4,     (L+w)/2,     0],
        [3*c/4,           L/2,        -d/2],
        [(3*c-2*d)/4,     (L-w)/2,     0],
        [c,               0,           d/2]], float)
    if t == 2:
        P[:, 2] *= -1
    return P

def variants(c, w, d, L):
    """All 8 structures keyed (prev, current, next). Only P1 and P9 depend on neighbours."""
    V = {}
    for p, t, n in itertools.product((1, 2), repeat=3):
        P = base_nodes(c, w, d, L, t)
        if p != t:   # P1 = midpoint of previous loop's P9 (translated -c in x) and this P1
            prevP9 = base_nodes(c, w, d, L, p)[8] - np.array([c, 0, 0])
            P[0] = (prevP9 + P[0]) / 2
        if n != t:   # P9 = midpoint of this P9 and next loop's P1 (translated +c in x)
            nextP1 = base_nodes(c, w, d, L, n)[0] + np.array([c, 0, 0])
            P[8] = (P[8] + nextP1) / 2
        V[(p, t, n)] = P
    return V

def build_unit(M, c, V):
    """Matrix of nodes for one repeat unit: unit[r][i] = 9x3 array (wrap used only to pick the variant)."""
    R, C = len(M), len(M[0])
    unit = []
    for r in range(R):
        row = []
        for i in range(C):
            key = (M[r][(i-1) % C], M[r][i], M[r][(i+1) % C])
            P = V[key].copy()
            P[:, 0] += i * c
            P[:, 1] += r * c / 2          # course r starts at y = r*c/2
            row.append(P)
        unit.append(row)
    return unit

def tile_courses(unit, c, tiles_x, tiles_y):
    """Repeat the unit; return list of courses, each a list of 9x3 loop arrays."""
    R, C = len(unit), len(unit[0])
    courses = []
    for ty in range(tiles_y):
        for r in range(R):
            loops = []
            for tx in range(tiles_x):
                for i in range(C):
                    P = unit[r][i].copy()
                    P[:, 0] += tx * C * c
                    P[:, 1] += ty * R * c / 2
                    loops.append(P)
            courses.append(loops)
    return courses

def merge_course(loops, tol):
    """Concatenate loop nodes; a node coincident with the previous node is overwritten (kept once)."""
    nodes = [loops[0][0]]
    merged = 0
    for k, P in enumerate(loops):
        for j in range(9):
            if k > 0 and j == 0:
                if np.linalg.norm(P[0] - nodes[-1]) > tol:
                    raise ValueError("loop %d: P1 not coincident with previous P9" % k)
                merged += 1
                continue
            if j == 0:
                continue
            nodes.append(P[j])
    return np.array(nodes), merged

def spline_course(nodes, per_seg=16):
    """Cubic spline through this course's nodes only; chord-length parameter, natural ends."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))])
    cs = CubicSpline(s, nodes, bc_type='natural')
    ss = np.concatenate([np.linspace(s[i], s[i+1], per_seg, endpoint=False) for i in range(len(s)-1)] + [[s[-1]]])
    return s, cs, ss, cs(ss)

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--ne', type=float, required=True)
    a.add_argument('--cpi', type=float, required=True)
    a.add_argument('--wpi', type=float, required=True)
    a.add_argument('--matrix', required=True)
    a.add_argument('--tiles', type=int, default=25)
    a.add_argument('--per_seg', type=int, default=16)
    a.add_argument('--out', default='out')
    g = a.parse_args()
    M = [[int(ch) for ch in row.replace(' ', '')] for row in g.matrix.split('/')]
    assert len({len(r) for r in M}) == 1 and all(v in (1, 2) for r in M for v in r), "matrix must be rectangular, digits 1/2"
    d, c, w, L = params(g.ne, g.cpi, g.wpi)
    V = variants(c, w, d, L)
    unit = build_unit(M, c, V)
    courses = tile_courses(unit, c, g.tiles, g.tiles)
    os.makedirs(g.out, exist_ok=True)
    tol = 1e-9 * c
    rows, nrows, total_merged, lens = [], [], 0, []
    for ci, loops in enumerate(courses):
        nodes, m = merge_course(loops, tol)
        total_merged += m
        s, cs, ss, pts = spline_course(nodes, g.per_seg)
        for k, p in enumerate(nodes): nrows.append((ci, k, *p))
        for p in pts: rows.append((ci, *p))
        lens.append(s[-1] / len(loops))
    np.savetxt(os.path.join(g.out, 'fab_nodes.csv'), nrows, delimiter=',', header='course,idx,x,y,z', comments='', fmt=['%d', '%d', '%.6f', '%.6f', '%.6f'])
    np.savetxt(os.path.join(g.out, 'fab_axis.csv'), rows, delimiter=',', header='course,x,y,z', comments='', fmt=['%d', '%.6f', '%.6f', '%.6f'])
    meta = dict(ne=g.ne, cpi=g.cpi, wpi=g.wpi, d=d, c=c, w=w, L=L, matrix=M, tiles=g.tiles,
                courses=len(courses), loops_per_course=len(courses[0]), merged_junctions=total_merged,
                mean_chord_length_per_loop_mm=float(np.mean(lens)))
    json.dump(meta, open(os.path.join(g.out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))

if __name__ == '__main__':
    main()
