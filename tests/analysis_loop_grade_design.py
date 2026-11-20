"""REFERENCE PROTOTYPE (independent, SciPy based; no simulation) of the sectioned starting loop-length field of relax_job --loop-grade. The production code is scripts/loop_grade.py (numpy only);
tests/test_loop_grade.py checks that it reproduces this prototype exactly. The prototype weights every loop equally; the production code weights by rest length per solver segment.
Run:  python tests/analysis_loop_grade_design.py        (needs numpy and scipy)

Every constant below is UNSOURCED (A, S_MIN, the linear strength mapping, the linear fall-off, weights W_B and W_T).
Definitions (one loop = one stitch, matrix[course][wale], first row = bottom course):
  section        4-connected region of one loop type. The fabric edge counts as a section boundary.
  boundary term  G_b = 1 at a loop next to its section boundary, falling linearly with the (Euclidean) distance in loops to the farthest loop of that section (0 there).
                 Loops of two sections that share a boundary both get G_b = 1 next to it, so the shared boundary is a two-loop-wide ridge.
  TM strength    for every wale (and every course): f = fraction of its majority type, s = max(0, 2 (f - 0.5)): 0 at f = 0.5, 1 at f = 1. A line is an axis if s >= S_MIN.
                 Consecutive axis lines with the same majority type form one band (strength = the mean of s over the band).
  TM term        G_t = sum over bands of  s_band * (1 - delta/delta_max_S),  delta = distance in loops across the band, delta_max_S = the largest delta in the loop's section S;
                 only loops whose section contains loops of the band's lines get it (a type-2 band acts on the type-2 section it runs through, not on type-1 sections).
  field          L = 1 + A (G - mean of G over the section), G = W_B G_b + W_T G_t: the total of every section is exactly the uniform start's, and a uniform term cancels exactly.
                 So A is the length difference between the target loops (G = 1) and the farthest loops (G = 0) as a fraction of the section's mean loop length.
                 mode 'boundary': W_T = 0;  'tm': W_B = 0;  'both': both 1.
"""
import sys
import numpy as np
from scipy import ndimage

A, S_MIN = 0.02, 0.2


def sections(M):
    lab = np.zeros(M.shape, int); n = 0
    for t in np.unique(M):
        l, k = ndimage.label(M == t); lab[l > 0] = l[l > 0] + n; n += k
    return lab, n

def strengths(M):
    """(strength per wale, majority type per wale, strength per course, majority type per course)."""
    def one(axis):
        out_s, out_t = [], []
        for i in range(M.shape[1 - axis]):
            line = M[:, i] if axis == 0 else M[i, :]
            f1 = (line == 1).mean(); f = max(f1, 1 - f1); out_s.append(max(0.0, 2 * (f - 0.5))); out_t.append(1 if f1 >= 0.5 else 2)
        return np.array(out_s), np.array(out_t)
    sw, tw = one(0); sc, tc = one(1)
    return sw, tw, sc, tc

def bands(s, t):
    """Consecutive axis lines (s >= S_MIN) of the same majority type -> list of (first, last, type, mean strength)."""
    out, i = [], 0
    while i < len(s):
        if s[i] >= S_MIN:
            j = i
            while j + 1 < len(s) and s[j + 1] >= S_MIN and t[j + 1] == t[i]: j += 1
            out.append((i, j, t[i], float(s[i:j + 1].mean()))); i = j + 1
        else: i += 1
    return out

def field(M, mode="both", a=A):
    lab, ns = sections(M); G_b = np.zeros(M.shape); G_t = np.zeros(M.shape)
    for k in range(1, ns + 1):
        mask = lab == k
        d = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]               # distance to the nearest loop outside the section (or the fabric edge), 1 next to it
        dm = d[mask].max(); G_b[mask] = 1.0 if dm <= 1 else 1.0 - (d[mask] - 1.0) / (dm - 1.0)
    sw, tw, sc, tc = strengths(M); yy, xx = np.mgrid[0:M.shape[0], 0:M.shape[1]]
    for (i, j, typ, s) in bands(sw, tw):                                              # wale bands: distance across the wales
        delta = np.maximum(0, np.maximum(i - xx, xx - j)).astype(float)
        for k in range(1, ns + 1):
            mask = lab == k
            if not (mask & (M == typ) & (xx >= i) & (xx <= j)).any(): continue       # this section does not contain loops of the band
            dm = delta[mask].max(); G_t[mask] += s * (1.0 if dm == 0 else 1.0 - delta[mask] / dm)
    for (i, j, typ, s) in bands(sc, tc):                                              # course bands: distance across the courses
        delta = np.maximum(0, np.maximum(i - yy, yy - j)).astype(float)
        for k in range(1, ns + 1):
            mask = lab == k
            if not (mask & (M == typ) & (yy >= i) & (yy <= j)).any(): continue
            dm = delta[mask].max(); G_t[mask] += s * (1.0 if dm == 0 else 1.0 - delta[mask] / dm)
    wb, wt = {"boundary": (1, 0), "tm": (0, 1), "both": (1, 1)}[mode]
    G = wb * G_b + wt * G_t; L = np.ones(M.shape)
    for k in range(1, ns + 1): L[lab == k] = 1.0 + a * (G[lab == k] - G[lab == k].mean())      # additive normalisation: the mean of every section is exactly 1 (loops weighted equally in this prototype), and a term that is uniform over a section drops out
    return L, lab, ns

def show(M, L, title, flip=True):
    print(title); rows = range(M.shape[0] - 1, -1, -1) if flip else range(M.shape[0])
    print("      wale " + "".join("%5d" % c for c in range(M.shape[1])))
    for r in rows: print("c%2d %s  " % (r, "".join(".#"[v - 1] for v in M[r])) + "".join("%+5.1f" % (100 * (x - 1)) for x in L[r]))
    print("    (% longer / shorter than the uniform start; type 1 = '.', type 2 = '#'; top row = last course)\n")

def pattern1():
    txt = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "campaigns", "pattern1_matrix.txt")).read().strip()
    return np.array([[int(c) for c in row.replace(" ", "")] for row in txt.split("/")])

if __name__ == "__main__":
    J = np.ones((8, 8), int)
    print("=" * 100 + "\nSINGLE JERSEY 8 x 8, a = %g (UNSOURCED): one section = the whole fabric, the boundary is the fabric edge\n" % A + "=" * 100)
    for mode in ("boundary", "both"):
        L, lab, ns = field(J, mode); show(J, L, "mode %s (sections: %d; mean %.6f; centre %+.2f %%, edge middle %+.2f %%, corner %+.2f %%)" % (mode, ns, L.mean(), 100 * (L[3, 3] - 1), 100 * (L[0, 3] - 1), 100 * (L[0, 0] - 1)))
    sw, tw, sc, tc = strengths(J); print("single jersey TM strength: every wale s = %.1f, every course s = %.1f, all one majority type -> ONE band covering the fabric -> the TM term is uniform and cancels in the section normalisation\n" % (sw[0], sc[0]))
    M = pattern1()
    print("=" * 100 + "\nPATTERN 1 (campaigns/pattern1_matrix.txt), %d courses x %d wales, a = %g\n" % (M.shape + (A,)) + "=" * 100)
    lab, ns = sections(M); print("sections:", ns, "-> sizes", sorted(int((lab == k).sum()) for k in range(1, ns + 1)), "\n")
    sw, tw, sc, tc = strengths(M)
    print("TM axis strength per wale  (s):", " ".join("%.2f" % s for s in sw)); print("  majority type per wale        :", " ".join("%4d" % t for t in tw))
    print("TM axis strength per course(s):", " ".join("%.2f" % s for s in sc)); print("  -> wale bands (first, last, type, strength):", bands(sw, tw), "| course bands:", bands(sc, tc), "\n")
    for mode in ("boundary", "tm", "both"):
        L, lab, ns = field(M, mode); show(M, L, "mode %s" % mode + ("   (per-section mean check: %s)" % ", ".join("%.6f" % L[lab == k].mean() for k in range(1, ns + 1))))
    print("=" * 100 + "\nDILUTED TEST MATRICES (Miura-ori-like: Pattern 1 with every k-th loop of its pure wales 6, 7, 14, 15 flipped to the other type)\n" + "=" * 100)
    print("%-26s %-8s %-8s | strength of wales 6,7,14,15 | bands (wale)" % ("dilution", "f", "s"))
    for k in (None, 10, 5, 3, 2):
        D = M.copy()
        if k:
            for w in (6, 7, 14, 15):
                for r in range(k - 1, M.shape[0], k): D[r, w] = 3 - D[r, w]
        sw, tw, sc, tc = strengths(D)
        f = (1.0 if not k else 1 - sum(D[:, 6] == 1) / M.shape[0]); fw = max((D[:, 6] == 2).mean(), (D[:, 6] == 1).mean())
        print("%-26s %-8.2f %-8.2f | %s | %s" % ("pure (no dilution)" if not k else "1 loop in %d (%.0f %%)" % (k, 100 / k), fw, max(0.0, 2 * (fw - .5)), " ".join("%.2f" % sw[w] for w in (6, 7, 14, 15)), [(b[0], b[1], int(b[2]), round(b[3], 2)) for b in bands(sw, tw)]))
    print("\nFull strength vectors for 1-in-10 dilution:")
    D = M.copy()
    for w in (6, 7, 14, 15):
        for r in range(9, M.shape[0], 10): D[r, w] = 3 - D[r, w]
    sw, tw, sc, tc = strengths(D); print("  per wale  :", " ".join("%.2f" % s for s in sw)); print("  per course:", " ".join("%.2f" % s for s in sc))
    L, lab, ns = field(D, "both"); show(D, L, "mode both on the 1-in-10 diluted matrix (the flipped loops are the other symbol in the axis wales)")
    print("Gain on the loops of the TM axis wales (6, 7, 14, 15; the flipped loops are excluded), in % longer than the uniform start, by dilution:")
    print("   %-16s %-8s %-12s %-12s" % ("dilution", "s", "mode tm", "mode both"))
    for k in (None, 10, 5, 3, 2):
        D = M.copy()
        if k:
            for w in (6, 7, 14, 15):
                for r in range(k - 1, M.shape[0], k): D[r, w] = 3 - D[r, w]
        sel = [(r, w) for w in (6, 7, 14, 15) for r in range(M.shape[0]) if D[r, w] == M[r, w]]
        sw_ = strengths(D)[0][6]; gains = []
        for mode in ("tm", "both"):
            L, lab, ns = field(D, mode); gains.append(100 * (np.mean([L[r, w] for r, w in sel]) - 1))
        print("   %-16s %-8.2f %+-12.2f %+-12.2f" % ("pure" if not k else "1 loop in %d" % k, sw_, *gains))
    print("(mode both changes with the dilution partly because flipped loops split the sections and move the section boundaries; mode tm shows the strength effect alone)")
