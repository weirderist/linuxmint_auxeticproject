#!/usr/bin/env python3
"""loop_grade.py - the optional STARTING rest-length field of relax_job.py (--loop-grade). Standard library plus numpy only: no JAX, no SciPy, so it can be tested without the solver.

What it does (opt-in; with --loop-grade 0 nothing here is called and the start is exactly the uniform ratio of relax_state.load):
  * One loop = one stitch of one course (course c, wale t). The loops carry the stitch type of the matrix AS BUILT (the matrix tiled `tiles` times, as relax_state.load does), because tile joins merge sections.
  * A SECTION is a 4-connected region of one loop type. The edge of the fabric counts as a section boundary.
  * Boundary term G_b: 1 next to the section boundary, falling linearly with the Euclidean distance (in loops) to 0 at the farthest loop of the section. Loops of two sections that share a boundary both
    get 1 next to it (a two-loop-wide ridge).
  * TM axis strength: for every wale and every course, f = fraction of its majority type, s = max(0, 2 (f - 0.5)); a line is an axis if s >= the minimum strength. Consecutive axis lines of the same
    majority type form one band (strength = the mean of s over the band).
  * TM term G_t: s times a linear fall-off with the distance across the band (1 on the band, 0 at the farthest loop of the section), only in sections that contain loops of the band's lines.
  * Target mode: boundary (G = G_b), tm (G = G_t) or both (G = G_b + G_t, the default).
  * Field per solver segment: f = 1 + a (G - G_S), G the overlap-weighted G of the loops the segment lies in, G_S the rest-length-weighted (= chord-weighted) mean of G over the segments of its section S.
    So every section keeps EXACTLY the total rest length of the uniform start, and a term that is uniform over a section drops out. `a` is the length difference between the target loops and the farthest
    loops as a fraction of the section's mean loop length.
UNSOURCED (no measured loop-length field exists yet; K will set them from measurements by region): the value of a, the minimum axis strength (default 0.2), the linear strength mapping s = 2 (f - 0.5),
the linear fall-off with distance, the Euclidean distance to the section boundary, the mode weights (both 1), the additive per-section normalisation, the merging of consecutive same-type axis lines
into one band with their mean strength, and the numerical guard FACTOR_FLOOR.
Creep (--creep-tau, --hours) moves every rest length toward the current length each outer round and does NOT conserve the total, so it can partly erase this starting grade.
This module also holds the loop-boundary code shared with webui/runview.py (resample, boundaries, loop_sum): the viewer and the field use the same loop identity.
"""
import os
import numpy as np

DEFAULT_MIN_STRENGTH = 0.2        # UNSOURCED
W_BOUNDARY, W_TM = 1.0, 1.0       # UNSOURCED mode weights
FACTOR_FLOOR = 0.2                # numerical guard: a segment may not get a rest-length factor below this
TARGETS = ('boundary', 'tm', 'both')
UNSOURCED = ['a (the size of the grade)', 'minimum axis strength (default 0.2)', 'strength mapping s = 2 (f - 0.5)', 'linear fall-off with distance', 'Euclidean distance to the section boundary',
             'mode weights (boundary 1, tm 1)', 'additive per-section normalisation', 'merging of consecutive same-type axis lines into one band (mean strength)', 'factor floor 0.2 (numerical guard)']


# ------------------------------------------------------------------ loop identity (shared with webui/runview.py)
def resample(P, ds):
    """Same as relax_state.resample (tests/test_loop_grade.py checks they are equal): the solver's vertices of one course."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.linspace(0, s[-1], max(3, int(round(s[-1] / ds)) + 1))
    return np.c_[[np.interp(u, s, P[:, k]) for k in range(3)]].T

def cum_at(cum, p):
    i = np.minimum(np.floor(p).astype(int), len(cum) - 2)
    return cum[i] + (p - i) * (cum[i + 1] - cum[i])

def loop_sum(v, beta):
    """Sum of a per-segment quantity over each loop. beta = the loop boundaries in segment units (fractional); the segments that straddle a boundary are split in proportion."""
    a = cum_at(np.r_[0.0, np.cumsum(v)], beta)
    return a[1:] - a[:-1]

def wale_spacing(meta):
    """Wale spacing W of a build. The top-level meta['W'] (full precision) comes first: structure['W'] is rounded to 6 significant digits (build_knit.fnum), and with that rounding x / W of the
    k-th loop boundary misses an integer by more than the 1e-6 tolerance of boundaries() for most WPI (all of 10..40 except 10, 16, 20, 25, 32 and 40 at 48 wales; checked 2026-11-25), so the exact rule
    silently fell back to the x = k W crossing. Identical for WPI 16 (25.4/16 = 1.5875 exactly), so the fixtures and golden fingerprints do not change."""
    return meta.get('W') or (meta.get('structure') or {}).get('W')

def boundaries(P, W, ncol):
    """Fine-curve sample indices of the ncol + 1 loop boundaries of one course, from the build itself: build_knit.py builds every stitch from control points that start and end exactly on x = k W (16 per
    stitch) and the Catmull-Rom curve passes through its control points, so the samples with x / W an integer are the loop boundaries. None if that does not give ncol + 1 boundaries."""
    w = P[:, 0] / W; b = np.where(np.abs(w - np.round(w)) < 1e-6)[0]
    if len(b) != ncol + 1 or b[0] != 0 or b[-1] != len(P) - 1 or not np.array_equal(np.round(w[b]).astype(int), np.arange(ncol + 1)): return None
    return b

def course_boundaries(P, X, W, ncol):
    """(b, beta): b = fine-curve boundary samples (None if the exact rule does not apply), beta = the boundaries in segment units of the resampled course X (X is the solver's course, P the fine curve).
    Fallback (foreign builds): the first crossing of x = k W along X."""
    n = len(X); b = boundaries(P, W, ncol)
    if b is not None:
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; beta = s[b] / (s[-1] / (n - 1))
    else:
        xm = np.maximum.accumulate(X[:, 0]); beta = np.interp(np.arange(ncol + 1) * W, xm, np.arange(n), right=n - 1)
    beta = np.maximum.accumulate(np.clip(beta, 0, n - 1)); beta[0], beta[-1] = 0.0, n - 1.0
    return b, beta


# ------------------------------------------------------------------ sections, strengths, the field on the loop grid
def tiled_matrix(M, tiles):
    return np.tile(np.array(M), (int(tiles), int(tiles)))

def label_sections(T):
    """4-connected regions of equal value. Returns (labels 1..n, n)."""
    T = np.asarray(T); lab = np.zeros(T.shape, int); n = 0
    for r in range(T.shape[0]):
        for c in range(T.shape[1]):
            if lab[r, c]: continue
            n += 1; lab[r, c] = n; stack = [(r, c)]
            while stack:
                y, x = stack.pop()
                for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= yy < T.shape[0] and 0 <= xx < T.shape[1] and not lab[yy, xx] and T[yy, xx] == T[y, x]: lab[yy, xx] = n; stack.append((yy, xx))
    return lab, n

def dist_to_outside(mask):
    """Euclidean distance (in loops) of every loop of a section to the nearest loop outside it; the fabric edge counts as outside. 1 next to the boundary."""
    pad = np.pad(mask, 1); oy, ox = np.nonzero(~pad); out = np.c_[oy, ox].astype(float); iy, ix = np.nonzero(pad); ins = np.c_[iy, ix].astype(float)
    d = np.empty(len(ins))
    for s in range(0, len(ins), 256): d[s:s + 256] = np.sqrt(((ins[s:s + 256, None, :] - out[None, :, :]) ** 2).sum(2).min(1))
    res = np.zeros(mask.shape); res[iy - 1, ix - 1] = d
    return res

def strengths(T):
    """(strength per wale, majority type per wale, strength per course, majority type per course). s = max(0, 2 (f - 0.5)), f = fraction of the majority type (UNSOURCED mapping)."""
    T = np.asarray(T)
    def one(lines):
        ss, tt = [], []
        for line in lines:
            f1 = float((line == 1).mean()); f = max(f1, 1 - f1); ss.append(max(0.0, 2 * (f - 0.5))); tt.append(1 if f1 >= 0.5 else 2)
        return np.array(ss), np.array(tt)
    sw, tw = one(T.T); sc, tc = one(T)
    return sw, tw, sc, tc

def bands(s, t, s_min):
    """Consecutive axis lines (s >= s_min) of the same majority type -> list of (first, last, type, mean strength)."""
    out, i = [], 0
    while i < len(s):
        if s[i] >= s_min:
            j = i
            while j + 1 < len(s) and s[j + 1] >= s_min and t[j + 1] == t[i]: j += 1
            out.append((int(i), int(j), int(t[i]), float(np.mean(s[i:j + 1])))); i = j + 1
        else: i += 1
    return out

def loop_G_parts(T, s_min=DEFAULT_MIN_STRENGTH):
    """(G_b, G_t, section labels, number of sections, info): the two terms on the loop grid, before the target mode is applied."""
    T = np.asarray(T); lab, ns = label_sections(T); G_b = np.zeros(T.shape); G_t = np.zeros(T.shape); yy, xx = np.mgrid[0:T.shape[0], 0:T.shape[1]]
    for k in range(1, ns + 1):
        mask = lab == k; d = dist_to_outside(mask); dm = d[mask].max(); G_b[mask] = 1.0 if dm <= 1 else 1.0 - (d[mask] - 1.0) / (dm - 1.0)
    sw, tw, sc, tc = strengths(T); wb, wc = bands(sw, tw, s_min), bands(sc, tc, s_min)
    for lines, pos in ((wb, xx), (wc, yy)):
        for (i, j, typ, s) in lines:
            delta = np.maximum(0, np.maximum(i - pos, pos - j)).astype(float)
            for k in range(1, ns + 1):
                mask = lab == k
                if not (mask & (T == typ) & (pos >= i) & (pos <= j)).any(): continue     # this section has no loop of the band's lines
                dm = delta[mask].max(); G_t[mask] += s * (1.0 if dm == 0 else 1.0 - delta[mask] / dm)
    info = dict(sections=[dict(id=k, type=int(T[lab == k][0]), loops=int((lab == k).sum())) for k in range(1, ns + 1)], wale_bands=wb, course_bands=wc,
                strength_wale=[float(x) for x in sw], strength_course=[float(x) for x in sc])
    return G_b, G_t, lab, ns, info

def loop_G(T, target='both', s_min=DEFAULT_MIN_STRENGTH):
    """(G per loop, section labels, number of sections, info). G = W_B G_b + W_T G_t according to the target mode."""
    if target not in TARGETS: raise ValueError('target must be one of %s' % (TARGETS,))
    T = np.asarray(T); G_b, G_t, lab, ns, info = loop_G_parts(T, s_min)
    G = (W_BOUNDARY * G_b if target in ('boundary', 'both') else 0.0) + (W_TM * G_t if target in ('tm', 'both') else 0.0)
    return np.asarray(G, float) * np.ones(T.shape), lab, ns, info

def field_equal_weights(T, a, target='both', s_min=DEFAULT_MIN_STRENGTH):
    """The field on the loop grid with every loop weighted equally (what tests/analysis_loop_grade_design.py shows)."""
    G, lab, ns, _ = loop_G(T, target, s_min); L = np.ones(np.shape(T))
    for k in range(1, ns + 1): L[lab == k] = 1.0 + a * (G[lab == k] - G[lab == k].mean())
    return L, lab, ns


# ------------------------------------------------------------------ the field per solver segment
def segment_overlaps(beta, nseg, ncol):
    """(nseg, ncol) matrix of the length (in segments) of each solver segment that lies in each loop; every row sums to 1. beta = loop boundaries in segment units."""
    i = np.arange(nseg, dtype=float)[:, None]
    return np.clip(np.minimum(i + 1, beta[None, 1:]) - np.maximum(i, beta[None, :-1]), 0, None)

def segment_field(sim, folder, meta, target='both', s_min=DEFAULT_MIN_STRENGTH):
    """The loop field mapped onto the solver's segments, WITHOUT changing anything: dict(T, lab, ns, info, G (per loop), Gs (per yarn: G per segment, overlap-weighted), secs (per yarn: section of each segment, the
    loop with the larger overlap), wts (per yarn: current l0), betas (per yarn: loop boundaries in segment units), ov (per yarn: segment x loop overlap fractions), nfall). Call right after relax_state.load()."""
    M = meta.get('matrix'); tiles = meta.get('tiles'); W = wale_spacing(meta)
    if not (W and M and tiles): raise ValueError('the starting loop grade needs the stitch matrix, tiles and W in meta.json (result built by an older pipeline)')
    T = tiled_matrix(M, tiles); nrow, ncol = T.shape
    G, lab, ns, info = loop_G(T, target, s_min)
    A = np.loadtxt(os.path.join(folder, 'fab_axis.csv'), delimiter=',', skiprows=1)
    Gs, secs, wts, betas, ovs, nfall = [], [], [], [], [], 0
    for y in sim.yarns:
        c = int(y.course)
        if not 0 <= c < nrow: raise ValueError('course %d is outside the %d courses of the tiled matrix' % (c, nrow))
        P = A[A[:, 0] == c][:, 1:]; b, beta = course_boundaries(P, y.X, W, ncol); nfall += b is None
        ov = segment_overlaps(beta, len(y.X) - 1, ncol); ov = ov / ov.sum(1, keepdims=True)
        Gs.append(ov @ G[c]); secs.append(lab[c][np.argmax(ov, axis=1)]); wts.append(np.asarray(y.l0, float)); betas.append(beta); ovs.append(ov)
    Gsm = []
    for y, beta, g in zip(sim.yarns, betas, Gs):                      # smooth version for the persistent drive: loop values interpolated linearly along the yarn between the loop centres
        centres = 0.5*(beta[:-1] + beta[1:]); Gsm.append(np.interp(np.arange(len(g)) + 0.5, centres, G[int(y.course)]))
    nseg = sum(len(g) for g in Gs); per_loop = nseg / float(nrow*ncol)
    return dict(T=T, lab=lab, ns=ns, info=info, G=G, Gs=Gs, Gs_smooth=Gsm, segments_per_loop=per_loop, secs=secs, wts=wts, betas=betas, ov=ovs, nfall=nfall, ncol=ncol, nrow=nrow)

def drive_field(f):
    """G per segment for the persistent drive (slip_ext.SlipExt): the loop values interpolated along the yarn, multiplied by the mean number of segments per loop. The slip rule acts only at contact nodes and
    a loop boundary is rarely a contact node, so a field that jumps at the boundary would act almost nowhere; the smooth field has a small step at every node, and the step times the segments per loop is the
    G change per loop, which makes the drive strength e 'the effective strain imbalance per unit G difference between adjacent loops'. UNSOURCED modelling choice."""
    return [g*f['segments_per_loop'] for g in f['Gs_smooth']]

def mean_loop_length(folder, meta):
    """Mean as-built yarn length per loop (mm in mm results), from the fine fab_axis.csv curve between the exact loop boundaries; None if the boundaries or the matrix are not available."""
    M = meta.get('matrix'); tiles = meta.get('tiles'); W = wale_spacing(meta)
    if not (W and M and tiles): return None
    ncol = len(M[0]) * int(tiles); A = np.loadtxt(os.path.join(folder, 'fab_axis.csv'), delimiter=',', skiprows=1); tot, n = 0.0, 0
    for c in np.unique(A[:, 0]):
        P = A[A[:, 0] == c][:, 1:]; b = boundaries(P, W, ncol)
        if b is None: return None
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1); cum = np.r_[0.0, np.cumsum(seg)]; tot += float(cum[b[-1]] - cum[b[0]]); n += ncol
    return tot / n if n else None

def tightness_record(meta, folder):
    """Knitting tightness factor TF = sqrt(tex) / loop length, recorded in meta.json, NO effect on any calculation. tex = 590.5 / Ne (standard cotton-count conversion); loop length = the mean as-built
    yarn length per loop of the build. Returns None for results without Ne (R-unit results) or without exact loop boundaries."""
    ne = meta.get('ne') or (meta.get('structure') or {}).get('ne')
    if not ne: return None
    L = mean_loop_length(folder, meta)
    if not L: return None
    tex = 590.5 / float(ne)
    return dict(ne=float(ne), tex=tex, sqrt_tex=float(np.sqrt(tex)), loop_length_mm=float(L), tightness_factor_per_mm=float(np.sqrt(tex) / L), tightness_factor_per_cm=float(np.sqrt(tex) / (L / 10.0)),
                definition='TF = sqrt(tex) / loop length; tex = 590.5 / Ne; loop length = mean as-built yarn length per loop of the build (not a measured value)')

def apply_start_grade(sim, folder, meta, a, target='both', s_min=DEFAULT_MIN_STRENGTH, with_shell=False):
    """Multiply the starting rest length of every segment by its grade factor, in place, and return the record for meta.json. Call right after relax_state.load(), before anything reads l0 or moves X
    (the loop of a segment is found from the as-built geometry in `folder`). Raises ValueError for a meta.json without the matrix, tiles and W, or for a grade that would shrink a segment below
    FACTOR_FLOOR."""
    f = segment_field(sim, folder, meta, target, s_min); T, lab, ns, info, Gs, secs, wts, nfall, nrow, ncol = f['T'], f['lab'], f['ns'], f['info'], f['Gs'], f['secs'], f['wts'], f['nfall'], f['nrow'], f['ncol']
    Gall, sall, wall = np.concatenate(Gs), np.concatenate(secs), np.concatenate(wts)
    mean_S = {int(k): float((wall[sall == k] * Gall[sall == k]).sum() / wall[sall == k].sum()) for k in np.unique(sall)}
    lut = np.zeros(ns + 1)
    for k, v in mean_S.items(): lut[k] = v
    factors = [1.0 + a * (g - lut[s]) for g, s in zip(Gs, secs)]
    fall = np.concatenate(factors)
    if fall.min() < FACTOR_FLOOR: raise ValueError('the grade is too large: some segments would get a rest-length factor of %.3f (floor %.2f)' % (fall.min(), FACTOR_FLOOR))
    before = {int(k): float(wall[sall == k].sum()) for k in np.unique(sall)}
    for y, f in zip(sim.yarns, factors): y.l0 = y.l0 * f
    after = {int(k): float(sum((y.l0[s == k]).sum() for y, s in zip(sim.yarns, secs))) for k in np.unique(sall)}
    sections = []
    for k in np.unique(sall):
        m = sall == k; sizes = [d for d in info['sections'] if d['id'] == k][0]
        sections.append(dict(id=int(k), type=sizes['type'], loops=sizes['loops'], segments=int(m.sum()), l0_before=before[int(k)], l0_after=after[int(k)], mean_factor=float((wall[m] * fall[m]).sum() / wall[m].sum()),
                             factor_min=float(fall[m].min()), factor_max=float(fall[m].max())))
    return dict(a=float(a), target=target, min_strength=float(s_min), mode_weights=dict(boundary=W_BOUNDARY, tm=W_TM), with_shell=bool(with_shell), matrix_as_built=[int(nrow), int(ncol)],
                sections_count=int(ns), sections=sections, wale_bands=[list(b) for b in info['wale_bands']], course_bands=[list(b) for b in info['course_bands']], factor_min=float(fall.min()), factor_max=float(fall.max()),
                total_l0_before=float(sum(before.values())), total_l0_after=float(sum(after.values())), loop_identity=('exact (build boundaries)' if nfall == 0 else '%d courses used the x = k W fallback' % nfall),
                rule='sections = 4-connected same-type regions of the tiled matrix (fabric edge = boundary); boundary term + graded TM-axis term, linear fall-off; per-section rest-length-weighted conservation; '
                     'segment section = loop with the larger overlap, G = overlap-weighted G of its loops',
                creep_note='creep (--creep-tau) moves rest lengths without conserving the total and can partly erase this grade', unsourced=UNSOURCED)
