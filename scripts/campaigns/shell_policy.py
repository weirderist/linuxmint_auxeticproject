#!/usr/bin/env python3
"""shell_policy.py - the approved rules of the pure-shell campaign, enforced in code, and the deterministic generator of seeds, controls, variants and filler. Standard library only.

Policy file: campaigns/shell_policy.json (approved by K 2026-11-21; it is copied into the campaign's spec.json at plan time, so the campaign keeps the rules it started with).
ESSENTIAL settings (fold-critical, the verified 13 Nov success values) are never varied: violations() lists every difference, and Camp.enqueue / the runner refuse such an entry. The only exceptions are
control runs, labelled, that change a key listed in policy['controls']['allowed_keys'] ('iter').
FREE: the pattern (matrix), CPI and WPI (through R = CPI/WPI at constant stitch density CPI x WPI), tiles (3 to 6, stitch cap). Ne does not enter the shell and is recorded as 12.
All magnitudes of the ranges are in campaigns/shell_policy.json with their sources; those without one are marked UNSOURCED there.
Generator: every random choice comes from random.Random(seed), so the same seed gives the same queue."""
import hashlib, json, math, random


def load(path):
    return json.load(open(path, encoding='utf8'))


def cpi_wpi(R, density):
    """CPI and WPI with CPI / WPI = R and CPI x WPI = density."""
    return round(math.sqrt(density * R), 4), round(math.sqrt(density / R), 4)


def n_stitches(s): return len(s['matrix']) * len(s['matrix'][0]) * s['tiles'] ** 2


def knit_fraction(M): return sum(v == 1 for r in M for v in r) / float(len(M) * len(M[0]))


def violations(s, policy, control=None):
    """List of sentences (empty = allowed). s is a normalised settings dict (shell_campaign.normalise). control = None, or {'reason': str, 'keys': [...]} for a labelled control run."""
    v = []; keys = set((control or {}).get('keys') or []); rg = policy['ranges']
    if control:
        bad = keys - set(policy['controls']['allowed_keys'])
        if bad or not control.get('reason'): v.append('a control run needs a reason and may only change %s (asked: %s)' % (policy['controls']['allowed_keys'], sorted(keys)))
    for k, val in policy['essential'].items():
        if s.get(k) != val and k not in keys: v.append('ESSENTIAL setting %s = %r differs from the approved value %r' % (k, s.get(k), val))
    for k, val in (policy.get('no_effect') or {}).items():
        if s.get(k) != val: v.append('%s = %r differs from the recorded constant %r' % (k, s.get(k), val))
    R, d = s['cpi'] / s['wpi'], s['cpi'] * s['wpi']; lo, hi = rg['ratio_cpi_over_wpi']
    if not (lo - 1e-9 <= R <= hi + 1e-9): v.append('CPI/WPI = %.4f is outside the approved range %s' % (R, rg['ratio_cpi_over_wpi']))
    if 'density_range' in rg:          # policy v2 (shell2, 2026-11-27): the stitch density CPI x WPI varies within a range
        if not (rg['density_range'][0] - 1e-6 <= d <= rg['density_range'][1] + 1e-6): v.append('CPI x WPI = %.2f is outside the approved range %s' % (d, rg['density_range']))
    elif abs(d / rg['density_cpi_x_wpi'] - 1) > rg['density_tolerance']: v.append('CPI x WPI = %.2f differs from the constant density %.1f by more than %.1f %%' % (d, rg['density_cpi_x_wpi'], 100 * rg['density_tolerance']))
    if not (rg['tiles'][0] <= s['tiles'] <= rg['tiles'][1]): v.append('tiles = %d is outside %s' % (s['tiles'], rg['tiles']))
    if 'matrix_side' in rg and not all(rg['matrix_side'][0] <= n <= rg['matrix_side'][1] for n in (len(s['matrix']), len(s['matrix'][0]))): v.append('matrix %dx%d: each side must be within %s' % (len(s['matrix']), len(s['matrix'][0]), rg['matrix_side']))
    if n_stitches(s) > rg['max_stitches']: v.append('%d stitches per sheet exceed the cap of %d' % (n_stitches(s), rg['max_stitches']))
    return v


# ------------------------------------------------------------------ pattern edits (slight edits of a seed)
def _flip_rect(M, rng, side):
    R, C = len(M), len(M[0]); h, w = rng.randint(1, side), rng.randint(1, side); r0, c0 = rng.randrange(R - h + 1), rng.randrange(C - w + 1)
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w): M[r][c] = 3 - M[r][c]
    return True

def _shift_boundary(M, rng, maxshift, vertical):
    """Move a block boundary by 1..maxshift cells along a run of rows (vertical boundary) or columns (horizontal boundary)."""
    if not vertical: T = [list(x) for x in zip(*M)]; ok = _shift_boundary(T, rng, maxshift, True); M[:] = [list(x) for x in zip(*T)]; return ok
    R, C = len(M), len(M[0]); cands = [(r, c) for r in range(R) for c in range(1, C) if M[r][c - 1] != M[r][c]]
    if not cands: return False
    r0, c = rng.choice(cands); left = M[r0][c - 1]; a = b = r0
    while a > 0 and M[a - 1][c - 1] != M[a - 1][c] and M[a - 1][c - 1] == left: a -= 1
    while b < R - 1 and M[b + 1][c - 1] != M[b + 1][c] and M[b + 1][c - 1] == left: b += 1
    d = rng.randint(1, maxshift); right_move = rng.random() < 0.5
    if right_move and c + d > C: right_move = False
    if not right_move and c - d < 0: right_move = True
    if right_move and c + d > C: return False
    for r in range(a, b + 1):
        if right_move: M[r][c:c + d] = [left] * d
        else: M[r][c - d:c] = [M[r][c]] * d
    return True

def edit_matrix(base, rng, policy):
    """A slight edit of `base`: 1 to max_edit_operations operations (flip a rectangle up to side x side, or shift a block boundary by up to max_boundary_shift), size unchanged, knit fraction within
    knit_fraction_max_delta of the base, both loop types present, not equal to the base. Returns the new matrix, or None after 40 failed attempts."""
    rg = policy['ranges']; f0 = knit_fraction(base)
    for _ in range(40):
        M = [list(r) for r in base]
        for _op in range(rng.randint(1, rg['max_edit_operations'])):
            k = rng.random()
            if k < 0.4: _flip_rect(M, rng, rg['max_rectangle_side'])
            else: _shift_boundary(M, rng, rg['max_boundary_shift'], vertical=(rng.random() < 0.5))
        f = knit_fraction(M)
        if M != base and 0 < f < 1 and abs(f - f0) <= rg['knit_fraction_max_delta'] + 1e-12: return M
    return None


# ------------------------------------------------------------------ the queue: seeds, controls, variants, filler
def _setting(policy, matrix, cpi, wpi, tiles, **kw):
    s = dict(policy['essential']); s.update(policy.get('no_effect') or {}); s.update(matrix=matrix, cpi=cpi, wpi=wpi, tiles=tiles); s.update(kw); return s

def _sha(s): return hashlib.sha1(json.dumps(s, sort_keys=True).encode()).hexdigest()

def _logu(rng_u, lo, hi): return math.exp(math.log(lo) + rng_u * (math.log(hi) - math.log(lo)))

def build_plan(policy, seeds, rng_seed, n_variants, n_filler, seed_tiles=4):
    """seeds: list of dicts {id, matrix}. Returns a list of entries {settings, source, tier, control (or None), label}. Tiers (lower runs first): 0 seeds, 1 controls, 2 variants, 9 filler (the proposer uses 5).
    Seeds are at K's fabric (CPI 22, WPI 16) and the 13 Nov success settings; Pattern 1 appears at tiles 3 (the verified 13 Nov geometry) and at the seed tiles. Variants: slight edits of the seeds, R log-uniform
    and stratified over the approved range at constant stitch density, tiles = seed tiles. Filler: further edits of the seeds, tiles 3 to 6 within the stitch cap. Every entry is unique by settings."""
    rg = policy['ranges']; lo, hi = rg['ratio_cpi_over_wpi']; dens = rg['density_cpi_x_wpi']; plan = []; seen = set(); P1 = next(s for s in seeds if s['id'] == 'pattern1')
    def add(settings, source, tier, control=None, label=''):
        if violations(settings, policy, control): raise ValueError('plan entry %s breaks the policy: %s' % (label or source, violations(settings, policy, control)))
        h = _sha(settings)
        if h in seen: return False
        seen.add(h); plan.append(dict(settings=settings, source=source, tier=tier, control=control, label=label)); return True
    for s in seeds:
        add(_setting(policy, s['matrix'], 22.0, 16.0, 3 if s['id'] == 'pattern1' else seed_tiles), 'seed:' + s['id'], 0, label=s['id'])
    add(_setting(policy, P1['matrix'], 22.0, 16.0, seed_tiles), 'seed:pattern1:tiles%d' % seed_tiles, 0, label='pattern1 at tiles %d' % seed_tiles)
    for it in (800, 200):
        add(_setting(policy, P1['matrix'], 22.0, 16.0, 3, iter=it), 'control:iter%d' % it, 1, control=dict(reason='sensitivity of the Pattern 1 shell to the iteration cap (400 is the approved value)', keys=['iter']), label='control iter %d' % it)
    rng = random.Random(rng_seed); perm = list(range(n_variants)); rng.shuffle(perm); k = 0; i = 0
    while k < n_variants and i < n_variants * 20:
        base = seeds[k % len(seeds)]; M = edit_matrix(base['matrix'], rng, policy); i += 1
        if M is None: continue
        R = _logu((perm[k] + rng.random()) / n_variants, lo, hi); cpi, wpi = cpi_wpi(R, dens)
        if add(_setting(policy, M, cpi, wpi, seed_tiles), 'variant:' + base['id'], 2, label='variant %03d of %s, R %.3f' % (k + 1, base['id'], cpi / wpi)): k += 1
    rng2 = random.Random(rng_seed + 1); k = 0; i = 0
    while k < n_filler and i < n_filler * 20:
        base = seeds[rng2.randrange(len(seeds))]; M = edit_matrix(base['matrix'], rng2, policy); i += 1
        if M is None: continue
        R = _logu(rng2.random(), lo, hi); cpi, wpi = cpi_wpi(R, dens)
        feasible = [t for t in range(rg['tiles'][0], rg['tiles'][1] + 1) if len(M) * len(M[0]) * t * t <= rg['max_stitches']]
        if not feasible: continue
        if add(_setting(policy, M, cpi, wpi, rng2.choice(feasible)), 'filler:' + base['id'], 9, label='filler of %s' % base['id']): k += 1
    return plan


# ------------------------------------------------------------------ policy v2 (shell2, 2026-11-27): random patterns, resized seed edits, R and stitch density varied together
def random_blocks(rng, rows, cols, min_side=2):
    """Random block pattern: the rectangle is split recursively (guillotine cuts at random positions, every block at least min_side cells per side) and every block gets a random digit; redrawn until both digits occur."""
    while True:
        M = [[0] * cols for _ in range(rows)]; stack = [(0, 0, rows, cols)]
        while stack:
            r0, c0, h, w = stack.pop(); can_h, can_w = h >= 2 * min_side, w >= 2 * min_side
            if (can_h or can_w) and rng.random() < 0.8 and h * w > 4 * min_side * min_side:
                if can_h and (not can_w or rng.random() < h / float(h + w)): k = rng.randint(min_side, h - min_side); stack += [(r0, c0, k, w), (r0 + k, c0, h - k, w)]
                else: k = rng.randint(min_side, w - min_side); stack += [(r0, c0, h, k), (r0, c0 + k, h, w - k)]
            else:
                d = rng.choice((1, 2))
                for r in range(r0, r0 + h):
                    for c in range(c0, c0 + w): M[r][c] = d
        if 0 < knit_fraction(M) < 1: return M

def random_pixels(rng, rows, cols):
    """Pixel-random pattern: every cell knit with probability p, p uniform in [0.3, 0.7] per pattern (UNSOURCED bounds); redrawn until both digits occur."""
    p = rng.uniform(0.3, 0.7)
    while True:
        M = [[1 if rng.random() < p else 2 for _ in range(cols)] for _ in range(rows)]
        if 0 < knit_fraction(M) < 1: return M

def resize_matrix(base, rng, lo, hi, maxd):
    """Slight size change: add or remove 1..maxd rows and/or columns (an added row/column repeats a neighbouring one, so the block structure is kept); every side stays in [lo, hi]."""
    M = [list(r) for r in base]
    for axis in (0, 1):
        if rng.random() < 0.5: continue
        if axis == 1: M = [list(x) for x in zip(*M)]
        n = len(M); d = rng.randint(1, maxd) * (1 if rng.random() < 0.5 else -1); d = max(lo - n, min(hi - n, d))
        for _ in range(abs(d)):
            i = rng.randrange(len(M))
            if d > 0: M.insert(i, list(M[i]))
            elif len(M) > lo: M.pop(i)
        if axis == 1: M = [list(x) for x in zip(*M)]
    return M

def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0: f /= b; r += f * (i % b); i //= b
    return r

def tiles_for(policy, M):
    """The largest tile count in the policy range whose sheet stays within the stitch cap (None if none fits)."""
    rg = policy['ranges']; ok = [t for t in range(rg['tiles'][0], rg['tiles'][1] + 1) if len(M) * len(M[0]) * t * t <= rg['max_stitches']]
    return max(ok) if ok else None

def build_plan_v2(policy, seeds, rng_seed, n_total):
    """Plan of a v2 campaign (shell2). Tier 0: every seed at K's fabric (CPI 22, WPI 16). Tier 2: n_total generated runs in BALANCED order (every prefix of the queue has the target mix and
    covers the (R, density) square): mix policy['mix'] {blocks, pixels, edits} (K 2026-11-26: 60 % random = 2/3 blocks + 1/3 pixels, 40 % seed edits), the next category is always the one furthest
    below its target count; (R, density) from a Halton sequence (bases 2, 3) with a random shift, R log-uniform in ratio_cpi_over_wpi and density uniform in density_range. Random sizes: rows and
    columns uniform in matrix_side; edits: resize_matrix (up to max_resize rows/columns) then edit_matrix. Tiles: tiles_for (largest within the stitch cap)."""
    rg = policy['ranges']; lo, hi = rg['ratio_cpi_over_wpi']; dlo, dhi = rg['density_range']; slo, shi = rg['matrix_side']; mix = policy['mix']; plan = []; seen = set()
    def add(M, cpi, wpi, source, tier, label):
        t = tiles_for(policy, M)
        if t is None: return False
        st = _setting(policy, M, cpi, wpi, t)
        if violations(st, policy): raise ValueError('plan entry %s breaks the policy: %s' % (label, violations(st, policy)))
        h = _sha(st)
        if h in seen: return False
        seen.add(h); plan.append(dict(settings=st, source=source, tier=tier, control=None, label=label)); return True
    for s in seeds: add(s['matrix'], 22.0, 16.0, 'seed:' + s['id'], 0, s['id'])
    rng = random.Random(rng_seed); u0, v0 = rng.random(), rng.random(); counts = {k: 0 for k in mix}; k = 0; tries = 0
    while k < n_total and tries < 50 * n_total:
        tries += 1; cat = max(mix, key=lambda c: (mix[c] * (k + 1) - counts[c], c))
        R = math.exp(math.log(lo) + ((halton(k + 1, 2) + u0) % 1.0) * (math.log(hi) - math.log(lo))); dens = dlo + ((halton(k + 1, 3) + v0) % 1.0) * (dhi - dlo); cpi, wpi = cpi_wpi(R, dens)
        if cat == 'edits':
            base = seeds[rng.randrange(len(seeds))]; B = resize_matrix(base['matrix'], rng, slo, shi, rg['max_resize']); M = edit_matrix(B, rng, policy); src = 'edit:' + base['id']
            if M is None: continue
        else:
            M = (random_blocks if cat == 'blocks' else random_pixels)(rng, rng.randint(slo, shi), rng.randint(slo, shi)); src = 'random:' + cat
        if add(M, cpi, wpi, src, 2, '%s %04d, %dx%d, R %.3f, CPI x WPI %.0f' % (src, k + 1, len(M), len(M[0]), cpi / wpi, cpi * wpi)): counts[cat] += 1; k += 1
    return plan
