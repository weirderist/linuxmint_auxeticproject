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
    if abs(d / rg['density_cpi_x_wpi'] - 1) > rg['density_tolerance']: v.append('CPI x WPI = %.2f differs from the constant density %.1f by more than %.1f %%' % (d, rg['density_cpi_x_wpi'], 100 * rg['density_tolerance']))
    if not (rg['tiles'][0] <= s['tiles'] <= rg['tiles'][1]): v.append('tiles = %d is outside %s' % (s['tiles'], rg['tiles']))
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
