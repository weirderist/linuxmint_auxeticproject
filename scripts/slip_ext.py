#!/usr/bin/env python3
"""slip_ext.py - two OPT-IN extensions of the capstan slip rule of relax_state.migrate (numpy; the root finder is SciPy's brentq, the same as relax_state.slip_at_node; no JAX).
Both are off by default: with neither switched on relax_state.migrate does not touch this module and the run is bit-identical to a run without it.

(1) Load-dependent slip limit (--slip-friction-load mu_n): the capstan limit exp(mu phi) T_lo uses only the wrap angle. The limit becomes  T_hi <= max(T_lo, 0) exp(mu phi) + mu_n F_n,
    F_n = kc x overlap of the contact (kc = k_contact, overlap = max(0, 2R - contact distance), the penalty force the solver already has). A tightly packed fabric presses its yarns together harder,
    so its slip limit rises and the tension stays stored as strain; a loose one slips freely. mu_n is UNSOURCED (no measured value).
(2) Persistent drive (--slip-drive e, --slip-drive-decay N): at the node between two segments with the loop-field values G_a and G_b (the sectioned field of scripts/loop_grade.py, targets and minimum
    strength from --loop-grade-target / --loop-grade-min-strength; loop_grade.drive_field: the loop values interpolated along the yarn between the loop centres and multiplied by the segments per loop, so that
    G_b - G_a is the G change per loop), the effective tensions get an offset  +-0.5 e EA (G_b - G_a) k(round):  the segment in the loop with the higher G looks more
    tense, so rest length flows toward it (toward the section boundary and the TM axes) every round, for as long as the strains do not build a back-stress that cancels it. e is the effective strain
    imbalance per unit G difference (dimensionless, signed: a negative e pushes the other way). The strength falls as exp(-round / N) when N > 0 (outer rounds; N = 0 = no decay).
    e and N are UNSOURCED; the form (an offset in effective tension), the exponential decay and the use of the sectioned field are modelling choices.
Everything here is UNSOURCED: mu_n, e, N, the additive form of both terms.
"""
import math
import numpy as np

UNSOURCED = ['mu_n (contact-load friction coefficient)', 'e (drive strain per unit G difference)', 'N (decay, outer rounds)', 'the additive form of the friction term mu_n F_n',
             'the effective-tension-offset form of the drive and its split +-0.5', 'the exponential decay', 'use of the sectioned loop-grade field as the drive potential']


def slip_at_node_ext(la, l0a, lb, l0b, EA, mu, phi, step_max, tension, off_a=0.0, off_b=0.0, extra=0.0):
    """relax_state.slip_at_node with two additions: off_a / off_b (N) are added to the effective tensions of segment a and b (the persistent drive), and `extra` (N) is added to the friction limit
    (mu_n x F_n). With off_a = off_b = extra = 0 it returns exactly what slip_at_node returns (tests/test_slip_ext.py). Returns the rest length moved from b to a (negative: the other way)."""
    from scipy.optimize import brentq
    Ta, Tb = tension(la, l0a, EA) + off_a, tension(lb, l0b, EA) + off_b
    hi_is_a = Ta >= Tb; Thi, Tlo = (Ta, Tb) if hi_is_a else (Tb, Ta)
    ratio = np.exp(mu*phi)
    if Thi <= max(Tlo, 0.0)*ratio + extra + 1e-12 or Thi <= 0: return 0.0            # holds: T_hi <= T_lo exp(mu phi) + mu_n F_n
    src_l0 = l0b if hi_is_a else l0a
    cap = step_max*src_l0
    def gap(dl):
        a0, b0 = (l0a + dl, l0b - dl) if hi_is_a else (l0a - dl, l0b + dl)
        Ta2, Tb2 = tension(la, a0, EA) + off_a, tension(lb, b0, EA) + off_b; Thi2, Tlo2 = (Ta2, Tb2) if hi_is_a else (Tb2, Ta2)
        return Thi2 - ratio*max(Tlo2, 0.0) - extra
    dl = cap if gap(cap) > 0 else brentq(gap, 0.0, cap)
    return dl if hi_is_a else -dl


class SlipExt:
    """Holds the constants and the loop field of the two extensions; relax_state.migrate calls .slip() for every contact node when sim.slip_ext is set."""
    def __init__(self, mu_n=0.0, drive=0.0, decay_rounds=0.0, Gseg=None, kc=0.0, R=0.0):
        if mu_n < 0: raise ValueError('mu_n must be >= 0')
        if decay_rounds < 0: raise ValueError('the drive decay must be >= 0 (0 = none)')
        if drive and Gseg is None: raise ValueError('the persistent drive needs the loop field per segment (Gseg)')
        self.mu_n, self.drive, self.decay, self.Gseg, self.kc, self.R = float(mu_n), float(drive), float(decay_rounds), Gseg, float(kc), float(R)

    def strength(self, k):
        """Drive strength e(k) of outer round k (0-based): e exp(-k / N), or e when N = 0."""
        return self.drive*math.exp(-k/self.decay) if self.decay > 0 else self.drive

    def normal_force(self, dist):
        """F_n of a contact = kc x overlap (N in mm results). The solver's penalty force at the contact pair."""
        return self.kc*max(0.0, 2.0*self.R - dist)

    def offsets(self, yi, i, EA, k):
        """(off_a, off_b) for the node i of yarn yi between segment i-1 (a) and segment i (b): the segment in the loop with the higher G looks more tense."""
        if not self.drive: return 0.0, 0.0
        dT = self.strength(k)*EA*float(self.Gseg[yi][i] - self.Gseg[yi][i - 1])
        return -0.5*dT, 0.5*dT

    def slip(self, yi, i, la, l0a, lb, l0b, EA, mu, phi, step_max, tension, Fn, k):
        off_a, off_b = self.offsets(yi, i, EA, k)
        return slip_at_node_ext(la, l0a, lb, l0b, EA, mu, phi, step_max, tension, off_a, off_b, self.mu_n*Fn)

    def record(self, field=None):
        return dict(mu_n=self.mu_n, drive=self.drive, drive_decay_rounds=self.decay, field=field, form='limit T_hi <= max(T_lo,0) exp(mu phi) + mu_n F_n, F_n = kc x overlap; drive = effective tension offset '
                    '+-0.5 e EA (G_b - G_a) exp(-round/N)', unsourced=UNSOURCED)
