"""
yarn_physics.py — Yarn and loop geometry relations for the auxetic knit solver.

TRACEABILITY (read this before trusting or changing any number below):

1. Yarn diameter from count:
   d = 1 / (28 * sqrt(Ne))  [inches], converted to mm.
   Source: Peirce's empirical relation, derived from an assumed cotton
   specific volume of 1.1 cm^3/g. Widely reproduced in textile engineering
   references (e.g. d(mm) = 0.0037 * sqrt(tex) is the equivalent tex form).
   Does NOT correct for twist level — stated simplification, not an
   oversight. Confirm "3/30s" vs "10s" convention with your machine setup
   before trusting this for anything other than the assumed 10s Ne case.

2. Loop curvature radius:
   R = 4.172 * d
   Source: F.T. Peirce (1947), "Geometrical Principles Applicable to the
   Design of Functional Fabrics," as reported in secondary textile-geometry
   references (e.g. Textile School, "Geometrical Modelling of Plain Weft
   Knitted Fabrics"). This is Peirce's derived radius of curvature required
   for yarn interlocking in a "normal" (compact, yarns-in-contact)
   structure — R/d is stated as a constant specifically for that packing
   assumption, not a general-purpose radius formula.

3. Course/wale spacing:
   Taken DIRECTLY from measured courses-per-inch (CPI) and wales-per-inch
   (WPI), NOT derived from Munden's k1/k2 empirical constants. Munden
   (1959) established that c = k_c * l and w = k_w * l with k_c, k_w
   constant for a given fibre and relaxation state — this relationship is
   real and well established, but the actual numeric values of k_c/k_w
   vary by fibre, structure, and relaxation state and are reported in
   tables across multiple papers (e.g. Pavko-Cuden et al., "Parameters of
   compact single weft knitted structure"). No single trustworthy pair of
   numbers could be confirmed for this project's specific yarn/state in
   the time available, so rather than hardcode a guessed constant, this
   module uses your directly measured CPI/WPI instead. This is more
   accurate for your case anyway, since you already measure density
   per sample rather than needing to predict it from loop length alone.

4. Stitch density consistency check (Doyle's relation):
   S = c * w is proportional to l^2 (Doyle, reported via ScienceDirect
   "Loop Geometry" topic overview, citing Munden's later generalisation
   that this holds independent of d/l). Used here only as a DIAGNOSTIC —
   to flag if your measured c, w, l are wildly inconsistent with each
   other — not as a predictive formula.

5. Tightness factor:
   T.F. = sqrt(tex) / l(cm)
   Standard textile relation (reproduced in NPTEL knitting technology
   course material). Provided as an additional consistency/reference
   check, not used elsewhere in the solver.

Anything NOT listed above (e.g. the specific curl magnitude used later in
solver_3d.py) is explicitly flagged there as a calibratable assumption,
not a literature-sourced constant. Do not add new "sourced" formulas to
this file without the same citation discipline as above.
"""

import math
from dataclasses import dataclass


def yarn_diameter_from_Ne(Ne: float) -> float:
    """
    Yarn diameter in mm from English cotton count Ne (single ply).
    d(in) = 1 / (28 * sqrt(Ne))
    """
    if Ne <= 0:
        raise ValueError(f"Ne must be positive, got {Ne}")
    d_in = 1.0 / (28.0 * math.sqrt(Ne))
    return d_in * 25.4


def peirce_loop_radius(yarn_diameter_mm: float) -> float:
    """
    Loop arc curvature radius, Peirce (1947), 'normal' compact structure.
    R = 4.172 * d
    """
    if yarn_diameter_mm <= 0:
        raise ValueError(f"yarn_diameter_mm must be positive, got {yarn_diameter_mm}")
    return 4.172 * yarn_diameter_mm


def spacing_from_density(cpi: float, wpi: float) -> tuple:
    """
    Course spacing and wale spacing in mm, from measured courses-per-inch
    and wales-per-inch. Plain unit conversion, not a model.
    Returns (course_spacing_mm, wale_spacing_mm).
    """
    if cpi <= 0 or wpi <= 0:
        raise ValueError(f"cpi and wpi must be positive, got cpi={cpi}, wpi={wpi}")
    course_spacing_mm = 25.4 / cpi
    wale_spacing_mm = 25.4 / wpi
    return course_spacing_mm, wale_spacing_mm


def stitch_density_consistency_ratio(course_spacing_mm: float,
                                       wale_spacing_mm: float,
                                       loop_length_mm: float) -> float:
    """
    Doyle's diagnostic ratio: S / l^2, where S = c * w.
    This is NOT asserted to equal a specific constant (no confirmed
    literature value for this project's exact fibre/state) — use it to
    compare relative consistency across your own samples, not against
    an assumed universal number.
    """
    if loop_length_mm <= 0:
        raise ValueError("loop_length_mm must be positive")
    S = course_spacing_mm * wale_spacing_mm
    return S / (loop_length_mm ** 2)


def tightness_factor(tex: float, loop_length_cm: float) -> float:
    """
    T.F. = sqrt(tex) / l(cm)
    Reference/diagnostic only — not used elsewhere in the pipeline.
    """
    if loop_length_cm <= 0:
        raise ValueError("loop_length_cm must be positive")
    return math.sqrt(tex) / loop_length_cm


@dataclass
class LoopGeometry:
    """
    Bundled, per-sample geometric parameters. All fields either directly
    measured (loop_length_mm, course_spacing_mm, wale_spacing_mm) or
    derived via a cited relation above (yarn_diameter_mm, radius_mm).
    """
    yarn_diameter_mm: float
    radius_mm: float
    course_spacing_mm: float
    wale_spacing_mm: float
    loop_length_mm: float

    @classmethod
    def from_measurements(cls, Ne: float, cpi: float, wpi: float, loop_length_mm: float):
        d = yarn_diameter_from_Ne(Ne)
        r = peirce_loop_radius(d)
        c, w = spacing_from_density(cpi, wpi)
        return cls(
            yarn_diameter_mm=d,
            radius_mm=r,
            course_spacing_mm=c,
            wale_spacing_mm=w,
            loop_length_mm=loop_length_mm,
        )
