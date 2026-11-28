"""yarn_defaults.py - the ONE set of standard yarn-relaxation settings (2026-11-28, K: "ensure consistency"). Used by relax_job.py (its command-line defaults), the shell-fold path
(shell_derived_target/shell_target.py and shell_fold.py) and the Simulate page (webui/main.py CONFIG). tests/test_yarn_defaults.py checks that they agree.

Why this file exists: on 2026-11-24 the shell-fold path got its own defaults (contact stiffness 40 ABSOLUTE, which is 0.19 EA/R for a Ne 12 mm build, the straight-yarn reference
ref-scale 0, curl 0.5, tension 17 cN, perturbation 0.2) instead of the Simulate page's standard (30 EA/R, knitted-shape reference). MEASURED 2026-11-28 on the flat S000617 build
(24 x 24 stitches, 10 x 300 iterations): with those defaults the yarns passed through each other (closest axes 88 % of 2R), the sheet swelled from 5.6 R to 24 R and the yarn
strain reached 6 %; with this standard: closest axes 100.0 % of 2R, size +-0.1 %, thickness 5.6 -> 5.7 R, largest yarn strain 0.30 % (99 % of segments below 0.22 %).

Every number is a numerical / modelling choice, not a measured property, unless its note says otherwise."""
import math

STANDARD = dict(
    E=500.0,            # yarn modulus, MPa (K's choice 2026-11-26; Xiong et al. 2026 give 480-880 MPa for cotton yarns; source note in webui/main.py CONFIG)
    mu=0.49,            # yarn-yarn friction coefficient (the Simulate page default)
    kc_ratio=30.0,      # contact stiffness as a multiple of EA/R (numerical choice, the Simulate page since 13 Nov 2026). NEVER pass an absolute number: it does not scale with the yarn
    ref_scale=1.0,      # stress-free curvature = the knitted loop shape (0 = straight yarn: the loops unravel, measured 2026-11-27 and 2026-11-28)
    curl=0.0,           # stored curl, rad per vertex (off)
    tension_cN=10.0,    # feeder tension, cN (mm builds only; source note in webui/main.py CONFIG)
    perturb=0.02,       # out-of-plane symmetry-breaking amplitude, x R (numerical device)
    outer=200,          # outer rounds at most: the run normally ends earlier by the stall rule below (K 2026-11-28: run until the energy stops falling, not to a fixed count)
    maxiter=300,        # L-BFGS iterations per inner round
    stall_tol=1e-6,     # settled when the relative energy change per outer round stays below this, with no rest-length change and little movement, for stall_rounds rounds. UNSOURCED
    stall_rounds=3,
    stall_move=0.01,    # largest vertex move per outer round allowed for the stall rule, x R. UNSOURCED
)

# Yarn stretch (K 2026-11-28: "yarn stretching must be so low it is invisible to the eye; yarn must not break or be cut off merely due to folding").
# The largest axial strain |l - l0| / l0 of any segment is reported every round and at the end; a run above STRAIN_FLAG is flagged. 1 % is UNSOURCED (K to choose):
# a correctly relaxed S000617 build stays at 0.30 %, the broken runs reached 6-13 %.
STRAIN_FLAG = 0.01


def k_contact(E, R, kc_ratio=STANDARD['kc_ratio'], EA=None):
    """Contact stiffness in the solver's units: kc_ratio x EA / R, with EA = E pi R^2 unless a measured EA is given (the same rule as webui/main.py _sim_cmd)."""
    return kc_ratio * (EA if EA else E * math.pi * R ** 2) / R


def max_strain(yarns):
    """(largest |strain|, largest stretch, largest compression) over all segments: strain = |X[i+1] - X[i]| / l0[i] - 1 (yarns: objects with X (n x 3) and l0 (n-1))."""
    import numpy as np
    hi, lo = 0.0, 0.0
    for y in yarns:
        X = np.asarray(y.X); l0 = np.asarray(y.l0)
        if len(X) < 2 or len(l0) == 0: continue
        n = min(len(X) - 1, len(l0)); e = np.linalg.norm(np.diff(X[:n + 1], axis=0), axis=1) / l0[:n] - 1.0
        hi = max(hi, float(e.max())); lo = min(lo, float(e.min()))
    return max(hi, -lo), hi, lo
