"""
solver_3d.py — Builds per-loop 3D centerlines and runs the rule-based
relaxation that produces predicted out-of-plane puckering.

WHAT IS AND ISN'T SOURCED (read before changing constants):

- Loop centerline shape: built from two circular arcs (radius R, from
  Peirce's R = 4.172*d relation in yarn_physics.py) joined by straight
  legs, spanning the measured wale spacing (W) and course spacing (H).
  This is a SIMPLIFIED representation of the "arcs joined by straight
  legs" structure Peirce and later authors describe (e.g. the loop
  described as a quarter-circle / straight-segment / quarter-circle path
  in secondary sources on Peirce's model). It is NOT a full reproduction
  of Peirce's complete 3D cylinder-surface derivation — that requires
  additional parameters (yarn path angle on the cylinder surface, etc.)
  that could not be confidently sourced in the time available. Treat the
  resulting shape as a representative, curvature-correct visualization,
  not a submission-grade reproduction of the original 3D model.

- Curl direction (front loops curl one way, back loops the other) is a
  real, well-established qualitative fact from Munden / Kurbak & Ekmen's
  work on face/back loop curl asymmetry.

- Curl MAGNITUDE (CURL_AMPLITUDE_MM below) is NOT taken from a specific
  Kurbak & Ekmen closed-form equation — no such equation could be
  verified from available sources in this session. It is set as a
  starting value on the order of one yarn diameter (a physically
  reasonable length scale for out-of-plane loop displacement) and is
  explicitly a CALIBRATABLE PARAMETER: tune it against your own
  Tracker/microscope measurements of real puckering amplitude, don't
  treat its current default as a literature-derived number.

- relax_step / relax_to_convergence implement a plain discrete iterative
  update (pull each point toward its curl target, pull each point toward
  its neighbours' average, repeat until change is small). This is the
  "rule-based relaxation" already agreed for this project — explicitly
  NOT a finite-element or force-based physical simulation.
"""

import math
import numpy as np
from dataclasses import dataclass, field

from yarn_physics import LoopGeometry


@dataclass
class MaterialProps:
    Ne: float = 10.0
    cpi: float = 14.0
    wpi: float = 18.0
    loop_length_mm: float = 3.2
    curl_amplitude_mm: float = None  # set to yarn diameter by default; see docstring
    geometry: LoopGeometry = field(init=False, repr=False)

    def __post_init__(self):
        self.geometry = LoopGeometry.from_measurements(
            Ne=self.Ne, cpi=self.cpi, wpi=self.wpi, loop_length_mm=self.loop_length_mm
        )
        if self.curl_amplitude_mm is None:
            self.curl_amplitude_mm = self.geometry.yarn_diameter_mm

        # Physical consistency check: the arcs need chord <= radius to be
        # geometrically constructible. If this fails, the supplied
        # wale-spacing/loop-length/count combination doesn't correspond
        # to a realizable compact loop under this model.
        half_chord = self.geometry.wale_spacing_mm / 2.0
        if half_chord > self.geometry.radius_mm:
            raise ValueError(
                f"Geometrically inconsistent inputs: half wale-spacing "
                f"({half_chord:.4f} mm) exceeds Peirce loop radius "
                f"({self.geometry.radius_mm:.4f} mm). Check Ne/CPI/WPI "
                f"values -- this combination can't form a valid compact "
                f"loop arc under this model."
            )


def loop_centerline_local(props: MaterialProps, n_samples: int = 24) -> np.ndarray:
    """
    Builds one loop's centerline in local 2D coordinates (u = wale axis,
    v = course axis), origin at the cell's bottom-left corner, z=0
    (flat / unrelaxed state). Shape: bottom arc, right leg, top arc, left
    leg -- see module docstring for what this does and doesn't represent.

    Returns an (n_samples, 3) array (z column all zeros at this stage).
    """
    R = props.geometry.radius_mm
    W = props.geometry.wale_spacing_mm
    H = props.geometry.course_spacing_mm

    half_chord = W / 2.0
    half_angle = math.asin(half_chord / R)  # validity already checked in __post_init__

    # Arc centers placed so the arc's chord spans the loop width, offset
    # inward from top/bottom edges by the arc's sagitta.
    sagitta = R - R * math.cos(half_angle)

    n_arc = max(4, n_samples // 4)
    n_leg = max(2, (n_samples - 4 * n_arc) // 2)

    pts = []

    # Bottom arc (sinker loop): circle center at v=R, so the arc's
    # deepest point touches v=0 at the middle (u=W/2), and both ends
    # (where it meets the legs) sit at v=sagitta.
    bottom_center_v = R
    for i in range(n_arc):
        t = -half_angle + 2 * half_angle * (i / (n_arc - 1))
        u = W / 2.0 + R * math.sin(t)
        v = bottom_center_v - R * math.cos(t)
        pts.append((u, v))

    # Right leg: straight run from the bottom arc's end up to the top
    # arc's end, both at u=W.
    leg_bottom_v = sagitta
    leg_top_v = H - sagitta
    for i in range(1, n_leg):
        frac = i / n_leg
        u = W
        v = leg_bottom_v + frac * (leg_top_v - leg_bottom_v)
        pts.append((u, v))

    # Top arc (needle loop): circle center at v=H-R, mirrored in u so the
    # path continues from u=W back to u=0, peak touches v=H at u=W/2.
    top_center_v = H - R
    for i in range(n_arc):
        t = -half_angle + 2 * half_angle * (i / (n_arc - 1))
        u = W / 2.0 - R * math.sin(t)
        v = top_center_v + R * math.cos(t)
        pts.append((u, v))

    # Left leg back down to the bottom arc's start (u=0).
    for i in range(1, n_leg):
        frac = i / n_leg
        u = 0.0
        v = leg_top_v - frac * (leg_top_v - leg_bottom_v)
        pts.append((u, v))

    arr = np.array(pts, dtype=float)
    z = np.zeros((arr.shape[0], 1))
    return np.hstack([arr, z])


def curl_bias(cell_value: int) -> float:
    """
    +1 for front loop (1), -1 for back loop (2), 0 for background/
    unrecognized (0 or 3) -- no curl target for cells that aren't real
    loops. Direction is the sourced part; see module docstring.
    """
    if cell_value == 1:
        return 1.0
    elif cell_value == 2:
        return -1.0
    return 0.0


def build_flat_mesh(grid_matrix, props: MaterialProps, n_samples: int = 24):
    """
    'Before' state: every loop placed at its flat grid position, z=0
    everywhere. Returns a list of (rows x cols) local centerlines already
    offset into global (x, y, z) coordinates, plus the curl target grid
    for use by the relaxation step.
    """
    rows = len(grid_matrix)
    cols = len(grid_matrix[0])
    W = props.geometry.wale_spacing_mm
    H = props.geometry.course_spacing_mm
    local = loop_centerline_local(props, n_samples=n_samples)

    loops = []
    curl_targets = np.zeros((rows, cols))
    for r in range(rows):
        row_loops = []
        for c in range(cols):
            offset = np.array([c * W, r * H, 0.0])
            row_loops.append(local + offset)
            curl_targets[r, c] = curl_bias(grid_matrix[r][c]) * props.curl_amplitude_mm
        loops.append(row_loops)
    return loops, curl_targets


def relax_step(z_grid: np.ndarray, curl_targets: np.ndarray,
                alpha: float = 0.3, beta: float = 0.2) -> np.ndarray:
    """
    One relaxation iteration:
      z_new = z + alpha*(curl_target - z) + beta*(neighbour_mean - z)
    Plain discrete update -- not a force/FEA solve.
    """
    rows, cols = z_grid.shape
    neighbor_mean = np.zeros_like(z_grid)
    counts = np.zeros_like(z_grid)

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r0, r1 = max(0, dr), rows + min(0, dr)
        c0, c1 = max(0, dc), cols + min(0, dc)
        sr0, sr1 = max(0, -dr), rows + min(0, -dr)
        sc0, sc1 = max(0, -dc), cols + min(0, -dc)
        neighbor_mean[sr0:sr1, sc0:sc1] += z_grid[r0:r1, c0:c1]
        counts[sr0:sr1, sc0:sc1] += 1

    counts[counts == 0] = 1
    neighbor_mean /= counts

    z_new = z_grid + alpha * (curl_targets - z_grid) + beta * (neighbor_mean - z_grid)
    return z_new


def relax_to_convergence(curl_targets: np.ndarray, max_iter: int = 200,
                           tol: float = 1e-5, alpha: float = 0.3, beta: float = 0.2):
    """
    Iterates relax_step until the largest change between iterations drops
    below tol, or max_iter is reached. Returns (z_grid, iterations_used) --
    iterations_used is REAL, not a placeholder.
    """
    z = np.zeros_like(curl_targets)
    for i in range(1, max_iter + 1):
        z_new = relax_step(z, curl_targets, alpha=alpha, beta=beta)
        delta = np.max(np.abs(z_new - z))
        z = z_new
        if delta < tol:
            return z, i
    return z, max_iter


def apply_relaxed_z(loops, z_grid: np.ndarray):
    """
    Takes the flat-state loop centerlines and the converged per-cell z
    grid, and returns a new set of centerlines with each loop's z column
    shifted to its converged value ('after' / puckered state). Each
    loop's own internal shape is preserved; the whole loop is displaced
    rigidly in z by its cell's converged value. This is a simplification
    -- a fully accurate model would vary z continuously along each
    loop's own centerline, not just shift it as a rigid body -- flagged
    here explicitly rather than presented as more precise than it is.
    """
    rows = len(loops)
    cols = len(loops[0])
    relaxed = []
    for r in range(rows):
        row_out = []
        for c in range(cols):
            loop = loops[r][c].copy()
            loop[:, 2] += z_grid[r, c]
            row_out.append(loop)
        relaxed.append(row_out)
    return relaxed
