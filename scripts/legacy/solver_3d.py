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

TOPOLOGY (rewritten -- read this if you touched the pre-rebuild version):

Loops used to be built as fully independent closed centerlines, offset
into position and never connected to their neighbours -- confirmed bug:
adjacent loops that touched at z=0 in the flat state pulled apart after
relaxation, because each loop's whole centerline was shifted rigidly by
its own cell's z, with nothing forcing the shared contact point to move
together.

Physical picture this needed to match: loop(r,c)'s needle-arc (top,
peak at u=W/2, v=H) and loop(r+1,c)'s sinker-arc (bottom, peak at
u=W/2, v=0) are two *different* loops' yarn -- they are not the same
piece of yarn -- but they rest against each other at a single contact
point where they interlock. That contact point is one physical location
in space and must move as one point during relaxation, even though the
two loops' yarn paths on either side of it are independent.

Implementation: a CONTACT-POINT GRID of shape (rows+1, cols) holds one
z-value per contact -- contact[r, c] is the point shared between
loop(r-1, c)'s top and loop(r, c)'s bottom (row 0's bottom contacts and
row `rows`'s top contacts are the fabric's free top/bottom edges, not
shared with another loop). Relaxation runs on this contact grid, not on
independent per-loop z. Each loop's own interior points (arcs off the
contact peak, legs) are no longer rigidly shifted as a block -- they are
DEFORMED by linearly blending from its bottom-contact z to its
top-contact z along its own path parameter, so the loop bends
continuously between its two now-synchronized ends instead of
translating as a rigid body. This is still a simplification (linear
z-blend along path parameter, not a real bending-stiffness solve) but it
is a stated one, and it is a strict improvement over rigid-shift: loops
now share real endpoints and bend, rather than floating disconnected
copies of themselves.
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


def _catmull_rom_closed(control_points: np.ndarray, n_per_seg: int) -> np.ndarray:
    """
    Samples a closed, uniform Catmull-Rom spline through control_points.
    Passes exactly through every control point (at sample index
    i * n_per_seg), which is what lets contact points land on exact
    sample indices the same way the old arc-midpoint construction did.
    """
    n = control_points.shape[0]
    out = []
    for i in range(n):
        p0 = control_points[(i - 1) % n]
        p1 = control_points[i % n]
        p2 = control_points[(i + 1) % n]
        p3 = control_points[(i + 2) % n]
        for t in np.linspace(0.0, 1.0, n_per_seg, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            out.append(pt)
    return np.array(out)


def loop_centerline_local(props: MaterialProps, n_samples: int = 24):
    """
    Builds one loop's centerline in local 2D coordinates (u = wale axis,
    v = course axis), origin at the cell's bottom-left corner, z=0
    (flat / unrelaxed state).

    SHAPE (rewritten -- was arc/leg/arc/leg, a stadium-like outline that
    read as "boxy" and didn't match how a loop actually looks under a
    counting glass): now a single closed Catmull-Rom spline through 10
    control points -- needle loop crown, two needle-loop shoulders, two
    right-leg pinch points (where the loop's yarn crosses and the loop
    narrows to its waist -- real loops pinch inward here, they are not
    parallel straight legs), needle loop... sinker loop crown, mirrored
    shoulders/pinch points on the left. This gives continuous curvature
    all the way around -- no straight segments, no corner where an arc
    used to meet a leg -- which is the actual qualitative shape reported
    in the knitting-geometry literature for loop models improved beyond
    the plain semicircle-arc Pierce construction (e.g. elliptical
    needle/sinker heads with a spline-described pillar, Sha et al. 2021;
    Čiukas's elliptical-head reinterpretation, Petrulyte & Petrulis
    2021). This is still a representative visualization, not a
    verified reproduction of any one named model's exact control-point
    placement -- WAIST_FRACTION and shoulder placement below are tuned
    to produce a recognizable loop shape, not sourced from a specific
    paper's coordinates.

    Returns (arr, bottom_contact_idx, top_contact_idx):
      arr               -- (n_samples, 3) array, z column all zeros here.
      bottom_contact_idx -- index into arr of the bottom (sinker loop)
                            crown, exactly at (u=W/2, v=0) -- the point
                            shared with the loop below.
      top_contact_idx    -- index into arr of the top (needle loop)
                            crown, exactly at (u=W/2, v=H) -- the point
                            shared with the loop above.
    Both land exactly on sample points (not interpolated) because the
    spline passes exactly through every control point, and the crowns
    are placed as control points 0 and 5 of 10.
    """
    W = props.geometry.wale_spacing_mm
    H = props.geometry.course_spacing_mm

    # How far the legs pinch in toward the loop's own center-line at
    # their narrowest (the crossover / interlocking region). Tuned for
    # a recognizable pinched-waist loop shape; not a literature value.
    WAIST_FRACTION = 0.22
    waist = W * WAIST_FRACTION

    # Control points, going counter-clockwise from the bottom (sinker)
    # crown, up the right side, across the top (needle) crown, down the
    # left side. Left side mirrors the right side around u=W/2.
    control = np.array([
        (W / 2.0,               0.0),          # 0: sinker loop crown (bottom contact)
        (W * 0.88,              H * 0.16),      # 1: sinker loop right shoulder
        (W - waist,             H * 0.40),      # 2: right leg, lower pinch
        (W - waist * 0.65,      H * 0.60),      # 3: right leg, upper pinch
        (W * 0.88,              H * 0.84),      # 4: needle loop right shoulder
        (W / 2.0,               H),             # 5: needle loop crown (top contact)
        (W * 0.12,              H * 0.84),      # 6: needle loop left shoulder
        (waist * 0.65,          H * 0.60),      # 7: left leg, upper pinch
        (waist,                 H * 0.40),      # 8: left leg, lower pinch
        (W * 0.12,              H * 0.16),      # 9: sinker loop left shoulder
    ])

    n_control = control.shape[0]
    n_per_seg = max(2, n_samples // n_control)
    pts2d = _catmull_rom_closed(control, n_per_seg=n_per_seg)

    bottom_contact_idx = 0
    top_contact_idx = 5 * n_per_seg

    z = np.zeros((pts2d.shape[0], 1))
    arr = np.hstack([pts2d, z])

    # Sanity: contact points must actually land at v=0 and v=H (u=W/2),
    # exactly -- downstream relaxation/export code assumes this.
    assert abs(arr[bottom_contact_idx, 0] - W / 2.0) < 1e-9
    assert abs(arr[bottom_contact_idx, 1] - 0.0) < 1e-9
    assert abs(arr[top_contact_idx, 0] - W / 2.0) < 1e-9
    assert abs(arr[top_contact_idx, 1] - H) < 1e-9

    return arr, bottom_contact_idx, top_contact_idx


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
    everywhere.

    Returns:
      loops         -- (rows x cols) list of local centerlines, already
                       offset into global (x, y, z). Each loop's bottom-
                       and top-contact SAMPLE POINTS are numerically
                       identical to the neighbouring loop's corresponding
                       contact point at this flat stage (z=0 for all), and
                       stay that way through relaxation -- see
                       contact_map below.
      curl_targets  -- (rows x cols) curl target per loop, unchanged from
                       before.
      contact_map   -- dict describing shared-contact bookkeeping, passed
                       through to relax_to_convergence / apply_relaxed_z:
                         'bottom_idx', 'top_idx' -- sample index (within
                             a loop's own point array) of its bottom/top
                             contact point.
                         'contact_z_shape' -- (rows+1, cols), the shape of
                             the shared contact-z grid used during
                             relaxation. contact_z[r, c] is the point
                             shared between loop(r-1, c)'s top and
                             loop(r, c)'s bottom. contact_z[0, c] is the
                             fabric's free bottom edge (only loop(0, c)'s
                             bottom, nothing below it); contact_z[rows, c]
                             is the free top edge (only loop(rows-1, c)'s
                             top, nothing above it).
    """
    rows = len(grid_matrix)
    cols = len(grid_matrix[0])
    W = props.geometry.wale_spacing_mm
    H = props.geometry.course_spacing_mm
    local, bottom_idx, top_idx = loop_centerline_local(props, n_samples=n_samples)

    loops = []
    curl_targets = np.zeros((rows, cols))
    for r in range(rows):
        row_loops = []
        for c in range(cols):
            offset = np.array([c * W, r * H, 0.0])
            row_loops.append(local.copy() + offset)
            curl_targets[r, c] = curl_bias(grid_matrix[r][c]) * props.curl_amplitude_mm
        loops.append(row_loops)

    # Verify the flat-state sharing assumption actually holds numerically
    # before relying on it anywhere else (loop(r,c) top vs loop(r+1,c)
    # bottom must coincide in x,y at z=0).
    for r in range(rows - 1):
        for c in range(cols):
            top_pt = loops[r][c][top_idx]
            bottom_pt = loops[r + 1][c][bottom_idx]
            assert np.allclose(top_pt[:2], bottom_pt[:2], atol=1e-9), (
                f"contact mismatch at row {r}->{r+1}, col {c}: "
                f"top={top_pt[:2]} bottom={bottom_pt[:2]}"
            )

    contact_map = {
        "bottom_idx": bottom_idx,
        "top_idx": top_idx,
        "contact_z_shape": (rows + 1, cols),
    }
    return loops, curl_targets, contact_map


def _contact_curl_target(curl_targets: np.ndarray, r_contact: int, cols: int) -> np.ndarray:
    """
    Per-column curl target FOR A CONTACT ROW r_contact in the (rows+1,
    cols) contact grid. A contact point is shared between loop(r_contact-1)
    below and loop(r_contact) above (where those loops exist). Its target
    is the mean of whichever of those two loops' own curl targets exist
    at that column -- e.g. contact row 0 (fabric's free bottom edge) only
    has loop(0) above it, so it takes loop(0)'s target directly; an
    interior contact row averages the loop below's and loop above's
    targets, since both loops pull on that one shared point.
    """
    rows = curl_targets.shape[0]
    out = np.zeros(cols)
    for c in range(cols):
        vals = []
        if r_contact - 1 >= 0:
            vals.append(curl_targets[r_contact - 1, c])
        if r_contact <= rows - 1:
            vals.append(curl_targets[r_contact, c])
        out[c] = sum(vals) / len(vals)
    return out


def relax_step(contact_z: np.ndarray, contact_targets: np.ndarray,
                alpha: float = 0.3, beta: float = 0.2) -> np.ndarray:
    """
    One relaxation iteration, run on the (rows+1, cols) SHARED CONTACT
    grid (not on independent per-loop z -- see module docstring):
      z_new = z + alpha*(target - z) + beta*(neighbour_mean - z)
    Neighbours here are a contact point's row-adjacent (same course,
    next column -- lateral yarn continuity) and column-adjacent (next
    contact row, same column -- the two contacts a single loop's legs
    connect) contacts. Plain discrete update -- not a force/FEA solve.
    """
    rows, cols = contact_z.shape
    neighbor_mean = np.zeros_like(contact_z)
    counts = np.zeros_like(contact_z)

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r0, r1 = max(0, dr), rows + min(0, dr)
        c0, c1 = max(0, dc), cols + min(0, dc)
        sr0, sr1 = max(0, -dr), rows + min(0, -dr)
        sc0, sc1 = max(0, -dc), cols + min(0, -dc)
        neighbor_mean[sr0:sr1, sc0:sc1] += contact_z[r0:r1, c0:c1]
        counts[sr0:sr1, sc0:sc1] += 1

    counts[counts == 0] = 1
    neighbor_mean /= counts

    z_new = contact_z + alpha * (contact_targets - contact_z) + beta * (neighbor_mean - contact_z)
    return z_new


def relax_to_convergence(curl_targets: np.ndarray, contact_map: dict, max_iter: int = 200,
                           tol: float = 1e-5, alpha: float = 0.3, beta: float = 0.2):
    """
    Builds the (rows+1, cols) contact target grid from the per-loop curl
    targets, then iterates relax_step on the SHARED CONTACT grid until
    the largest change between iterations drops below tol, or max_iter
    is reached. Returns (contact_z, iterations_used) -- iterations_used
    is REAL, not a placeholder.
    """
    contact_rows, cols = contact_map["contact_z_shape"]
    contact_targets = np.zeros((contact_rows, cols))
    for rc in range(contact_rows):
        contact_targets[rc, :] = _contact_curl_target(curl_targets, rc, cols)

    z = np.zeros((contact_rows, cols))
    for i in range(1, max_iter + 1):
        z_new = relax_step(z, contact_targets, alpha=alpha, beta=beta)
        delta = np.max(np.abs(z_new - z))
        z = z_new
        if delta < tol:
            return z, i
    return z, max_iter


def apply_relaxed_z(loops, contact_z: np.ndarray, contact_map: dict):
    """
    Takes the flat-state loop centerlines and the converged shared-
    CONTACT z grid, and returns a new set of centerlines in the 'after'
    / puckered state.

    Unlike the old rigid-shift behaviour, each loop's own points are now
    DEFORMED, not translated as a block: every point in the loop gets a
    z offset linearly blended between its loop's bottom-contact z and
    top-contact z, based on that point's position along the loop's own
    path index (0 at the bottom contact's index, 1 at the top contact's
    index, wrapping around through whichever arc/leg segment the point
    sits on). This guarantees:
      1. The two loops sharing a contact point end up with IDENTICAL z
         at that point (they both read from the same contact_z entry) --
         the disconnected-loop bug is fixed by construction, not by
         coincidence.
      2. Each loop bends continuously between its two ends instead of
         jumping rigidly, which is the "actual loops in a fabric" shape
         requested -- not floating flat rectangles with a step
         discontinuity at each edge.
    """
    rows = len(loops)
    cols = len(loops[0])
    bottom_idx = contact_map["bottom_idx"]
    top_idx = contact_map["top_idx"]
    n_pts = loops[0][0].shape[0]

    # Path-parameter t in [0, 1] for every sample index, measured as
    # fractional distance travelled along the loop's own point sequence
    # from bottom_idx to top_idx going the "short way" through the
    # right leg (increasing index), and the remaining points continue
    # past top_idx, through the left leg, back around to bottom_idx --
    # i.e. this is arc-length-INDEX blend (by point count), not true
    # arc-length blend. Stated simplification: point sampling is already
    # near-uniform per segment (see loop_centerline_local), so index
    # fraction is a reasonable proxy for path fraction without needing
    # a separate arc-length integration pass.
    t = np.zeros(n_pts)
    up_span = (top_idx - bottom_idx) % n_pts
    for i in range(n_pts):
        forward = (i - bottom_idx) % n_pts
        if forward <= up_span:
            t[i] = 0.5 * (forward / up_span) if up_span > 0 else 0.0
        else:
            down_span = n_pts - up_span
            t[i] = 0.5 + 0.5 * ((forward - up_span) / down_span) if down_span > 0 else 1.0
    # t=0 and t=1 both land on bottom_idx conceptually (closed loop);
    # bottom_idx itself is exactly t=0, top_idx is exactly t=0.5. Fold
    # so the BLEND weight (not the path position) is 0 at bottom_idx,
    # 1 at top_idx, and back to 0 approaching bottom_idx from the other
    # side -- a triangle wave in t, since both legs connect the same two
    # contact points and should both bend fully between them.
    blend = np.where(t <= 0.5, t / 0.5, (1.0 - t) / 0.5)

    relaxed = []
    for r in range(rows):
        row_out = []
        for c in range(cols):
            loop = loops[r][c].copy()
            z_bottom = contact_z[r, c]
            z_top = contact_z[r + 1, c]
            loop[:, 2] = z_bottom + blend * (z_top - z_bottom)
            # The blend formula is exact at its endpoints in principle
            # (blend=0 -> z_bottom, blend=1 -> z_top) but float64
            # multiply-then-add can lose the last bit at blend=1 (e.g.
            # 1.0*(z_top-z_bottom)+z_bottom != z_top bit-for-bit). Force
            # exact equality at the two contact indices directly rather
            # than trust the arithmetic, since downstream code
            # (export_mesh) assumes shared vertices coincide exactly.
            loop[bottom_idx, 2] = z_bottom
            loop[top_idx, 2] = z_top
            assert loop[bottom_idx, 2] == z_bottom
            assert loop[top_idx, 2] == z_top
            row_out.append(loop)
        relaxed.append(row_out)
    return relaxed
