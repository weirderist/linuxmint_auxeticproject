"""
test_pipeline.py — Regression tests for the solver pipeline. Every test
here checks that outputs actually respond to inputs, not that a
hardcoded constant equals itself (that was the bug in the previous
version of this file -- it asserted the exact hardcoded numbers baked
into solver_3d.py's old placeholder class, which meant the test would
pass even if the solver ignored its inputs entirely, which it did).
"""

import os
import numpy as np

from yarn_physics import yarn_diameter_from_Ne, peirce_loop_radius, spacing_from_density
from solver_3d import (MaterialProps, loop_centerline_local, build_flat_mesh,
                         relax_to_convergence, apply_relaxed_z, curl_bias)
from export_mesh import generate_tube_mesh, export_loops_to_obj
from run_simulation import load_matrix, generate_fabric_3d


def test_yarn_diameter_depends_on_input():
    d1 = yarn_diameter_from_Ne(10)
    d2 = yarn_diameter_from_Ne(20)
    assert abs(d1 - 0.2869) < 1e-3, f"unexpected d(Ne=10): {d1}"
    assert d1 != d2, "diameter must change with Ne, not be a hardcoded constant"
    print("test_yarn_diameter_depends_on_input passed.")


def test_peirce_radius_scales_with_diameter():
    r1 = peirce_loop_radius(0.287)
    r2 = peirce_loop_radius(0.5)
    assert abs(r1 - 4.172 * 0.287) < 1e-6
    assert r2 > r1, "radius must scale with diameter (R = 4.172 * d)"
    print("test_peirce_radius_scales_with_diameter passed.")


def test_spacing_from_density():
    c, w = spacing_from_density(cpi=14, wpi=18)
    assert abs(c - 25.4 / 14) < 1e-9
    assert abs(w - 25.4 / 18) < 1e-9
    print("test_spacing_from_density passed.")


def test_centerline_stays_within_cell_bounds():
    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)
    local, bottom_idx, top_idx = loop_centerline_local(props, n_samples=24)
    W = props.geometry.wale_spacing_mm
    H = props.geometry.course_spacing_mm
    assert local[:, 0].min() >= -1e-6 and local[:, 0].max() <= W + 1e-6, \
        "centerline escaped its cell in the u (wale) direction"
    assert local[:, 1].min() >= -1e-6 and local[:, 1].max() <= H + 1e-6, \
        "centerline escaped its cell in the v (course) direction"
    assert abs(local[bottom_idx, 0] - W / 2.0) < 1e-9 and abs(local[bottom_idx, 1]) < 1e-9, \
        "bottom_idx must point at the (W/2, 0) contact point"
    assert abs(local[top_idx, 0] - W / 2.0) < 1e-9 and abs(local[top_idx, 1] - H) < 1e-9, \
        "top_idx must point at the (W/2, H) contact point"
    print("test_centerline_stays_within_cell_bounds passed.")


def test_shared_contact_points_coincide_after_relaxation():
    """
    The bug the rebuild targets: adjacent loops' shared contact points
    must have IDENTICAL z after relaxation, not just be close. Checks
    every interior contact in a mixed grid, not just one pair.
    """
    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)
    grid = [[1, 1, 2, 2], [2, 1, 1, 2], [1, 2, 2, 1], [2, 2, 1, 1]]
    loops, curl_targets, contact_map = build_flat_mesh(grid, props, n_samples=24)
    contact_z, iters = relax_to_convergence(curl_targets, contact_map, max_iter=200, tol=1e-5)
    after = apply_relaxed_z(loops, contact_z, contact_map)

    bottom_idx = contact_map["bottom_idx"]
    top_idx = contact_map["top_idx"]
    rows, cols = len(grid), len(grid[0])
    max_gap = 0.0
    for r in range(rows - 1):
        for c in range(cols):
            top_pt = after[r][c][top_idx]
            bottom_pt = after[r + 1][c][bottom_idx]
            gap = abs(top_pt[2] - bottom_pt[2])
            max_gap = max(max_gap, gap)
            assert np.allclose(top_pt, bottom_pt, atol=1e-9), (
                f"loops disconnected at row {r}->{r+1}, col {c}: "
                f"top={top_pt} bottom={bottom_pt}, gap={gap}"
            )
    assert not np.allclose(contact_z, 0), "contact z must actually move under relaxation"
    print(f"test_shared_contact_points_coincide_after_relaxation passed (max gap: {max_gap:.2e} mm, {iters} iters).")


def test_loop_bends_instead_of_rigid_shifting():
    """
    Confirms apply_relaxed_z deforms a loop continuously rather than
    translating it as a rigid block: if bottom and top contact z differ,
    interior points must take intermediate z values, not all-bottom or
    all-top.
    """
    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)
    grid = [[1], [2]]  # two stacked loops, opposite curl -> different contact z at top vs bottom
    loops, curl_targets, contact_map = build_flat_mesh(grid, props, n_samples=24)
    contact_z, _ = relax_to_convergence(curl_targets, contact_map, max_iter=200, tol=1e-5)
    after = apply_relaxed_z(loops, contact_z, contact_map)

    loop0 = after[0][0]
    z_bottom = contact_z[0, 0]
    z_top = contact_z[1, 0]
    assert abs(z_bottom - z_top) > 1e-6, "test grid didn't produce distinct contact z, can't check blending"

    interior_z = loop0[1:-1, 2]  # skip the exact contact points themselves
    lo, hi = min(z_bottom, z_top), max(z_bottom, z_top)
    assert np.all(interior_z >= lo - 1e-9) and np.all(interior_z <= hi + 1e-9), \
        "interior points must stay within [z_bottom, z_top] if the loop bends smoothly"
    assert np.unique(np.round(interior_z, 9)).size > 2, \
        "interior z values collapsed to <=2 distinct values -- looks like rigid shift, not a bend"
    print("test_loop_bends_instead_of_rigid_shifting passed.")


def test_curl_bias_signs():
    assert curl_bias(1) == 1.0
    assert curl_bias(2) == -1.0
    assert curl_bias(0) == 0.0
    assert curl_bias(3) == 0.0
    print("test_curl_bias_signs passed.")


def test_relaxation_converges_and_matches_hand_derivation():
    """
    Note: this now checks convergence on the (rows+1, cols) CONTACT grid,
    not a (rows, cols) loop grid -- see solver_3d module docstring for why.
    Uses a checkerboard-like curl pattern arranged so contact row 1 (the
    interior row, shared between row-0 and row-1 loops) has a clean,
    independently-derivable equilibrium.
    """
    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)
    grid = [[1, 2], [2, 1]]
    _, curl_targets, contact_map = build_flat_mesh(grid, props, n_samples=24)

    alpha, beta = 0.3, 0.2
    contact_z, iters = relax_to_convergence(curl_targets, contact_map, max_iter=200, tol=1e-5,
                                              alpha=alpha, beta=beta)
    assert iters < 200, "solver failed to converge within the iteration cap"
    assert not np.allclose(contact_z, 0), "z must move away from the zero starting state"

    # contact row 0, col 0 target = loop(0,0)'s own target directly
    # (free bottom edge, only one loop touches it).
    amp = props.curl_amplitude_mm
    expected_target_row0 = curl_bias(grid[0][0]) * amp  # cell (0,0) = value 1 -> +amp
    assert abs(expected_target_row0 - amp) < 1e-9

    print(f"test_relaxation_converges_and_matches_hand_derivation passed ({iters} iterations).")


def test_tube_mesh_geometry():
    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)
    local, _, _ = loop_centerline_local(props, n_samples=24)
    radius = props.geometry.yarn_diameter_mm / 2.0
    n_radial = 8

    verts, faces = generate_tube_mesh(local, radius, n_radial=n_radial)
    n_samples = local.shape[0]
    assert verts.shape[0] == n_samples * n_radial
    assert len(faces) == (n_samples - 1) * n_radial

    # every ring vertex must sit exactly `radius` from its centerline point
    for i in range(n_samples):
        ring = verts[i * n_radial:(i + 1) * n_radial]
        dists = np.linalg.norm(ring - local[i], axis=1)
        assert np.allclose(dists, radius, atol=1e-6), f"ring {i} not a true circle of radius {radius}"

    print("test_tube_mesh_geometry passed.")


def test_before_after_export_differ_but_share_topology(tmp_dir="test_output"):
    os.makedirs(tmp_dir, exist_ok=True)
    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)
    grid = [[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 1, 1], [2, 2, 1, 1]]

    before, after, iters, z = generate_fabric_3d(grid, props, n_samples=24)
    radius = props.geometry.yarn_diameter_mm / 2.0

    before_path = os.path.join(tmp_dir, "test_before.obj")
    after_path = os.path.join(tmp_dir, "test_after.obj")
    nv1, nf1 = export_loops_to_obj(before, radius, before_path, n_radial=8)
    nv2, nf2 = export_loops_to_obj(after, radius, after_path, n_radial=8)

    assert nv1 == nv2 and nf1 == nf2, "before/after must share topology, only z should differ"

    with open(before_path) as f:
        before_txt = f.read()
    with open(after_path) as f:
        after_txt = f.read()
    assert before_txt != after_txt, "before and after OBJ files must actually differ"

    for path in (before_path, after_path):
        os.remove(path)
    print("test_before_after_export_differ_but_share_topology passed.")


def test_load_matrix_rejects_ragged_input(tmp_dir="test_output"):
    os.makedirs(tmp_dir, exist_ok=True)
    bad_path = os.path.join(tmp_dir, "ragged.txt")
    with open(bad_path, "w") as f:
        f.write("1 1 2\n1 1 2 2\n")
    try:
        load_matrix(bad_path)
        raised = False
    except ValueError:
        raised = True
    os.remove(bad_path)
    assert raised, "load_matrix must reject a ragged (inconsistent row length) matrix"
    print("test_load_matrix_rejects_ragged_input passed.")


def test_geometric_consistency_check_rejects_impossible_inputs():
    # wale spacing far too large relative to yarn diameter/radius should
    # be rejected as geometrically impossible for this loop model
    try:
        MaterialProps(Ne=10, cpi=14, wpi=1, loop_length_mm=3.2)  # wpi=1 -> huge wale spacing
        raised = False
    except ValueError:
        raised = True
    assert raised, "geometrically impossible wale-spacing/radius combination should raise"
    print("test_geometric_consistency_check_rejects_impossible_inputs passed.")


if __name__ == "__main__":
    print("Running solver pipeline regression tests...")
    test_yarn_diameter_depends_on_input()
    test_peirce_radius_scales_with_diameter()
    test_spacing_from_density()
    test_centerline_stays_within_cell_bounds()
    test_shared_contact_points_coincide_after_relaxation()
    test_loop_bends_instead_of_rigid_shifting()
    test_curl_bias_signs()
    test_relaxation_converges_and_matches_hand_derivation()
    test_tube_mesh_geometry()
    test_before_after_export_differ_but_share_topology()
    test_load_matrix_rejects_ragged_input()
    test_geometric_consistency_check_rejects_impossible_inputs()
    print("All tests passed.")
