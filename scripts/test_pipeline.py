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
    local = loop_centerline_local(props, n_samples=24)
    W = props.geometry.wale_spacing_mm
    H = props.geometry.course_spacing_mm
    assert local[:, 0].min() >= -1e-6 and local[:, 0].max() <= W + 1e-6, \
        "centerline escaped its cell in the u (wale) direction"
    assert local[:, 1].min() >= -1e-6 and local[:, 1].max() <= H + 1e-6, \
        "centerline escaped its cell in the v (course) direction"
    print("test_centerline_stays_within_cell_bounds passed.")


def test_curl_bias_signs():
    assert curl_bias(1) == 1.0
    assert curl_bias(2) == -1.0
    assert curl_bias(0) == 0.0
    assert curl_bias(3) == 0.0
    print("test_curl_bias_signs passed.")


def test_relaxation_converges_and_matches_hand_derivation():
    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)
    grid = [[1, 2], [2, 1]]
    _, curl_targets = build_flat_mesh(grid, props, n_samples=24)

    alpha, beta = 0.3, 0.2
    z_final, iters = relax_to_convergence(curl_targets, max_iter=200, tol=1e-5,
                                            alpha=alpha, beta=beta)
    assert iters < 200, "solver failed to converge within the iteration cap"
    assert not np.allclose(z_final, 0), "z must move away from the zero starting state"

    # Independent hand-derived equilibrium for a checkerboard grid:
    # at steady state, each cell's 4 neighbours are all opposite sign,
    # so neighbor_mean = -z*. Setting the update to zero:
    #   0 = alpha*(target - z*) + beta*(-z* - z*)
    #   z* = alpha / (alpha + 2*beta) * target
    target = curl_targets[0, 0]
    z_hand = alpha / (alpha + 2 * beta) * target
    assert abs(z_hand - z_final[0, 0]) < 1e-3, \
        f"solver result {z_final[0,0]} doesn't match hand-derived equilibrium {z_hand}"
    print(f"test_relaxation_converges_and_matches_hand_derivation passed ({iters} iterations).")


def test_tube_mesh_geometry():
    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)
    local = loop_centerline_local(props, n_samples=24)
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
    test_curl_bias_signs()
    test_relaxation_converges_and_matches_hand_derivation()
    test_tube_mesh_geometry()
    test_before_after_export_differ_but_share_topology()
    test_load_matrix_rejects_ragged_input()
    test_geometric_consistency_check_rejects_impossible_inputs()
    print("All tests passed.")
