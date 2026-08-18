"""
run_simulation.py - Runs fabric relaxation solver and exports output .obj files.
"""

import os
import numpy as np
from solver_3d import MaterialProps, build_flat_mesh, relax_to_convergence, apply_relaxed_z
from export_mesh import export_loops_to_obj

def load_matrix(txt_path: str):
    grid = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [int(x) for x in line.split()]
            grid.append(row)

    if not grid:
        raise ValueError(f"No rows read from {txt_path}")
    row_len = len(grid[0])
    for i, row in enumerate(grid):
        if len(row) != row_len:
            raise ValueError(f"Row {i} has {len(row)} entries, expected {row_len}")

    if any(cell == 0 or cell == 3 for row in grid for cell in row):
        print("WARNING: matrix contains unclassified (0) or unrecognized (3) cells.")

    return grid

def generate_fabric_3d(grid_matrix, props: MaterialProps, n_samples: int = 24, max_iter: int = 200, tol: float = 1e-5):
    loops, curl_targets = build_flat_mesh(grid_matrix, props, n_samples=n_samples)
    z_relaxed, iters = relax_to_convergence(curl_targets, max_iter=max_iter, tol=tol)
    after_loops = apply_relaxed_z(loops, z_relaxed)
    return loops, after_loops, iters, z_relaxed

if __name__ == "__main__":
    sample_grid = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [2, 2, 1, 1],
        [2, 2, 1, 1],
    ]
    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)
    before, after, iters, z = generate_fabric_3d(sample_grid, props)
    print(f"Converged in {iters} iterations.")
    print("z_relaxed:")
    print(z)

    # Export output .obj file
    out_dir = "/home/sarvesh/auxetic_project/sim_output/obj"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "fabric_output.obj")
    
    # Calculate yarn radius based on material props
    yarn_radius = props.yarn_d_mm / 2.0 if hasattr(props, "yarn_d_mm") else 0.1
    export_loops_to_obj(after, radius=yarn_radius, filepath=out_file)
    print(f"\n[SUCCESS] Exported 3D mesh to: {out_file}")
