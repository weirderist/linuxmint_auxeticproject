"""
export_obj.py -- runs the fabric simulation pipeline end-to-end and
saves the resulting before/after loop geometry as .obj files, with
filenames timestamped to when the export was run.

Usage:
    python3 export_obj.py

Output:
    /home/sarvesh/auxetic_project/sim_output/obj/before_YYYYMMDD_HHMMSS.obj
    /home/sarvesh/auxetic_project/sim_output/obj/after_YYYYMMDD_HHMMSS.obj
"""

import os
from datetime import datetime

from solver_3d import MaterialProps
from run_simulation import generate_fabric_3d
from export_mesh import export_loops_to_obj

OUTPUT_DIR = "/home/sarvesh/auxetic_project/sim_output/obj"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    props = MaterialProps(Ne=10, cpi=14, wpi=18, loop_length_mm=3.2)

    block = 3
    rows, cols = 6, 6
    grid = [
        [1 if ((r // block) + (c // block)) % 2 == 0 else 2 for c in range(cols)]
        for r in range(rows)
    ]

    before, after, iterations, contact_z = generate_fabric_3d(grid, props, n_samples=40)
    print(f"Relaxation converged in {iterations} iterations. "
          f"z range: [{contact_z.min():.4f}, {contact_z.max():.4f}] mm")

    radius = props.geometry.yarn_diameter_mm / 2.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    before_path = os.path.join(OUTPUT_DIR, f"before_{timestamp}.obj")
    after_path = os.path.join(OUTPUT_DIR, f"after_{timestamp}.obj")

    export_loops_to_obj(before, radius, before_path, n_radial=12)
    export_loops_to_obj(after, radius, after_path, n_radial=12)

    print(f"Saved: {before_path}")
    print(f"Saved: {after_path}")


if __name__ == "__main__":
    main()
