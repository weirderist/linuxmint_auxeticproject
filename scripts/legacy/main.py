"""
main.py — CLI entry point for the rule-based relaxation + puckering
prediction pipeline. Produces before.obj (flat) and after.obj (relaxed)
from a classified F/B matrix file, and optionally 8 rendered views
(4 before + 4 after) if Blender is available.
"""

import os
import argparse
import json

from solver_3d import MaterialProps
from run_simulation import load_matrix, generate_fabric_3d
from export_mesh import export_loops_to_obj, render_turntable_views


def main():
    parser = argparse.ArgumentParser(description="Auxetic knit puckering solver")
    parser.add_argument("--matrix", required=True, help="Path to matrix .txt (crop_classify.py output)")
    parser.add_argument("--Ne", type=float, default=10.0)
    parser.add_argument("--cpi", type=float, required=True, help="Courses per inch, measured")
    parser.add_argument("--wpi", type=float, required=True, help="Wales per inch, measured")
    parser.add_argument("--loop-length-mm", type=float, required=True)
    parser.add_argument("--output-dir", default="sim_output")
    parser.add_argument("--n-samples", type=int, default=24, help="Centerline points per loop")
    parser.add_argument("--n-radial", type=int, default=8, help="Cross-section resolution")
    parser.add_argument("--render", action="store_true", help="Also render 4+4 turntable views via Blender")
    parser.add_argument("--n-views", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    grid = load_matrix(args.matrix)
    props = MaterialProps(Ne=args.Ne, cpi=args.cpi, wpi=args.wpi,
                            loop_length_mm=args.loop_length_mm)

    print(f"Yarn diameter: {props.geometry.yarn_diameter_mm:.4f} mm")
    print(f"Peirce loop radius (R = 4.172 x d): {props.geometry.radius_mm:.4f} mm")
    print(f"Course spacing: {props.geometry.course_spacing_mm:.4f} mm  (from CPI={args.cpi})")
    print(f"Wale spacing: {props.geometry.wale_spacing_mm:.4f} mm  (from WPI={args.wpi})")
    print(f"Curl amplitude (calibratable, default = yarn diameter): {props.curl_amplitude_mm:.4f} mm")

    before, after, iters, z = generate_fabric_3d(
        grid, props, n_samples=args.n_samples
    )
    print(f"Relaxation converged in {iters} iterations.")

    radius = props.geometry.yarn_diameter_mm / 2.0
    before_path = os.path.join(args.output_dir, "before.obj")
    after_path = os.path.join(args.output_dir, "after.obj")

    export_loops_to_obj(before, radius, before_path, n_radial=args.n_radial)
    export_loops_to_obj(after, radius, after_path, n_radial=args.n_radial)

    # Save the calculation summary alongside the mesh -- this is what
    # should accompany the 8 images, not just the pictures alone.
    summary = {
        "inputs": {
            "Ne": args.Ne, "cpi": args.cpi, "wpi": args.wpi,
            "loop_length_mm": args.loop_length_mm,
        },
        "derived": {
            "yarn_diameter_mm": props.geometry.yarn_diameter_mm,
            "peirce_radius_mm": props.geometry.radius_mm,
            "course_spacing_mm": props.geometry.course_spacing_mm,
            "wale_spacing_mm": props.geometry.wale_spacing_mm,
            "curl_amplitude_mm": props.curl_amplitude_mm,
        },
        "relaxation": {
            "iterations": iters,
            "converged": iters < 200,
        },
        "grid_shape": [len(grid), len(grid[0])],
    }
    summary_path = os.path.join(args.output_dir, "calculation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Calculation summary written to {summary_path}")

    if args.render:
        before_views_dir = os.path.join(args.output_dir, "views_before")
        after_views_dir = os.path.join(args.output_dir, "views_after")
        render_turntable_views(before_path, before_views_dir, n_views=args.n_views)
        render_turntable_views(after_path, after_views_dir, n_views=args.n_views)
        print(f"Rendered {args.n_views} before-views to {before_views_dir}")
        print(f"Rendered {args.n_views} after-views to {after_views_dir}")


if __name__ == "__main__":
    main()
