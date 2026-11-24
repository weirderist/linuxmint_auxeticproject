import os
import sys
import cv2
import numpy as np

BASE_DIR = "/home/sarvesh/auxetic_project"
SIM_INPUT_DIR = os.path.join(BASE_DIR, "input_grids")
SIM_OUTPUT_DIR = os.path.join(BASE_DIR, "debug_output")

# Measured reference BGR values
KNOWN_RED_BGR = np.array([87.0, 87.0, 255.0], dtype=float)
KNOWN_GREEN_BGR = np.array([87.0, 217.0, 126.0], dtype=float)

def process_grid(filename, expected_rows=None, expected_cols=None):
    input_path = os.path.join(SIM_INPUT_DIR, filename)
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    img = cv2.imread(input_path)
    if img is None:
        print(f"ERROR: Failed to load image {input_path}")
        sys.exit(1)

    if expected_rows is None or expected_cols is None:
        print("ERROR: Dimensions not provided. Explicit row and column dimensions are required.")
        sys.exit(1)

    rows = expected_rows
    cols = expected_cols
    print(f"Using explicit dimensions: {rows} Rows x {cols} Cols")

    # Detect colored grid bounding box using Saturation (HSV)
    # Ignores white background and gray/black border lines
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    color_mask = sat > 30

    ys, xs = np.where(color_mask)
    if ys.size == 0 or xs.size == 0:
        print("ERROR: No grid content detected in image.")
        sys.exit(1)

    grid_x0, grid_x1 = int(xs.min()), int(xs.max())
    grid_y0, grid_y1 = int(ys.min()), int(ys.max())
    grid_w = grid_x1 - grid_x0
    grid_h = grid_y1 - grid_y0
    print(f"Detected grid content bounds: x[{grid_x0}:{grid_x1}] y[{grid_y0}:{grid_y1}]")

    cell_h = grid_h / rows
    cell_w = grid_w / cols

    if cell_w <= 1.0 or cell_h <= 1.0:
        print("ERROR: Invalid cell dimensions (image too small for requested grid).")
        sys.exit(1)

    debug_img = img.copy()
    grid_matrix = []

    inset_y = cell_h * 0.15
    inset_x = cell_w * 0.15

    for r in range(rows):
        row_data = []
        for c in range(cols):
            y1 = int(grid_y0 + r * cell_h + inset_y)
            y2 = int(grid_y0 + (r + 1) * cell_h - inset_y)
            x1 = int(grid_x0 + c * cell_w + inset_x)
            x2 = int(grid_x0 + (c + 1) * cell_w - inset_x)

            y1, y2 = max(0, y1), min(img.shape[0], y2)
            x1, x2 = max(0, x1), min(img.shape[1], x2)

            cell = img[y1:y2, x1:x2]
            if cell.size == 0:
                mean_bgr_arr = np.array(cv2.mean(img)[:3], dtype=float)
            else:
                mean_bgr = cv2.mean(cell)[:3]
                mean_bgr_arr = np.array(mean_bgr, dtype=float)

            # Nearest color matching (1 = Red, 2 = Green)
            dist_red = np.linalg.norm(mean_bgr_arr - KNOWN_RED_BGR)
            dist_green = np.linalg.norm(mean_bgr_arr - KNOWN_GREEN_BGR)

            if dist_red < dist_green:
                row_data.append(1)  # Red
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            else:
                row_data.append(2)  # Green
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        grid_matrix.append(row_data)

    print("Grid Matrix Parsed Successfully:")
    for row in grid_matrix:
        print(" ".join(map(str, row)))

    os.makedirs(os.path.join(BASE_DIR, "output_matrices"), exist_ok=True)
    matrix_filename = os.path.splitext(filename)[0] + "_matrix.txt"
    matrix_path = os.path.join(BASE_DIR, "output_matrices", matrix_filename)
    with open(matrix_path, "w") as f:
        for row in grid_matrix:
            f.write(" ".join(map(str, row)) + "\n")
    print(f"Matrix saved to: {matrix_path}")

    os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)
    debug_filename = os.path.splitext(filename)[0] + "_debug.png"
    debug_path = os.path.join(SIM_OUTPUT_DIR, debug_filename)
    cv2.imwrite(debug_path, debug_img)
    print(f"Debug image saved to: {debug_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python crop_classify.py <filename> <rows> <cols>")
        sys.exit(1)

    fname = sys.argv[1]
    exp_r = int(sys.argv[2])
    exp_c = int(sys.argv[3])

    print(f"Processing image: {fname}")
    process_grid(fname, exp_r, exp_c)
