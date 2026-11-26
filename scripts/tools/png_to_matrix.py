#!/usr/bin/env python3
"""png_to_matrix.py - grid-editor PNG (red and green boxes with black borders) -> stitch matrix text, without being told the number of rows and columns.

Same classification as webui/main.py /api/classify and crop_classify.py (measured reference colours; mean colour of the inner part of each cell; nearest reference; 1 = red = front, 2 = green = back),
which need rows and columns as input. Here the grid is found from the black border lines: the coloured bounding box (saturation > 30) gives the extent, the black lines inside it (dark pixels in more than
60 % of a column or row) give the cell boundaries, so cells need not be equal. Row order = image order: the first row of the matrix is the TOP row of the picture (checked against campaigns/pattern1_matrix.txt,
whose first rows have the structure of the top block of seed PNG 2; K confirms with the picture-beside-matrix check).
NOTE (2026-11-26): shell_sim.py, build_knit.py and campaign matrices read the FIRST row as the BOTTOM course, so a matrix written here is simulated upside down relative to its picture. For the
shell this is a mirror image (same fold, reflected); the fabrics are the same on both faces (K). Left as it is so that new seeds stay consistent with the shell1 seeds; flip the rows if a picture's
top/bottom ever matters (e.g. comparing a yarn run course by course with a photograph).
Output format = campaigns/pattern1_matrix.txt: rows separated by /, digits separated by spaces.
Usage: python3 png_to_matrix.py a.png [b.png ...] [--out-dir DIR]   (writes NN.txt per PNG into DIR and prints size and low-confidence cells)
numpy and Pillow (cv2 is not needed)."""
import io, os, sys
import numpy as np

RED_RGB = (255.0, 87.0, 87.0); GREEN_RGB = (126.0, 217.0, 87.0)       # webui/main.py RED_BGR (87, 87, 255), GREEN_BGR (87, 217, 126), reversed to RGB


def load_rgb(path):
    from PIL import Image
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA'); bg = Image.new('RGBA', im.size, (255, 255, 255, 255)); bg.alpha_composite(im); im = bg
    return np.asarray(im.convert('RGB'), dtype=float)


def _centres(frac, lo, hi, thr=0.6):
    idx = np.where(frac[max(lo - 6, 0):hi + 7] > thr)[0] + max(lo - 6, 0); out = []
    if idx.size == 0: return out
    start = prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1: out.append((start + prev) / 2.0); start = i
        prev = i
    out.append((start + prev) / 2.0); return out


def convert(path):
    img = load_rgb(path); mx, mn = img.max(2), img.min(2); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9) * 255.0, 0.0)
    ys, xs = np.where(sat > 30)
    if ys.size == 0: raise ValueError('%s: no coloured grid found' % path)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()); dark = mx < 90
    vx = _centres(dark[y0:y1 + 1, :].mean(0), x0, x1); hy = _centres(dark[:, x0:x1 + 1].mean(1), y0, y1)
    if len(vx) < 3 or len(hy) < 3: raise ValueError('%s: could not find the black cell borders (%d vertical, %d horizontal lines)' % (path, len(vx), len(hy)))
    cols, rows = len(vx) - 1, len(hy) - 1; red, green = np.array(RED_RGB), np.array(GREEN_RGB); M, conf = [], []
    for r in range(rows):
        row, crow = [], []
        for c in range(cols):
            ya, yb, xa, xb = hy[r], hy[r + 1], vx[c], vx[c + 1]; dy, dx = 0.2 * (yb - ya), 0.2 * (xb - xa)
            cell = img[int(ya + dy):max(int(yb - dy), int(ya + dy) + 1), int(xa + dx):max(int(xb - dx), int(xa + dx) + 1)]
            m = cell.reshape(-1, 3).mean(0); dr, dg = np.linalg.norm(m - red), np.linalg.norm(m - green)
            row.append(1 if dr < dg else 2); crow.append(float(abs(dr - dg) / max(dr + dg, 1e-9)))
        M.append(row); conf.append(crow)
    cw = np.diff(vx); ch = np.diff(hy)
    return dict(matrix=M, rows=rows, cols=cols, confidence=conf, box=[x0, y0, x1, y1], cell_w=(float(cw.min()), float(cw.max())), cell_h=(float(ch.min()), float(ch.max())),
                low_confidence=[[r, c] for r in range(rows) for c in range(cols) if conf[r][c] < 0.25])


def to_text(M): return '/'.join(' '.join(str(v) for v in row) for row in M)


if __name__ == '__main__':
    a = sys.argv[1:]; out = None
    if '--out-dir' in a: i = a.index('--out-dir'); out = a[i + 1]; a = a[:i] + a[i + 2:]
    for f in a:
        r = convert(f); print('%s: %d rows x %d cols, cell %s x %s px, %d low-confidence cells, %d red (1), %d green (2)' % (os.path.basename(f), r['rows'], r['cols'], r['cell_w'], r['cell_h'], len(r['low_confidence']),
              sum(v == 1 for row in r['matrix'] for v in row), sum(v == 2 for row in r['matrix'] for v in row)))
        if out:
            os.makedirs(out, exist_ok=True); p = os.path.join(out, os.path.splitext(os.path.basename(f))[0] + '.txt')
            if os.path.exists(p): sys.exit('refusing to overwrite %s' % p)
            open(p, 'w').write(to_text(r['matrix']))
