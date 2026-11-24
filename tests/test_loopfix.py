import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)          # repository root: the tests run from any folder
"""Tests for the loop-length colouring fix: baseline uniformity (real single-jersey build), real loop ids, rest-length mode with a known gradient, colour range."""
import os, sys, json, shutil, pathlib
P = pathlib.Path(_os.path.join(ROOT_STR)); SM = P / "smoke_test"
os.environ["CAMPAIGN_DIR"] = str(SM / "campaigns"); os.environ["AUX_AUTORESUME"] = "0"
if not (SM / "campaigns" / "fid_copy" / "runs" / "0020").exists(): sys.exit("fixtures missing: run  python tests/make_fixtures.py  (see tests/README.md)")
if not (SM / "campaigns" / "fid_copy" / "runs" / "0001").exists(): sys.exit("fixtures missing: run  python tests/make_fixtures.py  (see tests/README.md)")
sys.path.insert(0, str(P / "webui")); sys.path.insert(0, str(P / "scripts"))
import numpy as np
import main, runview as rv
from fastapi.testclient import TestClient
main.RESULTS = SM / "results_rv"; rv.RESULTS = main.RESULTS
c = TestClient(main.app); ok = True
def check(label, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + label, "" if cond else extra)
CAMP = SM / "campaigns" / "fid_copy"; J = CAMP / "runs" / "0020"; ID = "run~fid_copy~0020"
# ---------------------------------------------------------------- 1. baseline uniformity on a REAL single-jersey build (build_knit.py, matrix "1", 12 x 12 loops)
f = rv.fields(J); r = f.arc_asb / f.mean_asb
check("single jersey baseline: loop-length field uniform within 0.1 % (new rule)", r.max() - r.min() < 0.001, (r.min(), r.max()))
seg_col = [np.clip(((X[:-1, 0] + X[1:, 0]) / 2 // f.W).astype(int), 0, f.ncol - 1) for X in f.X0]
old = np.array([np.bincount(seg_col[i], weights=f.l_asb[i], minlength=f.ncol)[:f.ncol] for i in range(len(f.F))]); old = old / old.mean()
check("the OLD whole-segment rule was NOT uniform on the same data (>2 %): this is the bug that was fixed", old.max() - old.min() > 0.02, (old.min(), old.max()))
ex = np.array([[np.linalg.norm(np.diff(Pp[f.bidx[i][t]:f.bidx[i][t + 1] + 1], axis=0), axis=1).sum() for t in range(f.ncol)] for i, Pp in enumerate(f.F)])
check("exact fine-curve loop lengths (real build ids) are uniform too (<0.1 %)", (ex.max() - ex.min()) / ex.mean() < 0.001)
check("new rule agrees with the exact loop lengths up to the chord shortfall (0.4 %), uniformly", np.allclose(f.arc_asb / ex, (f.arc_asb / ex).mean(), atol=2e-4) and abs((f.arc_asb / ex).mean() - 1) < 0.006)
N = np.loadtxt(J / "fab_nodes.csv", delimiter=",", skiprows=1); same = True
for i, cc in enumerate(f.courses):
    nc = N[N[:, 0] == cc]; b = np.where(np.abs(nc[:, 2] / f.W - np.round(nc[:, 2] / f.W)) < 1e-6)[0]; per = (len(f.F[i]) - 1) / (len(nc) - 1); same &= np.array_equal(np.round(b * per).astype(int), f.bidx[i])
check("loop boundaries equal the builder's control-point boundaries (fab_nodes.csv), 16 control points per stitch", same and f.identity.startswith("loop boundaries from the build"))
check("loops cover every vertex: loop ids run 0..ncol-1 in order along each course, the ends belong to the end loops", all(v[0] == 0 and v[-1] == f.ncol - 1 and np.all(np.diff(v) >= 0) for v in f.vid0) and all(v[0] == 0 and v[-1] == f.ncol - 1 for v in f.vidF))
check("sub-segment loop sums add up to the whole yarn length", all(abs(_s - ll.sum()) < 1e-9 for _s, ll in zip(f.arc_asb.sum(1), f.l_asb)))
# the colour of the baseline of the single jersey: a flat field is ONE mid colour (no spurious yellow column)
r1 = c.get(f"/api/results/{ID}/colors?field=loop&variant=baseline&kind=axis"); leg = json.loads(r1.headers["X-Legend"]); rgb = np.frombuffer(r1.content, np.uint8).reshape(-1, 3)
mid = np.array(leg["stops"])[2:4].mean(0)
check("baseline legend: range 1 +- 0.01 (the UNSOURCED minimum half-width), as-built mean in the middle", abs(leg["vmin"] - 0.99) < 1e-9 and abs(leg["vmax"] - 1.01) < 1e-9 and leg["ticks"][1] == 1.0, leg)
check("baseline colours: one mid colour everywhere (no loop differs by more than ~6 colour units)", np.abs(rgb.astype(int) - np.median(rgb, 0).astype(int)).max() < 12, np.abs(rgb.astype(int) - np.median(rgb, 0).astype(int)).max())
check("baseline note says geometric arc length and does not show migration", "geometric arc length - does not show migration" in leg["note"])
# ---------------------------------------------------------------- 2. rest-length mode: synthetic relaxed_l0.csv with a KNOWN centre-to-edge gradient (centre about 4 % shorter, corners about 4 % longer, total conserved)
R21 = CAMP / "runs" / "0021"; R21.mkdir(exist_ok=True)
for fn in ("fab_axis.csv", "fab_nodes.csv", "meta.json"): shutil.copy(J / fn, R21 / fn)
sims = json.loads((R21 / "meta.json").read_text()); sims["simulations"] = [dict(params=dict(length_ratio=0.95, tension_cN=10.0, stretch_law="0.01:80,0.03:200,0.06:330"))]; (R21 / "meta.json").write_text(json.dumps(sims))
f0 = rv.fields(R21); ratio = f0.ratio
np.savetxt(R21 / "relaxed_axis.csv", np.vstack([np.c_[np.full(len(X), cc), X] for cc, X in zip(f0.courses, f0.X0)]), delimiter=",", header="course,x,y,z", comments="", fmt=["%d", "%.9f", "%.9f", "%.9f"])   # relaxed = as built (no geometry change)
cx, cy = (f0.ncol - 1) / 2, (f0.nrow - 1) / 2
_yy, _xx = np.mgrid[0:f0.nrow, 0:f0.ncol]; _d = np.hypot((_xx - cx) / cx, (_yy - cy) / cy) / np.sqrt(2); _dm = _d.mean()
def grad(cc, col):                                           # total rest length conserved (the gradient averages to zero): -4 % at the centre of the sheet, +4 % at the corners
    return 0.08 * (np.hypot((col - cx) / cx, (cc - cy) / cy) / np.sqrt(2) - _dm)
rows = []
for i, (cc, l) in enumerate(zip(f0.courses, f0.l_asb)):
    gseg = np.array([grad(cc, v) for v in f0.vid0[i][:-1]]); rows.append(np.c_[np.full(len(l), cc), np.arange(len(l)), l * ratio * (1 + gseg)])
np.savetxt(R21 / "relaxed_l0.csv", np.vstack(rows), delimiter=",", header="course,seg,l0", comments="", fmt=["%d", "%d", "%.12g"])
ID2 = "run~fid_copy~0021"; f2 = rv.fields(R21)
check("saved rest lengths are picked up", f2.rest_src == "saved" and f2.rest_exact)
g = f2.rest_sum / f2.rest0_mean; yy, xx = np.mgrid[0:f2.nrow, 0:f2.ncol]; expect = 1 + grad(yy, xx)
check("restlen field per loop reproduces the synthetic gradient (within the boundary-split error, 0.3 %)", np.abs(g - expect * (f2.arc_asb / f2.mean_asb)).max() < 0.003 and g[5:7, 5:7].mean() < 1 < g[0, 0] and g[0, 0] > g[5, 5] + 0.03, (g.min(), g.max()))
check("rest_rel (colour value) is the synthetic gradient itself, with no loop-type factor (within 0.3 %)", np.abs(f2.rest_rel - expect).max() < 0.003, (f2.rest_rel.min(), f2.rest_rel.max()))
leg2 = c.get(f"/api/results/{ID2}/field-info?field=loop&variant=relaxed").json()
check("field 'loop' picks the rest length for a run with relaxed_l0.csv, and the legend says so", leg2["field"] == "restlen" and "shows migration" in leg2["legend"]["note"] and "does not" not in leg2["legend"]["note"], leg2["legend"]["note"])
r2 = c.get(f"/api/results/{ID2}/colors?field=loop&variant=relaxed&kind=axis"); rgb2 = np.frombuffer(r2.content, np.uint8).reshape(-1, 3); L2 = json.loads(r2.headers["X-Legend"])
# colour of the vertices: centre loops yellow (R,G high, B low), edge loops dark blue (R,G low); correlation with the known gradient is strong and monotonic
vals, fu = rv._vertex_values(f2, "loop", "relaxed"); exp, _ = rv.colour_values(fu, vals, False, 98.0)
check("axis colours returned by the endpoint equal the colour map of the restlen values", np.array_equal(rgb2, exp[np.concatenate([rv.decim_idx(X, f2.R, 1.0, 8.0) + o for X, o in zip(f2.Xr, np.cumsum([0] + [len(x_) for x_ in f2.Xr])[:-1])])]))
centre_idx = [i for i, cc in enumerate(f2.courses) if abs(cc - cy) <= 1]; vc = np.concatenate([f2.vid0[i] for i in centre_idx]); xs = np.concatenate([f2.X0[i][:, 0] for i in centre_idx])
v_all = np.concatenate([np.full(len(X), np.nan) for X in f2.X0]); full_rgb, _ = rv.colour_values(fu, vals, False, 98.0)
off = np.cumsum([0] + [len(X) for X in f2.X0])[:-1]; cen = np.concatenate([full_rgb[o:o + len(X)][(f2.vid0[i] == 5) | (f2.vid0[i] == 6)] for i, (o, X) in enumerate(zip(off, f2.X0)) if abs(f2.courses[i] - cy) <= 0.6])
edge = np.concatenate([full_rgb[o:o + len(X)][(f2.vid0[i] == 0)] for i, (o, X) in enumerate(zip(off, f2.X0)) if f2.courses[i] in (0, f2.nrow - 1)])
check("centre loops are yellowish (high red+green, low blue), corner loops dark blue (low red+green)", cen[:, :2].mean() > 150 and cen[:, 2].mean() < 140 and edge[:, :2].mean() < 90, (cen.mean(0), edge.mean(0)))
cols = np.arange(f2.ncol); prof = (f2.rest_sum[int(cy)] / f2.rest0_mean);
check("the colour value rises monotonically from the middle column to the edge (known gradient)", np.all(np.diff(prof[f2.ncol // 2:]) > -1e-3) and np.all(np.diff(prof[:f2.ncol // 2][::-1]) > -1e-3), prof)
bl = c.get(f"/api/results/{ID2}/colors?field=restlen&variant=baseline&kind=axis"); check("restlen is not defined for the baseline", bl.status_code == 404)
rl = c.get(f"/api/results/run~fid_copy~0001/colors?field=restlen&variant=relaxed&kind=tube"); check("restlen refused for a run without relaxed_l0.csv (honest 404, with the reason)", rl.status_code == 404 and "not saved" in rl.json()["detail"], rl.text)
la = c.get(f"/api/results/run~fid_copy~0001/field-info?field=loop&variant=relaxed").json()
check("a run WITHOUT relaxed_l0.csv: 'loop' falls back to geometric arc length, labelled 'does not show migration'", la["field"] == "looplen" and "does not show migration" in la["legend"]["note"] and any("only through" in n for n in la["notes"]) and any("cannot be used as evidence" in n for n in la["notes"]), la["notes"])
# ---------------------------------------------------------------- 3. colour range: outliers must not compress the rest
R22 = CAMP / "runs" / "0022"; R22.mkdir(exist_ok=True)
for fn in ("fab_axis.csv", "fab_nodes.csv", "meta.json", "relaxed_axis.csv"): shutil.copy(R21 / fn, R22 / fn)
T = np.loadtxt(R21 / "relaxed_l0.csv", delimiter=",", skiprows=1)
idx = (T[:, 0] == f2.courses[3]) & (np.searchsorted(f2.beta[3], np.arange(len(f2.l_asb[3])), side="right") - 1 == 4)[np.minimum(T[:, 1].astype(int), len(f2.l_asb[3]) - 1)]
T[idx, 2] *= 1.5                                    # ONE loop 50 % longer: an extreme outlier
np.savetxt(R22 / "relaxed_l0.csv", T, delimiter=",", header="course,seg,l0", comments="", fmt=["%d", "%d", "%.12g"])
f3 = rv.fields(R22); v3, fu3 = rv._vertex_values(f3, "loop", "relaxed"); rgb3, L3 = rv.colour_values(fu3, v3, False, 98.0)
check("outlier loop (+50 %%) is in the data (max %.2f) but does not set the colour range" % v3.max(), v3.max() > 1.3 and L3["vmax"] < 1.1, L3)
ref = rv.colour_values(fu, vals, False, 98.0)[1]
check("the range with and without the outlier is nearly the same (robust)", abs(L3["half_width"] - ref["half_width"]) < 0.01, (L3["half_width"], ref["half_width"]))
bulk = rgb3[v3 < 1.1]; check("the bulk of the loops still spans many colours (not compressed into one)", len(np.unique(bulk, axis=0)) > 15, len(np.unique(bulk, axis=0)))
old_lo, old_hi = v3.min(), v3.max(); t_old = (np.median(v3) - old_lo) / (old_hi - old_lo)
check("(for contrast) a min-to-max range would have put the median loop at %.2f of the scale, i.e. near one end" % t_old, t_old < 0.2)
print("   legend numbers with the outlier:", {k: round(L3[k], 4) for k in ("vmin", "vmax", "half_width", "data_min", "data_max")})
print("ALL OK" if ok else "FAILURES")
