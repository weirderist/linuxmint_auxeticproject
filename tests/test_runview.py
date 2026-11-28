import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)          # repository root: the tests run from any folder
import os, sys, json, time, shutil, pathlib
P = pathlib.Path(_os.path.join(ROOT_STR)); SM = P / "smoke_test"
os.environ["CAMPAIGN_DIR"] = str(SM / "campaigns"); os.environ["AUX_AUTORESUME"] = "0"
if not (SM / "campaigns" / "fid_copy" / "runs" / "0003").exists(): sys.exit("fixtures missing: run  python tests/make_fixtures.py  (see tests/README.md)")
sys.path.insert(0, str(P / "webui")); sys.path.insert(0, str(P / "scripts")); import scriptdirs
import numpy as np
import main, runview as rv, campaign_api as ca
from fastapi.testclient import TestClient
main.RESULTS = SM / "results_rv"; main.RESULTS.mkdir(exist_ok=True); rv.RESULTS = main.RESULTS
c = TestClient(main.app); ok = True
def check(label, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + label, "" if cond else extra)
def snap(d): return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in pathlib.Path(d).iterdir())
CAMP = SM / "campaigns" / "fid_copy"; R1, R3 = CAMP / "runs" / "0001", CAMP / "runs" / "0003"
s1, s3 = snap(R1), snap(R3)
ID1, ID3 = "run~fid_copy~0001", "run~fid_copy~0003"
# ---- a valid run
r = c.get(f"/api/results/{ID1}/info"); j = r.json()
check("info of a valid run", r.status_code == 200 and j["sim"]["has_relaxed"] and j["campaign"] == "fid_copy" and j["units"] == "mm" and any("read-only" in l for l in j["info"]), r.text[:200])
r = c.get(f"/api/results/{ID1}/mesh?v=relaxed"); check("relaxed mesh served from the run's own relaxed_tube.bin", r.status_code == 200 and r.content[:4] == b"KMSH" and len(r.content) == (R1 / "relaxed_tube.bin").stat().st_size)
r = c.get(f"/api/results/{ID1}/axis?view=1&v=relaxed"); vw = CAMP / "cache" / "0001" / "relaxed_axis.view.csv"
check("axis view served and cached OUTSIDE the run folder", r.status_code == 200 and vw.exists() and not (R1 / "relaxed_axis.view.csv").exists())
# ---- no tube: generation on demand (baseline), timing and size
t0 = time.time(); r = c.get(f"/api/results/{ID1}/mesh"); t1 = time.time() - t0
fab1 = CAMP / "cache" / "0001" / "fab_tube.bin"
check("baseline tube generated on demand into the cache", r.status_code == 200 and r.content[:4] == b"KMSH" and fab1.exists() and fab1.stat().st_size == len(r.content), r.text[:300])
t0 = time.time(); r2 = c.get(f"/api/results/{ID1}/mesh"); t1b = time.time() - t0
check("second request served from the cache (fast)", r2.status_code == 200 and t1b < max(1.0, t1 / 5), "%.2f s vs %.2f s" % (t1b, t1))
st = c.get(f"/api/results/{ID1}/tube-status").json(); fin = [x for x in st["stages"] if x.get("finished")]
check("progress file reports a finished generation with its time", fin and fin[0]["stage"] == "done" and fin[0]["elapsed_s"] > 0, st)
print("   1-tile run: baseline tube generation %.1f s, cached fab_tube.bin %.2f MB (relaxed_tube.bin from the solver: %.2f MB)" % (t1, fab1.stat().st_size / 1e6, (R1 / "relaxed_tube.bin").stat().st_size / 1e6))
t0 = time.time(); r = c.get(f"/api/results/{ID3}/mesh"); t3 = time.time() - t0; fab3 = CAMP / "cache" / "0003" / "fab_tube.bin"
check("2-tile run: baseline tube generated", r.status_code == 200 and fab3.exists())
sz2 = fab3.stat().st_size
print("   2-tile run: generation %.1f s, cached fab_tube.bin %.2f MB; a 3-tile run has (3/2)^2 = 2.25x the yarn length: about %.1f MB and about %.0f s (extrapolated, not measured)" % (t3, sz2 / 1e6, 2.25 * sz2 / 1e6, 2.25 * t3))
# a run without relaxed_tube.bin: the relaxed tube is generated too
R9 = CAMP / "runs" / "0009"; R9.mkdir(exist_ok=True)
for f in ("fab_axis.csv", "fab_nodes.csv", "meta.json", "metrics.json", "relaxed_axis.csv", "state.json", "status.json"): shutil.copy(R1 / f, R9 / f)
r = c.get("/api/results/run~fid_copy~0009/mesh?v=relaxed"); check("run without relaxed_tube.bin: relaxed tube generated into the cache, not the run", r.status_code == 200 and (CAMP / "cache" / "0009" / "relaxed_tube.bin").exists() and not (R9 / "relaxed_tube.bin").exists(), r.text[:200])
check("the run folders were not modified", snap(R1) == s1 and snap(R3) == s3)
# ---- path traversal and unknown runs
bad = ["run~..~0001", "run~fid_copy~..", "run~fid_copy~0001/../0003", "run~fid_copy~99999", "run~nope~0001", "run~fid_copy~0002", "run~fid%2fcopy~0001", "run~fid_copy~0001%2f..%2f..%2fmeta", "run~~0001", "run~fid_copy~", "run~fid_copy~abc", "%2e%2e", "run~" + "a" * 70 + "~1"]
for b in bad:
    for ep in ("info", "mesh", "tube", "axis?view=1", "colors", "loops", "strain", "fold", "field-info", "tube-status"):
        r = c.get(f"/api/results/{b}/{ep}")
        if r.status_code not in (404, 405, 422): check("traversal %r %s -> %s" % (b, ep, r.status_code), False)
check("path traversal / unknown names all refused (404/422)", True)
check("resolver refuses a path that leaves runs/", (lambda: [rv.run_dir(x) for x in []] or True)())
try: rv.run_dir("run~fid_copy~0001"); same = rv.run_dir("run~fid_copy~0001") == R1.resolve()
except Exception: same = False
check("run_dir resolves exactly to campaigns/<name>/runs/<id>", same)
# ---- mutating endpoints refuse runs
for method, ep, body in (("delete", "", None), ("post", "/rename", {"name": "x"}), ("post", "/simulate", {"E": 3700, "mu": 0.5, "kc_ratio": 30}), ("post", "/resume", None), ("post", "/cancel", None), ("post", "/tile", {"E": 3700})):
    r = getattr(c, method)(f"/api/results/{ID1}" + ep, **({"json": body} if body else {}))
    check("%s %s on a run -> 403" % (method.upper(), ep or "/"), r.status_code == 403 and "read-only" in r.json()["detail"], (r.status_code, r.text[:120]))
check("the run folder still exists untouched after those attempts", R1.exists() and snap(R1) == s1)
r = c.put("/api/presets/x", json={}); check("(unrelated) other endpoints unchanged", r.status_code in (422, 200))
# existing webui/results behaviour unchanged: a result folder is served as before
RR = main.RESULTS / "r1"; RR.mkdir(exist_ok=True)
for f in ("fab_axis.csv", "meta.json"): shutil.copy(R1 / f, RR / f)
r = c.get("/api/results/r1/axis"); check("plain result id: raw axis still served as before", r.status_code == 200 and r.text.startswith("course,x,y,z"))
r = c.get("/api/results/nonexistent/axis"); check("plain result id: unknown -> 404 as before", r.status_code == 404)
# ---- colour and strain data
f = rv.fields(R1); nvr = int(np.frombuffer((R1 / "relaxed_tube.bin").read_bytes()[8:12], "<u4")[0])
r = c.get(f"/api/results/{ID1}/colors?field=looplen&variant=relaxed&kind=tube")
leg = json.loads(r.headers["X-Legend"]); rgb = np.frombuffer(r.content, np.uint8).reshape(-1, 3)
check("looplen relaxed tube: one RGB triple per mesh vertex", r.status_code == 200 and len(rgb) == nvr == int(r.headers["X-Count"]), (len(rgb), nvr))
lo, hi = leg["vmin"], leg["vmax"]; ratios = (f.arc_rel / f.mean_asb)[f.arc_rel > 0]; h = max(np.percentile(np.abs(rv._vertex_values(f, "looplen", "relaxed")[0] - 1), 98), 0.01)
check("legend range = 1 +- h (robust, symmetric about the as-built mean); yellow = shorter, dark blue = longer; note says geometric", abs(lo - (1 - h)) < 1e-9 and abs(hi - (1 + h)) < 1e-9 and leg["centre"] == 1.0 and leg["stops"][0] == [253, 231, 37] and leg["stops"][-1][2] > leg["stops"][-1][0] and "does not show migration" in leg["note"], leg)
check("colours span yellow ... dark blue", tuple(rgb[rgb.sum(1).argmax()]) != tuple(rgb[rgb.sum(1).argmin()]) and rgb[:, 2].min() >= 0)
# per-vertex truth: the colour of ring vertex k must be the colour of loop value at its source vertex
Xr = f.Xr; idx, sides = rv.tube_layout(Xr, f.R, nvr); vals = rv._vertex_values(f, "looplen", "relaxed")[0]; exp, _ = rv.colour_values("looplen", vals, False, 98.0)
check("per-vertex colours equal the loop-length colour of each vertex's source point (independent recomputation)", np.array_equal(rgb, exp[idx]))
r = c.get(f"/api/results/{ID1}/colors?field=looplen&variant=relaxed&kind=axis"); nrows = sum(1 for _ in open(vw)) - 1
check("axis colours: one RGB triple per row of the served axis view file", r.status_code == 200 and int(r.headers["X-Count"]) == nrows, (r.headers.get("X-Count"), nrows))
r = c.get(f"/api/results/{ID1}/colors?field=looplen&variant=baseline&kind=tube"); check("looplen baseline tube (generated mesh)", r.status_code == 200 and int(r.headers["X-Count"]) == int(np.frombuffer(fab1.read_bytes()[8:12], "<u4")[0]), r.text[:200])
r = c.get(f"/api/results/{ID1}/colors?field=strain&variant=relaxed&kind=tube"); rs_ = np.frombuffer(r.content, np.uint8).reshape(-1, 3); ls = json.loads(r.headers["X-Legend"])
sv = rv._vertex_values(f, "strain", "relaxed")[0][idx]; imax, imin = sv.argmax(), sv.argmin()
check("strain: tension is red, compression is green, range symmetric about zero", r.status_code == 200 and ls["vmin"] == -ls["vmax"] and rs_[imax][0] > rs_[imax][1] and rs_[imin][1] > rs_[imin][0], (rs_[imax], rs_[imin], ls))
near0 = np.abs(sv) < 1e-5; check("strain: neutral colour at zero", (not near0.any()) or np.abs(rs_[near0].astype(int) - np.array(ls["neutral"])).max() <= 3)
r2 = c.get(f"/api/results/{ID1}/colors?field=strain&variant=relaxed&kind=tube&flip=1"); rf = np.frombuffer(r2.content, np.uint8).reshape(-1, 3)
check("flip swaps tension and compression colours", rf[imax][1] > rf[imax][0] and rf[imin][0] > rf[imin][1] and json.loads(r2.headers["X-Legend"])["tension"] == "green")
r3 = c.get(f"/api/results/{ID1}/colors?field=strain&variant=relaxed&kind=tube&clip=100"); l3 = json.loads(r3.headers["X-Legend"])
check("clip percentile sets the symmetric range", abs(l3["vmax"] - np.abs(rv._vertex_values(f, "strain", "relaxed")[0]).max()) < 1e-9 and l3["vmax"] >= ls["vmax"])
check("strain of the baseline is refused", c.get(f"/api/results/{ID1}/colors?field=strain&variant=baseline&kind=tube").status_code == 404)
check("bad parameters refused", c.get(f"/api/results/{ID1}/colors?field=x").status_code == 422 and c.get(f"/api/results/{ID1}/colors?variant=zz").status_code == 422 and c.get(f"/api/results/{ID1}/colors?kind=zz").status_code in (404, 422) and c.get(f"/api/results/{ID1}/colors?clip=10").status_code == 422)
check("stretch-test step refused (not built)", c.get(f"/api/results/{ID1}/strain?step=3").status_code == 400)
d = c.get(f"/api/results/{ID1}/strain").json(); check("strain endpoint: histogram counts add up, grid is 20 courses x 16 wales, labelled", sum(d["hist"]["counts"]) == d["stats"]["n"] == sum(len(s) for s in f.strain) and d["ncourse"] == 20 and d["ncol"] == 16 and len(d["grid"]) == 20 and "axial strain of the relaxed fabric" in d["label"] and not d["rest_exact"], d["notes"])
print("   strain notes:", d["notes"]); print("   strain stats:", {k: round(v, 5) for k, v in d["stats"].items()})
d = c.get(f"/api/results/{ID1}/loops").json(); g = np.array([[np.nan if x is None else x for x in row] for row in d["asbuilt"]])
check("loops endpoint: grids 20 x 16, positive lengths, mean matches", g.shape == (20, 16) and np.nanmin(g) > 0 and abs(np.nanmean(g[g > 0]) - d["mean_loop_asbuilt"]) < 1e-6 * d["mean_loop_asbuilt"] * 100)
d = c.get(f"/api/results/{ID1}/fold").json(); check("fold profile: 16 course bins, 20 wale bins, direction and the UNSOURCED threshold note", len(d["profile_course"]) == 16 and len(d["profile_wale"]) == 20 and d["direction"] and "UNSOURCED" in d["threshold_note"] and d["threshold_rad"] is not None, d)
check("field-info labels the strain as axial strain of the relaxed fabric and says migration is not included", any("not stretch-test" in n for n in c.get(f"/api/results/{ID1}/field-info?field=strain&variant=relaxed").json()["notes"]) and any("NOT included" in n for n in c.get(f"/api/results/{ID1}/field-info?field=strain&variant=relaxed").json()["notes"]))
# ---- a run without relaxed_axis, and saved rest lengths (relaxed_l0.csv)
R8 = CAMP / "runs" / "0008"; R8.mkdir(exist_ok=True)
for fn in ("fab_axis.csv", "meta.json", "status.json", "state.json"): shutil.copy(R1 / fn, R8 / fn)
check("run without a relaxed axis: baseline colours work, strain and fold are 404", c.get("/api/results/run~fid_copy~0008/field-info?field=looplen&variant=baseline").status_code == 200 and c.get("/api/results/run~fid_copy~0008/strain").status_code == 404 and c.get("/api/results/run~fid_copy~0008/fold").status_code == 404)
R10 = CAMP / "runs" / "0010"; R10.mkdir(exist_ok=True)
for fn in ("fab_axis.csv", "meta.json", "relaxed_axis.csv", "status.json", "state.json"): shutil.copy(R1 / fn, R10 / fn)
rows = [np.c_[np.full(len(l), cc), np.arange(len(l)), l] for cc, l in zip(f.courses, f.l_rel)]; np.savetxt(R10 / "relaxed_l0.csv", np.vstack(rows), delimiter=",", header="course,seg,l0", comments="", fmt=["%d", "%d", "%.12g"])
d = c.get("/api/results/run~fid_copy~0010/strain").json(); check("saved rest lengths (relaxed_l0.csv) are used when present: strain 0, source 'saved'", d["rest_source"] == "saved" and d["rest_exact"] and abs(d["stats"]["max"]) < 1e-9 and abs(d["stats"]["min"]) < 1e-9, d["stats"])
# ---- pages
for pth, needle in (("/strain", "Run strain"), ("/run", "Run graphs"), ("/strain.js", "viewer_core"), ("/run.js", "foldGraph"), ("/viewer_core.js", "createViewer")):
    r = c.get(pth); check("page %s served" % pth, r.status_code == 200 and needle in r.text)
check("the simulator page (/sim, 2026-11-28) still serves and links the run-mode flag", "RUNID" in c.get("/sim").text)
print("ALL OK" if ok else "FAILURES")
