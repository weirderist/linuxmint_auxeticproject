"""Tests for the starting loop-length field (scripts/loop_grade.py, relax_job --loop-grade) WITHOUT the solver and without JAX.
Run from the repository root:  python tests/test_loop_grade.py     (needs the fixtures of tests/make_fixtures.py)
What this proves and what it does not is written in tests/README.md (section 'test_loop_grade')."""
import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)          # repository root: the tests run from any folder
import ast, json, os, shutil, sys, pathlib
P = pathlib.Path(ROOT_STR); SM = P / "smoke_test"; FID = SM / "campaigns" / "fid_copy" / "runs"
os.environ["CAMPAIGN_DIR"] = str(SM / "campaigns"); os.environ["CAMPAIGN_SCRIPTS"] = str(P / "scripts"); os.environ["AUX_AUTORESUME"] = "0"
for r in ("0001", "0003", "0020"):
    if not (FID / r).exists(): sys.exit("fixtures missing: run  python tests/make_fixtures.py  (see tests/README.md)")
sys.path.insert(0, str(P / "scripts")); sys.path.insert(0, str(P / "webui")); sys.path.insert(0, str(P / "tests"))
import numpy as np
from scipy import ndimage
import loop_grade as lg, relax_state as rs, campaign as cm
import analysis_loop_grade_design as proto            # the independent prototype (scipy based) the module is tested against
import main, runview as rv
from fastapi.testclient import TestClient
ok = True
def check(label, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + label, "" if cond else extra)

P1 = np.array([[int(c) for c in r.replace(" ", "")] for r in (P / "campaigns" / "pattern1_matrix.txt").read_text().strip().split("/")])
JER = np.ones((1, 1), int)

def scratch(name, src):
    """A scratch copy of a fixture run (fab_axis.csv and meta.json only): relax_state.load writes state.json into its folder, never into the fixtures."""
    d = SM / "grade_tmp" / name; d.mkdir(parents=True, exist_ok=True)
    for fn in ("fab_axis.csv", "meta.json"): shutil.copy(FID / src / fn, d / fn)
    return d, json.loads((d / "meta.json").read_text())

def load_sim(name, src, ratio=0.9488):
    d, meta = scratch(name, src); return rs.load(str(d), rs.Params(rest_length_ratio=ratio)), d, meta

# ---------------------------------------------------------------- 1. sections on the matrix AS BUILT (tiled): do tile joins merge sections?
COUNTS = {}
for tiles, exp_n, exp_sizes in ((1, 3, [60, 100, 160]), (2, 6, [60, 60, 200, 320, 320, 320]), (3, 9, [60, 60, 60, 300, 480, 480, 480, 480, 480])):
    T = lg.tiled_matrix(P1, tiles); lab, n = lg.label_sections(T); sizes = sorted(int((lab == k).sum()) for k in range(1, n + 1)); COUNTS[tiles] = n
    check("Pattern 1 at tiles %d: %d sections, sizes %s (a tile repeat alone would give %d)" % (tiles, n, sizes, 3 * tiles * tiles), n == exp_n and sizes == exp_sizes, (n, sizes))
check("tile joins MERGE sections: the count grows as 3 x tiles (%s), far below the 3 x tiles^2 of separate tiles (3, 12, 27)" % COUNTS, COUNTS == {1: 3, 2: 6, 3: 9} and COUNTS[2] < 12 and COUNTS[3] < 27)
for tiles in (1, 2, 8, 12): check("single jersey at tiles %d: one section of %d loops" % (tiles, tiles * tiles), lg.label_sections(lg.tiled_matrix(JER, tiles))[1] == 1)
for tiles in (1, 3):
    T = lg.tiled_matrix(P1, tiles); lab, n = lg.label_sections(T); ref = np.zeros(T.shape, int); m = 0
    for t in (1, 2):
        l, k = ndimage.label(T == t); ref[l > 0] = l[l > 0] + m; m += k
    same = n == m and all(len(set(ref[lab == k].tolist())) == 1 and int((lab == k).sum()) == int((ref == ref[lab == k][0]).sum()) for k in range(1, n + 1))
    check("section labelling equals scipy.ndimage.label (tiles %d)" % tiles, same)
    d_ok = all(np.allclose(lg.dist_to_outside(lab == k)[lab == k], ndimage.distance_transform_edt(np.pad(lab == k, 1))[1:-1, 1:-1][lab == k]) for k in range(1, n + 1))
    check("distance to the section boundary equals scipy's Euclidean distance transform (tiles %d)" % tiles, d_ok)

# ---------------------------------------------------------------- 2. strengths and the dilution case
sw, tw, sc, tc = lg.strengths(P1)
check("Pattern 1 strengths: wales 6, 7, 14, 15 = 1, every other wale 0, every course 0", [i for i, s in enumerate(sw) if s > 0] == [6, 7, 14, 15] and np.allclose(sw[[6, 7, 14, 15]], 1) and not sc.any(), (sw, sc))
check("Pattern 1 bands: wales 6-7 (type 2) and 14-15 (type 1), no course band", lg.bands(sw, tw, 0.2) == [(6, 7, 2, 1.0), (14, 15, 1, 1.0)] and lg.bands(sc, tc, 0.2) == [])
gains = []
for k in (None, 10, 5, 3, 2):
    D = P1.copy()
    if k:
        for w in (6, 7, 14, 15):
            for r in range(k - 1, P1.shape[0], k): D[r, w] = 3 - D[r, w]
    L, lab, ns = lg.field_equal_weights(D, 0.02, "tm"); sel = [(r, w) for w in (6, 7, 14, 15) for r in range(P1.shape[0]) if D[r, w] == P1[r, w]]
    gains.append(100 * (np.mean([L[r, w] for r, w in sel]) - 1)); s_ = lg.strengths(D)[0][6]
    check("dilution %s: axis strength %.2f (expected %.2f = 2 (f - 0.5))" % ("pure" if not k else "1 in %d" % k, s_, {None: 1.0, 10: 0.8, 5: 0.6, 3: 0.4, 2: 0.0}[k]), abs(s_ - {None: 1.0, 10: 0.8, 5: 0.6, 3: 0.4, 2: 0.0}[k]) < 1e-9)
check("dilution: the TM-only gain on the axis loops falls as the axis is diluted and is 0 at 1 in 2 (%s %%)" % ", ".join("%.2f" % g for g in gains), all(a > b - 1e-12 for a, b in zip(gains, gains[1:])) and abs(gains[-1]) < 1e-12 and gains[0] > 0.5, gains)
check("min strength 0.2 decides: at 1 loop in 3 (s 0.4) the axis counts; with min strength 0.5 it does not", lg.bands(np.array([0, 0.4, 0.4]), np.array([1, 2, 2]), 0.2) != [] and lg.bands(np.array([0, 0.4, 0.4]), np.array([1, 2, 2]), 0.5) == [])

# ---------------------------------------------------------------- 3. the module equals the independent prototype (all modes)
same = True
for name, M in (("jersey 8x8", np.ones((8, 8), int)), ("Pattern 1 tiles 1", P1)):
    for mode in ("boundary", "tm", "both"):
        L1, lab1, n1 = lg.field_equal_weights(M, 0.02, mode); L2, lab2, n2 = proto.field(M, mode, 0.02); same &= np.allclose(L1, L2) and n1 == n2
check("scripts/loop_grade.py reproduces the prototype field exactly (jersey and Pattern 1, boundary / tm / both)", same)
check("per-section equal-weight conservation of the loop field is exact", all(abs(lg.field_equal_weights(P1, 0.02, m)[0][lg.label_sections(P1)[0] == k].mean() - 1) < 1e-12 for m in ("both",) for k in (1, 2, 3)))

# ---------------------------------------------------------------- 4. shared loop identity
sim0, d0, meta0 = load_sim("same_resample", "0001")
courses = {int(y.course): y for y in sim0.yarns}; A = np.loadtxt(d0 / "fab_axis.csv", delimiter=",", skiprows=1)
check("lg.resample equals relax_state.resample on every course of a real fixture", all(np.array_equal(lg.resample(A[A[:, 0] == c][:, 1:], meta0["R"]), rs.resample(A[A[:, 0] == c][:, 1:], meta0["R"])) for c in courses))
check("the viewer (runview.py) uses the same loop code as the field (one loop identity)", rv._boundaries is lg.boundaries and rv._loop_sum is lg.loop_sum and rv.resample is lg.resample and rv.lg is lg)

# ---------------------------------------------------------------- 5. OPTION OFF: bit-identical start
sim, d, meta = load_sim("off", "0001"); ratio = 0.9488
indep = [np.linalg.norm(np.diff(y.X, axis=0), axis=1) * ratio for y in sim.yarns]
check("option off: the starting l0 equals the plain uniform formula chord x ratio, bit for bit (every segment of every course)", all(np.array_equal(y.l0, e) for y, e in zip(sim.yarns, indep)))
src = (P / "scripts" / "relax_job.py").read_text(encoding="utf8"); tree = ast.parse(src)
par = {}
for node in ast.walk(tree):
    for ch in ast.iter_child_nodes(node): par[ch] = node
def guarded_by(node, name):
    while node in par:
        node = par[node]
        if isinstance(node, ast.If) and isinstance(node.test, (ast.Attribute, ast.Name)) and getattr(node.test, "attr", getattr(node.test, "id", "")) == name: return True
    return False
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "apply_start_grade"]
check("relax_job.py calls apply_start_grade exactly once, only inside 'if g.loop_grade:' (so with 0 nothing runs)", len(calls) == 1 and guarded_by(calls[0], "loop_grade"))
upd = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "update" and isinstance(n.value, ast.Subscript) and "params" in ast.dump(n.value)]
check("the meta.json record (params and loop_grade block) is written only inside 'if grade_rec:'; the slip-extension record only inside 'if ext_rec:' (a run without the options writes what it wrote before)", len(upd) >= 2 and sum(guarded_by(u, "grade_rec") for u in upd) == 1 and sum(guarded_by(u, "ext_rec") for u in upd) == 1 and all(guarded_by(u, "grade_rec") or guarded_by(u, "ext_rec") for u in upd) and "run_entry['loop_grade'] = grade_rec" in src)
adds = {a.args[0].value: a for a in ast.walk(tree) if isinstance(a, ast.Call) and getattr(a.func, "attr", "") == "add_argument" and a.args and isinstance(a.args[0], ast.Constant)}
check("--loop-grade defaults to 0.0, --loop-grade-target to 'both' (choices boundary/tm/both), --loop-grade-min-strength to 0.2",
      all(k in adds for k in ("--loop-grade", "--loop-grade-target", "--loop-grade-min-strength")) and ast.literal_eval({k.arg: k.value for k in adds["--loop-grade"].keywords}["default"]) == 0.0
      and ast.literal_eval({k.arg: k.value for k in adds["--loop-grade-target"].keywords}["default"]) == "both" and ast.literal_eval({k.arg: k.value for k in adds["--loop-grade-target"].keywords}["choices"]) == ["boundary", "tm", "both"]
      and ast.literal_eval({k.arg: k.value for k in adds["--loop-grade-min-strength"].keywords}["default"]) == 0.2)
def help_of(flag): return ast.literal_eval({k.arg: k.value for k in adds[flag].keywords}["help"])
check("the help of --loop-grade, --loop-grade-target and --loop-grade-min-strength each says UNSOURCED; the --creep-tau help says creep does NOT conserve the total rest length",
      all("UNSOURCED" in help_of(f) for f in ("--loop-grade", "--loop-grade-target", "--loop-grade-min-strength")) and "does NOT conserve the total" in help_of("--creep-tau") and "UNSOURCED" in src.split('"""')[1])
check("the grade is no longer refused with --shell: that restriction is gone from relax_job.py (the other 'no shell' options keep theirs)", "--loop-grade is for runs without the shell" not in src and "--ramp is for runs without the shell" in src)

# ---------------------------------------------------------------- 6. OPTION ON: conservation, sign, shape
def loop_rest(sim, meta):
    """rest length of every loop (course x wale) from the segments' l0, boundaries split in proportion (the viewer's rest-length measure)."""
    st = meta["structure"]; ncol = len(meta["matrix"][0]) * meta["tiles"]; A = np.loadtxt(os.path.join(sim.folder, "fab_axis.csv"), delimiter=",", skiprows=1)
    out = []
    for y in sorted(sim.yarns, key=lambda y: y.course):
        b, beta = lg.course_boundaries(A[A[:, 0] == y.course][:, 1:], y.X, st["W"], ncol); out.append(lg.loop_sum(y.l0, beta))
    return np.array(out)
for label, src_run, a in (("single jersey 12x12", "0020", 0.02), ("Pattern 1 tiles 1", "0001", 0.02), ("Pattern 1 tiles 2", "0003", 0.02)):
    sim, d, meta = load_sim("on_" + src_run, src_run); before = [y.l0.copy() for y in sim.yarns]
    rec = lg.apply_start_grade(sim, str(d), meta, a, "both", 0.2)
    worst = max(abs(s["mean_factor"] - 1) for s in rec["sections"]); worst_tot = max(abs(s["l0_after"] / s["l0_before"] - 1) for s in rec["sections"])
    check("%s: %d sections; every section's total rest length is conserved exactly (worst relative deviation %.1e)" % (label, rec["sections_count"], worst_tot), worst_tot < 1e-12 and worst < 1e-12 and abs(rec["total_l0_after"] / rec["total_l0_before"] - 1) < 1e-12, (worst, worst_tot))
    check("%s: all factors positive, within %.4f to %.4f; a, target, minimum strength and the UNSOURCED list are in the record" % (label, rec["factor_min"], rec["factor_max"]), rec["factor_min"] > 0.9 and rec["factor_max"] < 1.1 and rec["a"] == a and rec["target"] == "both" and rec["min_strength"] == 0.2 and len(rec["unsourced"]) >= 8 and json.loads(json.dumps(rec)) == rec)
    if src_run == "0020":
        Lr = loop_rest(sim, meta); Lr = Lr / Lr.mean(); n = Lr.shape[0]; ring = [(r, c) for r in range(n) for c in range(n) if min(r, c, n - 1 - r, n - 1 - c) == 0]; core = [(r, c) for r in range(n) for c in range(n) if min(r, c, n - 1 - r, n - 1 - c) == n // 2 - 1]
        diff = np.mean([Lr[x] for x in ring]) - np.mean([Lr[x] for x in core])
        check("single jersey: the loops of the free edge are LONGER than the centre loops by about a = 0.02 (measured %.4f)" % diff, 0.015 < diff < 0.025, diff)
        row = Lr[n // 2]; half = row[n // 2:]
        check("single jersey: along the middle course the rest length rises monotonically from the centre to the edge", all(half[i + 1] >= half[i] - 1e-3 for i in range(len(half) - 1)) and half[-1] > half[0] + 0.01, half)
        check("single jersey: symmetric (left-right and top-bottom) to within the segment-boundary split", np.allclose(Lr, Lr[::-1], atol=3e-3) and np.allclose(Lr, Lr[:, ::-1], atol=3e-3))
    if src_run == "0001":
        Lu = loop_rest(load_sim("uniform_p1", src_run)[0], meta)           # each loop's own uniform start: Pattern 1's loops already differ by loop type (as built), the grade is measured against that
        Rg = loop_rest(sim, meta) / Lu; wale = Rg.mean(0); top = sorted(np.argsort(wale)[-4:].tolist())
        check("Pattern 1: relative to each loop's uniform start, the four longest wales on average are the TM axes 6, 7, 14, 15 (mean by wale: %s)" % np.round(wale, 4).tolist(), top == [6, 7, 14, 15], top)
        sim_b, _, _ = load_sim("on_b", src_run); lg.apply_start_grade(sim_b, str(d), meta, 0.02, "boundary", 0.2); sim_t, _, _ = load_sim("on_t", src_run); lg.apply_start_grade(sim_t, str(d), meta, 0.02, "tm", 0.2)
        wt = (loop_rest(sim_t, meta) / Lu).mean(0); wb = (loop_rest(sim_b, meta) / Lu).mean(0)
        check("Pattern 1 target 'tm': the axis wales are the longest; target 'boundary': they are not singled out", sorted(np.argsort(wt)[-4:].tolist()) == [6, 7, 14, 15] and sorted(np.argsort(wb)[-4:].tolist()) != [6, 7, 14, 15])
        sim_o, _, _ = load_sim("on_o", src_run); ro = lg.apply_start_grade(sim_o, str(d), meta, 0.0, "both", 0.2)
        check("a = 0 through the function: every factor is exactly 1 and l0 unchanged (belt and braces; relax_job does not even call it)", ro["factor_min"] == 1.0 and ro["factor_max"] == 1.0 and all(np.array_equal(y.l0, b) for y, b in zip(sim_o.yarns, before)))
# shell compatibility: the field does not depend on the shell flag
s1, d1, m1 = load_sim("sh1", "0001"); s2, d2, m2 = load_sim("sh2", "0001"); r1 = lg.apply_start_grade(s1, str(d1), m1, 0.02, "both", 0.2, with_shell=False); r2 = lg.apply_start_grade(s2, str(d2), m2, 0.02, "both", 0.2, with_shell=True)
check("with the shell on, the starting field is identical (the grade reads the as-built geometry and only changes l0); the record says so (with_shell %s -> %s)" % (r1["with_shell"], r2["with_shell"]), all(np.array_equal(a.l0, b.l0) for a, b in zip(s1.yarns, s2.yarns)) and r2["with_shell"] and not r1["with_shell"])
# guards
s3, d3, m3 = load_sim("guard", "0001"); old = lg.FACTOR_FLOOR; lg.FACTOR_FLOOR = 0.999
try: lg.apply_start_grade(s3, str(d3), m3, 0.02, "both", 0.2); raised = False
except ValueError as e: raised = "too large" in str(e)
lg.FACTOR_FLOOR = old; check("a grade that would shrink a segment below the factor floor is refused with a plain message", raised)
bad = dict(m3); bad.pop("matrix")
try: lg.apply_start_grade(load_sim("guard2", "0001")[0], str(d3), bad, 0.02, "both", 0.2); raised = False
except ValueError as e: raised = "needs the stitch matrix" in str(e)
check("a result without the stitch matrix in meta.json is refused with a plain message", raised)
try: lg.loop_G(P1, "nonsense"); raised = False
except ValueError: raised = True
check("an unknown target mode is refused", raised)

# ---------------------------------------------------------------- 7. command line: campaign.py picks the flags up from relax_job.py (no hand edits), dry-run printout
flags = cm.known_relax_flags()
check("campaign.py's allowed-flags logic finds --loop-grade, --loop-grade-target, --loop-grade-min-strength in relax_job.py", {"--loop-grade", "--loop-grade-target", "--loop-grade-min-strength"} <= flags)
spec_dir = SM / "campaigns" / "gradespec"; spec_dir.mkdir(parents=True, exist_ok=True)
spec = json.loads((P / "campaigns" / "selftest" / "spec.json").read_text()); spec["name"] = "gradespec"; spec["relax"].update(loop_grade=0.02, loop_grade_target="tm", loop_grade_min_strength=0.3)
spec["vary"].append(dict(name="loop_grade", min=0.0, max=0.05, scale="linear", integer=False)); spec["relax"].pop("perturb", None)
(spec_dir / "spec.json").write_text(json.dumps(spec))
try: sp = cm.load_spec("gradespec"); loaded = True
except SystemExit as e: loaded = False; print("   load_spec:", e)
check("a campaign spec may set loop_grade, loop_grade_target, loop_grade_min_strength in 'relax' and vary loop_grade (load_spec accepts it)", loaded)
if loaded:
    spec2 = json.loads(json.dumps(sp)); cmd = cm.relax_cmd(dict(spec2["relax"], loop_grade=0.02), 0.13, "/x")
    check("the dry-run command line (relax_cmd) carries the three flags with their values", cmd[cmd.index("--loop-grade") + 1] == "0.02" and cmd[cmd.index("--loop-grade-target") + 1] == "tm" and cmd[cmd.index("--loop-grade-min-strength") + 1] == "0.3", cmd)
    off = cm.relax_cmd({k: v for k, v in spec2["relax"].items() if not k.startswith("loop_grade")}, 0.13, "/x")
    check("without the keys the command line has no loop-grade flag (unchanged)", not any("loop-grade" in c for c in off))

# ---------------------------------------------------------------- 8. web UI backend: SimBody, _sim_cmd, job_request.json round trip, presets
main.PRESET_FILE = SM / "presets_grade.json"; main.RESULTS = SM / "results_grade"; main.RESULTS.mkdir(exist_ok=True); rv.RESULTS = main.RESULTS
launched = []; main._launch = lambda rid, d, cmd, resume=False: launched.append((rid, cmd, resume))
rg = main.RESULTS / "rg"; rg.mkdir(exist_ok=True); shutil.copy(FID / "0020" / "meta.json", rg / "meta.json"); shutil.copy(FID / "0020" / "fab_axis.csv", rg / "fab_axis.csv")
body0 = {"E": 3700, "mu": 0.49, "kc_ratio": 30}; sb0 = main.SimBody(**body0)
cmd0, _ = main._sim_cmd(rg, sb0, 0.13)
check("SimBody defaults: grade 0, target 'both', minimum strength 0.2; with the grade off the relax_job command contains NO loop-grade flag (unchanged)", sb0.loop_grade == 0.0 and sb0.loop_grade_target == "both" and sb0.loop_grade_min_strength == 0.2 and not any("loop-grade" in c for c in cmd0))
sb1 = main.SimBody(**body0, loop_grade=0.02, loop_grade_target="tm", loop_grade_min_strength=0.3); cmd1, _ = main._sim_cmd(rg, sb1, 0.13)
ig = cmd1.index("--loop-grade")
check("with the grade on the command has --loop-grade 0.02 --loop-grade-target tm --loop-grade-min-strength 0.3, and the rest of the command is unchanged", cmd1[ig:ig + 6] == ["--loop-grade", "0.02", "--loop-grade-target", "tm", "--loop-grade-min-strength", "0.3"] and cmd1[:ig] + cmd1[ig + 6:] == cmd0)
for bad_body in ({"loop_grade": 2.0}, {"loop_grade": -0.5}, {"loop_grade_target": "diagonal"}, {"loop_grade_min_strength": 1.5}):
    try: main.SimBody(**body0, **bad_body); raised = False
    except Exception: raised = True
    check("SimBody refuses %s" % bad_body, raised)
c = TestClient(main.app)
r = c.post("/api/results/rg/simulate", json=dict(body0, loop_grade=0.02, loop_grade_target="both", loop_grade_min_strength=0.25, shell=False)); jr = json.loads((rg / "job_request.json").read_text())
check("simulate with the grade: 200, the command goes to the job, and job_request.json records the grade (round trip through SimBody)", r.status_code == 200 and launched and "--loop-grade" in launched[-1][1] and jr["loop_grade"] == 0.02 and jr["loop_grade_target"] == "both" and jr["loop_grade_min_strength"] == 0.25 and main.SimBody(**jr).model_dump() == jr, (r.status_code, r.text[:200]))
sbj = main.SimBody(**jr); cmdr, _ = main._sim_cmd(rg, sbj, 0.13); check("a power-cut resume rebuilds the same command (loop-grade flags included) from job_request.json", cmdr[cmdr.index("--loop-grade"):cmdr.index("--loop-grade") + 6] == ["--loop-grade", "0.02", "--loop-grade-target", "both", "--loop-grade-min-strength", "0.25"])
launched.clear(); r = c.post("/api/results/rg/simulate", json=dict(body0, loop_grade=0.02, shell=True))
check("the grade is NOT refused together with the shell any more (no 422 about the shell)", not (r.status_code == 422 and "shell" in r.text and "grade" in r.text.lower()), r.text[:200])
nm = main.RESULTS / "rn"; nm.mkdir(exist_ok=True); mj = json.loads((rg / "meta.json").read_text()); mj.pop("matrix"); (nm / "meta.json").write_text(json.dumps(mj))
r = c.post("/api/results/rn/simulate", json=dict(body0, loop_grade=0.02)); check("a result without the stitch matrix gets a plain 422 for a non-zero grade", r.status_code == 422 and "stitch matrix" in r.json()["detail"], r.text[:200])
r = c.post("/api/results/rn/simulate", json=body0); check("...but is fine with the grade off (as before)", r.status_code in (200, 422) and "grade" not in r.text.lower() and "stitch matrix" not in r.text, r.text[:200])
r = c.put("/api/presets/gradetest", json=dict(body0, loop_grade=0.03, loop_grade_target="boundary")); pr = c.get("/api/presets").json().get("gradetest", {})
check("a saved preset keeps the grade settings", r.status_code == 200 and pr.get("loop_grade") == 0.03 and pr.get("loop_grade_target") == "boundary" and pr.get("loop_grade_min_strength") == 0.2)
html = (P / "webui" / "index.html").read_text(encoding="utf8")
check("the Simulate form has the three fields (a, target, minimum strength), their defaults, their save/load mapping, the UNSOURCED labels and the creep warning",
      all(x in html for x in ('id="fGrade"', 'id="fGradeT"', 'id="fGradeS"', "loop_grade: ['#fGrade', 'n']", "loop_grade_target: ['#fGradeT', 's']", "loop_grade_min_strength: ['#fGradeS', 'n']", "$('#fGradeS').value = 0.2")) and html.count("UNSOURCED") >= 4 and "Creep (above) moves rest lengths without conserving the total" in html)

# ---------------------------------------------------------------- 9. the viewer shows the graded rest length (synthetic relaxed_l0.csv made from the field)
def make_run(rn, src, a, target="both"):
    sim, d, meta = load_sim("viewer_" + rn, src, ratio=0.9488)          # the viewer reconstructs the same ratio from the run's simulation parameters written just below
    R = FID / rn; R.mkdir(exist_ok=True)
    for fn in ("fab_axis.csv", "fab_nodes.csv", "meta.json"): shutil.copy(FID / src / fn, R / fn)
    mj = json.loads((R / "meta.json").read_text()); mj["simulations"] = [dict(params=dict(length_ratio=0.9488, tension_cN=0.0, stretch_law="", ea=0.0, E=1.0, stretch_scale=1.0))]; (R / "meta.json").write_text(json.dumps(mj))
    lg.apply_start_grade(sim, str(d), meta, a, target, 0.2)
    np.savetxt(R / "relaxed_axis.csv", np.vstack([np.c_[np.full(len(y.X), y.course), y.X] for y in sim.yarns]), delimiter=",", header="course,x,y,z", comments="", fmt=["%d", "%.9f", "%.9f", "%.9f"])
    np.savetxt(R / "relaxed_l0.csv", np.vstack([np.c_[np.full(len(y.l0), y.course), np.arange(len(y.l0)), y.l0] for y in sim.yarns]), delimiter=",", header="course,seg,l0", comments="", fmt=["%d", "%d", "%.12g"])
    return R
make_run("0030", "0020", 0.02); make_run("0031", "0001", 0.02)
fi = c.get("/api/results/run~fid_copy~0030/field-info?field=loop&variant=relaxed").json()
check("viewer, graded single jersey: the 'loop' field is the REST length and the legend says it shows migration", fi["field"] == "restlen" and "shows migration" in fi["legend"]["note"] and fi["rest_source"] == "saved", fi)
f30 = rv.fields(FID / "0030"); vals, fu = rv._vertex_values(f30, "loop", "relaxed"); g = f30.rest_sum / f30.rest0_mean; n = g.shape[0]
ring = np.mean([g[r, c_] for r in range(n) for c_ in range(n) if min(r, c_, n - 1 - r, n - 1 - c_) == 0]); core = np.mean([g[r, c_] for r in range(n) for c_ in range(n) if min(r, c_, n - 1 - r, n - 1 - c_) == n // 2 - 1])
check("viewer, graded single jersey: the rest-length colouring values are lower in the centre and higher at the edges (edge %.4f, centre %.4f)" % (ring, core), ring - core > 0.015)
rgb, leg = rv.colour_values(fu, vals, False, 98.0); rgb_axis = np.frombuffer(c.get("/api/results/run~fid_copy~0030/colors?field=loop&variant=relaxed&kind=axis").content, np.uint8).reshape(-1, 3)
check("viewer: the endpoint serves one colour per vertex and the colours differ between centre and edge loops (yellow centre, darker edge)", len(rgb_axis) > 0 and len(np.unique(rgb_axis, axis=0)) > 5)
f31 = rv.fields(FID / "0031"); w31 = (f31.rest_sum / (f31.ratio * f31.arc_asb)).mean(0)          # relative to each loop's own uniform start
check("viewer, graded Pattern 1: the axis wales 6, 7, 14, 15 have the longest rest length relative to the start (mean by wale %s)" % np.round(w31, 4).tolist(), sorted(np.argsort(w31)[-4:].tolist()) == [6, 7, 14, 15])
wabs = (f31.rest_sum / f31.rest0_mean).mean(0)
check("viewer: the rest-length colouring on a mixed pattern ALSO carries the as-built differences between loop types (%.1f %% spread of the as-built loop length), so the legend/notes must say so" % (100 * (f31.arc_asb.max() - f31.arc_asb.min()) / f31.mean_asb), (f31.arc_asb.max() - f31.arc_asb.min()) / f31.mean_asb > 0.03 and not sorted(np.argsort(wabs)[-4:].tolist()) == [6, 7, 14, 15])
fi31 = c.get("/api/results/run~fid_copy~0031/field-info?field=loop&variant=relaxed").json()
check("the field-info notes state that the as-built loop-type differences are included and give their size", any("as-built" in n and "differ" in n and "spread" in n for n in fi31["notes"]), fi31["notes"])
old_run = c.get("/api/results/run~fid_copy~0001/field-info?field=loop&variant=relaxed").json()
check("an old run without relaxed_l0.csv still gets 'geometric arc length - does not show migration'", old_run["field"] == "looplen" and "does not show migration" in old_run["legend"]["note"])
print("ALL OK" if ok else "FAILURES")
