import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)          # repository root: the tests run from any folder
import os, sys, json, logging, pathlib, time
P = _os.path.join(ROOT_STR); SM = pathlib.Path(P) / "smoke_test"
os.environ["CAMPAIGN_DIR"] = str(SM / "campaigns"); os.environ.pop("AUX_AUTORESUME", None)
sys.path.insert(0, os.path.join(P, "webui"))
import main, campaign_api as ca
from fastapi.testclient import TestClient
main._boot_id = lambda: "CURBOOT"
launched = []; main._launch = lambda rid, d, cmd, resume=False: launched.append((rid, resume))
ALIVE = set(); ca._cm().alive = lambda att: bool(att) and att.get("pid") in ALIVE
lock = SM / "campaigns" / "smoke1" / "runner.lock"
class H(logging.Handler):
    def __init__(s): super().__init__(); s.m = []
    def emit(s, r): s.m.append(r.getMessage())
h = H(); lg = logging.getLogger("uvicorn.error"); lg.addHandler(h); lg.setLevel(logging.INFO)
ok = True
def mk(case, boot, cancelled=False, rid="r"):
    main.RESULTS = SM / "results_boot" / case; d = main.RESULTS / rid; d.mkdir(parents=True, exist_ok=True)
    json.dump({"R": 0.1, "structure": {"units": "mm"}, "simulations": [{"status": "cancelled" if cancelled else "running"}]}, open(d / "meta.json", "w"))
    json.dump({"state": "relaxing"}, open(d / "state.json", "w")); (d / "checkpoint.npz").write_bytes(b"x")
    json.dump({"E": 3700, "mu": 0.5, "kc_ratio": 30}, open(d / "job_request.json", "w"))
    jj = {"started": 1.0}
    if boot: jj["boot_id"] = boot
    json.dump(jj, open(d / "job.json", "w"))
def run(label, expect, runner=False):
    global ok; launched.clear(); h.m.clear()
    json.dump(dict(pid=9999, boot_id="", start_ticks="1"), open(lock, "w")); ALIVE.clear()
    if runner: ALIVE.add(9999)
    with TestClient(main.app): pass
    good = launched == expect; ok &= good
    print(("PASS " if good else "FAIL ") + label, "| launched", launched, "| log", h.m)
mk("older", "OLDBOOT");            run("job from an older boot resumes", [("r", True)])
mk("current", "CURBOOT");          run("job from the current boot does not", [])
mk("noid", None);                  run("job with no boot id does not (like 20261031-191813)", [])
mk("cancel", "OLDBOOT", True);     run("cancelled job from an older boot does not", [])
mk("older_r", "OLDBOOT");          run("older-boot job, runner alive: skipped", [], runner=True)
# manual endpoints with a runner alive / not alive
ALIVE.add(9999); c = TestClient(main.app); main.RESULTS = SM / "results_boot" / "noid"
r = c.post("/api/results/r/resume"); a = r.status_code == 409 and "smoke1" in r.json()["detail"]
r2 = c.post("/api/results/r/tile", json={"E": 3700}); b = r2.status_code == 409 and "smoke1" in r2.json()["detail"]
print(("PASS " if a and b else "FAIL ") + "manual /resume and /tile 409 while runner alive:", r.json(), r2.json()); ok &= a and b
ALIVE.clear(); launched.clear(); r = c.post("/api/results/r/resume"); good = r.status_code == 200 and launched == [("r", True)]
print(("PASS " if good else "FAIL ") + "manual /resume still works for the no-boot-id folder when no runner:", r.status_code); ok &= good
# _launch records the boot id
import subprocess; main.subprocess.Popen = lambda *a, **k: None
del main._launch; import importlib
src = open(os.path.join(P, "webui", "main.py"), encoding="utf8").read()
ns = {}; exec(compile(src[src.index("def _boot_id"):src.index("@app.post(\"/api/results/{rid}/simulate\")")], "x", "exec"), {**main.__dict__, "_boot_id": lambda: "CURBOOT"}, ns)
d = SM / "results_boot" / "launchtest"; d.mkdir(parents=True, exist_ok=True)
ns["_launch"].__globals__.update(JOBS={}); ns["_launch"]("x", d, ["true"])
jj = json.load(open(d / "job.json")); good = jj.get("boot_id") == "CURBOOT"; ok &= good
print(("PASS " if good else "FAIL ") + "_launch writes boot_id to job.json:", jj)
print("ALL OK" if ok else "FAILURES")
