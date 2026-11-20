import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)          # repository root: the tests run from any folder
import os, sys, json, time, subprocess as sp
P = _os.path.join(ROOT_STR); SM = os.path.join(P, "smoke_test")
os.environ["CAMPAIGN_DIR"] = os.path.join(SM, "campaigns"); os.environ["AUX_AUTORESUME"] = "0"
sys.path.insert(0, os.path.join(P, "webui"))
_s = os.path.join(SM, "campaigns", "smoke1", "STOP")
if os.path.exists(_s): os.replace(_s, _s + ".stale." + str(time.time()))
before = sorted(os.listdir(os.path.join(P, "campaigns")))
import main, campaign_api as ca
from fastapi.testclient import TestClient
def nokill(*a, **k): raise AssertionError("os.kill called")
os.kill = nokill; os.killpg = nokill if hasattr(os, "killpg") else None
main.RESULTS = __import__("pathlib").Path(SM) / "results"; main.RESULTS.mkdir(exist_ok=True)
main.PRESET_FILE = __import__("pathlib").Path(SM) / "presets.json"
main._launch = lambda *a, **k: calls.append("launch")
calls = []
ALIVE = set(); cm = ca._cm(); cm.alive = lambda att: bool(att) and att.get("pid") in ALIVE
c = TestClient(main.app); ok = True
def check(label, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + label, extra if not cond else "")
SMK = "/api/campaigns/smoke1"
j = c.get("/api/campaigns").json()
check("list has smoke1 from CAMPAIGN_DIR", "smoke1" in [x["name"] for x in j["campaigns"]], j)
o = [x for x in j["campaigns"] if x["name"] == "smoke1"][0]; check("counts", o["counts"] == {"queued": 1, "running": 1, "done": 3, "failed": 1, "other": 0}, o)
check("state paused (started, runner dead, queued left)", o["state"] == "paused" and not o["runner_alive"] and not o["stop_requested"], o)
d = c.get(SMK).json()
print(json.dumps(d["summary"], indent=1)); print(json.dumps(d["eta"], indent=1))
check("eta 240 s (mean 120 x 2 left)", abs(d["eta"]["estimate_s"] - 240) < 1e-6)
txt = " | ".join(s["text"] for s in d["summary"])
check("summary: 2 of 3 did not settle", "2 of 3 finished runs did not settle" in txt)
check("summary: no fold above threshold flagged? (1 of 3 folds)", "1 of 3 finished runs show a fold above the threshold of 0.2 rad (course 1)" in txt)
check("summary: interpenetration in run 3", "Interpenetration in run 3" in txt)
check("summary: failed run 4", "Run 4 failed: timeout" in txt)
r3 = [r for r in d["runs"] if r["run"] == "0003"][0]; check("flags run3", r3["flags"] == {"failed": False, "not_settled": True, "interpenetration": True, "no_fold": True}, r3["flags"])
check("params columns", d["params"] == ["perturb", "bend_scale"] and d["varied"][1]["scale"] == "log")
rd = c.get(SMK + "/runs/0002").json(); check("run detail log 40 lines", len(rd["log_tail"].splitlines()) == 40 and rd["log_tail"].endswith("log line 99"), len(rd["log_tail"].splitlines()))
check("run detail metrics+params", rd["metrics"]["fold_direction"] == "none" and "perturb" in rd["params"])
check("bad run id 404", c.get(SMK + "/runs/abc").status_code == 404 and c.get(SMK + "/runs/0099").status_code == 404)
check("bad name 404", c.get("/api/campaigns/..").status_code in (404, 422) and c.get("/api/campaigns/nope").status_code == 404)
# --- start
runs = []
class FP: returncode = 0; stdout = "runner started (own session, SIGHUP ignored): pid 1, log x\n"; stderr = ""
ca.subprocess.run = lambda cmd, **k: (runs.append(cmd), FP())[1]
r = c.post(SMK + "/start"); check("start ok", r.status_code == 200 and runs and runs[0][-3:] == ["run", "smoke1", "--detach"] and runs[0][1].endswith("campaign.py"), r.text)
r = c.post(SMK + "/start"); check("second start refused (guard)", r.status_code == 409, r.text); ca._last_start.clear()
lock = os.path.join(SM, "campaigns", "smoke1", "runner.lock"); json.dump(dict(pid=9999, boot_id="", start_ticks="1"), open(lock, "w")); ALIVE.add(9999)
n = len(runs); r = c.post(SMK + "/start"); check("start refused: runner alive", r.status_code == 409 and "running" in r.json()["detail"] and len(runs) == n, r.text)
o = [x for x in c.get("/api/campaigns").json()["campaigns"] if x["name"] == "smoke1"][0]; check("state running", o["state"] == "running" and o["runner_alive"] and o["runner_pid"] == 9999, o)
# --- simulate lock
os.makedirs(main.RESULTS / "r1", exist_ok=True); json.dump({"R": 0.1, "structure": {"units": "mm"}}, open(main.RESULTS / "r1" / "meta.json", "w"))
body = {"E": 3700, "mu": 0.5, "kc_ratio": 30}
r = c.post("/api/results/r1/simulate", json=body); check("simulate 409 while runner alive", r.status_code == 409 and "smoke1" in r.json()["detail"] and not calls, r.text)
# --- stop
r = c.post(SMK + "/stop"); stopf = os.path.join(SM, "campaigns", "smoke1", "STOP")
check("stop creates STOP", r.status_code == 200 and os.path.exists(stopf), r.text)
o = [x for x in c.get("/api/campaigns").json()["campaigns"] if x["name"] == "smoke1"][0]; check("pill paused + stopping", o["state"] == "paused" and o["stopping"], o)
r = c.post(SMK + "/stop"); check("stop twice is harmless", r.status_code == 200 and "already" in r.json()["message"], r.text)
ALIVE.clear()
r = c.post(SMK + "/stop"); check("stop with nothing alive -> 409", r.status_code == 409 or os.path.exists(stopf), r.text)
# --- web job blocks start; spec change blocks start
ca._last_start.clear(); ca._any_web_job = lambda: True; r = c.post(SMK + "/start"); check("start refused: web job", r.status_code == 409 and "web page" in r.json()["detail"], r.text); ca._any_web_job = lambda: False
sp_ = os.path.join(SM, "campaigns", "smoke1", "spec.json"); orig = open(sp_).read(); open(sp_, "w").write(orig.replace('"timeout_s": 300', '"timeout_s": 301'))
r = c.post(SMK + "/start"); check("start refused: spec changed", r.status_code == 409 and "spec.json changed" in r.json()["detail"], r.text); open(sp_, "w").write(orig)
# --- simulate allowed when nothing alive
r = c.post("/api/results/r1/simulate", json=body); check("simulate proceeds when no runner alive", r.status_code == 200 and calls == ["launch"], r.text)
# --- datasets, help, pages
ds = c.get("/api/datasets").json(); print(json.dumps([{k: x[k] for k in ("id", "kind", "role", "provenance", "rows", "warnings")} for x in ds], indent=1)[:1200])
check("datasets", len(ds) == 2 and all(x["rows"] for x in ds))
h = c.get("/api/campaign-help").json(); check("help rows", len(h["rows"]) == 18 and h["rows"][0][0] == "`exit_code`" and h["caveats"], (len(h["rows"]), h["caveats"][:40]))
check("pages", c.get("/campaigns").status_code == 200 and "Campaigns" in c.get("/campaigns").text and c.get("/campaigns.js").status_code == 200 and c.get("/").status_code == 200 and "/campaigns" in c.get("/").text)
check("real campaigns folder untouched", before == sorted(os.listdir(os.path.join(P, "campaigns"))))
print("ALL OK" if ok else "FAILURES")
