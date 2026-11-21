"""Campaigns control page: read-only views of campaigns/ and datasets, plus two guarded actions (start = the same detached command as
`campaign.py run <name> --detach`, pause = create the STOP file). Nothing is ever killed from here. The YARN campaigns are never deleted from here; the only delete is the one of the
"Pure shell only runs" section (below, /api/shell/...): it removes the heavy files shell.npz and shell.bin of chosen shell runs, inside campaigns/shell_runs only, with a path guard, a typed
confirmation and a log written first (scripts/shell_campaign.py delete_heavy).

Wired into webui/main.py with   campaign_api.setup(app, _any_running)   and   campaign_api.runner_busy()   (used by the Simulate endpoint).
Reuses scripts/campaign.py (alive(), the lock-file format, METRIC_COLS) and scripts/check_dataset.py by loading them from SCRIPTS.
Paths: CAMPAIGN_DIR / CAMPAIGN_SCRIPTS (same variables as campaign.py), AUX_DATA_DIR (default <project>/data).
"""
import csv, hashlib, importlib.util, json, math, os, re, subprocess, sys, threading, time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

HERE = Path(__file__).parent
ROOT = HERE.parent
CAMPAIGNS = Path(os.environ.get("CAMPAIGN_DIR") or ROOT / "campaigns")
SCRIPTS = Path(os.environ.get("CAMPAIGN_SCRIPTS") or ROOT / "scripts")
DATA = Path(os.environ.get("AUX_DATA_DIR") or ROOT / "data")
NAME_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")
RUN_RE = re.compile(r"^[0-9]{1,6}$")
MIN_DIST_FLAG = 0.97          # "below about 0.97 the yarns pass into each other": campaigns/README.md (results.csv columns)
TEXT_COLS = {"relax_state", "settled_by", "fold_direction", "reason"}
PROJECT_PY = Path(os.environ.get("CAMPAIGN_PYTHON") or ROOT / "venv" / "bin" / "python")      # the interpreter of the README commands (venv/bin/python); the web service itself runs in webui/venv
START_GUARD_S = 20            # a second Start for the same campaign within this time is refused (the first runner may not have written its lock yet)

router = APIRouter()
_any_web_job = lambda: False  # replaced by setup(): True while the web UI has a simulation or tile job of its own running
_last_start = {}
_start_lock = threading.Lock()
_mods = {}


def setup(app, any_running):
    global _any_web_job
    _any_web_job = any_running
    app.include_router(router)


def _load(name, path):
    if name not in _mods:
        spec = importlib.util.spec_from_file_location(name, str(path))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _mods[name] = m
    return _mods[name]

def _cm(): return _load("campaign_cli", SCRIPTS / "campaign.py")
def _tight(): return _load("shell_tightness", SCRIPTS / "shell_tightness.py")          # display-only tightness values, computed on read from the recorded CPI and WPI


# ------------------------------------------------------------------ small helpers
def _rjson(p, default=None):
    try:
        return json.loads(Path(p).read_text(), parse_constant=lambda c: None)
    except Exception:
        return default

def _num(x):
    if isinstance(x, bool) or x is None or x == "": return None
    try: v = float(x)
    except (TypeError, ValueError): return None
    return v if math.isfinite(v) else None

def _cdir(name):
    if not NAME_RE.match(name) or not (CAMPAIGNS / name).is_dir():
        raise HTTPException(404, "no such campaign")
    return CAMPAIGNS / name

def _manifest(d):
    m = _rjson(d / "manifest.json")
    if not m or not isinstance(m.get("runs"), list):
        raise HTTPException(409, "this campaign has no manifest.json yet: it must be planned first (campaign.py plan %s)" % d.name)
    return m

def _tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 65536)); data = f.read()
        return "\n".join(data.decode("utf8", "replace").splitlines()[-n:])
    except OSError:
        return ""

def _rid(r): return str(int(r)) if str(r).isdigit() else str(r)


# ------------------------------------------------------------------ liveness (pid, boot id and start time as recorded by campaign.py)
def _runner(d):
    """(alive, pid) of the runner of campaign folder d, from runner.lock. The lock file stays after the process is gone; alive() checks pid, boot id and start ticks."""
    lock = _rjson(d / "runner.lock")
    if isinstance(lock, dict) and lock.get("pid"):
        try: return bool(_cm().alive(lock)), lock["pid"]
        except Exception: return False, lock["pid"]
    return False, None

def _job_alive(d, runs):
    """True if a run marked running has a live job (a runner that died can leave one behind; it is adopted when a runner starts again)."""
    for r in runs:
        st = _rjson(d / "runs" / r["run"] / "status.json") or {}
        if st.get("status") == "running" and st.get("attempts"):
            try:
                if _cm().alive(st["attempts"][-1]): return True
            except Exception: pass
    return False

def runner_busy():
    """'' if no campaign runner, run job or shell run is alive, otherwise a plain sentence saying which."""
    return _yarn_busy() or _shell_busy()

def _shell_busy():
    try: return _sc().busy_message()
    except Exception: return ""

def _yarn_busy():
    try:
        for d in sorted(p for p in CAMPAIGNS.iterdir() if p.is_dir() and (p / "manifest.json").exists()):
            alive, pid = _runner(d)
            if alive: return "campaign '%s' is running (runner pid %s). Pause it on the Campaigns page and wait for it to stop before starting a simulation here." % (d.name, pid)
            m = _rjson(d / "manifest.json") or {}
            if _job_alive(d, m.get("runs") or []): return "a run of campaign '%s' is still running. Wait for it to finish before starting a simulation here." % d.name
    except OSError:
        pass
    return ""


# ------------------------------------------------------------------ overview
def _counts(d, runs):
    c = {"queued": 0, "running": 0, "done": 0, "failed": 0, "other": 0}
    for r in runs:
        s = (_rjson(d / "runs" / r["run"] / "status.json") or {}).get("status", "")
        c[s if s in c else "other"] += 1
    return c

def _pill(alive, stop, c, n):
    """running: runner alive, no STOP. paused: STOP requested (runner may still be finishing its run), or no runner but some runs already started and some still queued.
    idle: no runner and nothing started yet, or nothing left to run."""
    started = c["done"] + c["failed"] + c["running"] > 0
    if alive: return "paused" if stop else "running"
    if stop: return "paused"
    if c["queued"] + c["running"] > 0 and started: return "paused"
    return "idle"

def overview(d):
    man = _rjson(d / "manifest.json")
    stop = (d / "STOP").exists()
    if not man or not isinstance(man.get("runs"), list):
        return {"name": d.name, "planned": False, "n_runs": 0, "counts": {}, "runner_alive": False, "runner_pid": None, "job_alive": False, "stop_requested": stop,
                "state": "idle", "stopping": False, "complete": False, "can_start": False,
                "message": "Not planned yet: run  campaign.py plan %s  first (it fixes the list of runs)." % d.name}
    runs = man["runs"]; c = _counts(d, runs); alive, pid = _runner(d); ja = _job_alive(d, runs)
    state = _pill(alive or ja, stop, c, len(runs))
    left = c["queued"] + c["running"]
    return {"name": d.name, "planned": True, "n_runs": len(runs), "counts": c, "runner_alive": alive, "runner_pid": pid, "job_alive": ja, "stop_requested": stop,
            "state": state, "stopping": bool((alive or ja) and stop), "complete": left == 0, "can_start": left > 0, "message": ""}

@router.get("/api/campaigns")
def list_campaigns():
    out = []
    if CAMPAIGNS.is_dir():
        for d in sorted(p for p in CAMPAIGNS.iterdir() if p.is_dir() and not p.name.startswith(".") and ((p / "spec.json").exists() or (p / "manifest.json").exists())):
            out.append(overview(d))
    return {"campaigns": out, "busy": runner_busy(), "web_job_running": bool(_any_web_job())}


# ------------------------------------------------------------------ runs table, flags, estimate, summary
def _flags(row, th):
    """Flags of one run (True / False; None when the needed metric is missing). Thresholds: min distance 0.97 (campaigns/README.md), fold turn angle from spec.json."""
    done = row["status"] == "done"
    md = row.get("min_dist_over_2R_end")
    f = {"failed": row["status"] == "failed",
         "not_settled": (row.get("exit_code") not in (None, 0, 0.0)) if done else False,
         "interpenetration": (md < MIN_DIST_FLAG) if md is not None else None}
    fd, tx, ty = row.get("fold_direction"), row.get("fold_turn_x_rad"), row.get("fold_turn_y_rad")
    if not done: f["no_fold"] = None
    elif fd in ("none", "course", "wale", "both"): f["no_fold"] = fd == "none"
    elif tx is not None or ty is not None: f["no_fold"] = max(v for v in (tx, ty) if v is not None) < th
    else: f["no_fold"] = None
    return f

def _read_results(d, man, th, spec=None):
    cm = _cm(); mcols = list(cm.METRIC_COLS)
    csvrows = {}
    try:
        with open(d / "results.csv", newline="", encoding="utf8") as fh:
            for r in csv.DictReader(fh): csvrows[r.get("run", "")] = r
    except OSError:
        pass
    pcols = []
    for r in man["runs"]:
        for k in r.get("params", {}):
            if k not in pcols: pcols.append(k)
    rows = []
    for r in man["runs"]:
        rid = r["run"]; st = _rjson(d / "runs" / rid / "status.json") or {}
        met = _rjson(d / "runs" / rid / "metrics.json") or {}; cr = csvrows.get(rid, {})
        row = {"run": rid, "status": st.get("status") or cr.get("status") or "queued", "attempts": len(st.get("attempts") or []), "interrupted": st.get("interrupted_count", 0)}
        for k in pcols:
            v = r.get("params", {}).get(k); row[k] = _num(v) if _num(v) is not None else v
        for k in mcols:
            raw = cr.get(k, "")
            if raw in ("", None): raw = met.get(k, "")
            row[k] = (raw if raw != "" else None) if k in TEXT_COLS else _num(raw)
        if not row.get("reason") and st.get("reason"): row["reason"] = st["reason"]          # status.json is the newer record
        try:                                                                                # nominal S and TF_rel from this run's CPI, WPI and Ne (build settings: fixed values + run parameters); display only
            b, _ = cm.merged(spec, r.get("params", {})) if (spec or {}).get("fabric") and (spec or {}).get("relax") is not None else (dict(r.get("params", {})), None)
            tr = _tight().record(b, b.get("ne") or None)
        except Exception: tr = {"stitch_density": None, "tf_rel": None}
        row["stitch_density"], row["tf_rel"] = tr["stitch_density"], tr["tf_rel"]
        row["flags"] = _flags(row, th)
        rows.append(row)
    return pcols, mcols, rows

def _eta(rows):
    wall = [r["wall_s"] for r in rows if r["status"] == "done" and r.get("wall_s") and r["wall_s"] > 0]
    left = sum(1 for r in rows if r["status"] in ("queued", "running"))
    if not wall or not left:
        return {"estimate_s": None, "runs_left": left, "n_finished_used": len(wall), "mean_wall_s": None,
                "note": "No estimate: " + ("no finished run with a recorded wall time yet." if not wall else "no runs left.")}
    mean = sum(wall) / len(wall)
    return {"estimate_s": mean * left, "runs_left": left, "n_finished_used": len(wall), "mean_wall_s": mean,
            "note": "ESTIMATE: mean wall time of the %d finished run(s) (%.0f s) x %d run(s) not yet finished. The runs differ in their parameters, so later runs may take more or less; "
                    "a run in progress is counted in full; time lost to interruptions and waiting for other jobs is not included." % (len(wall), mean, left)}

def _runs_text(ids): return ("run " if len(ids) == 1 else "runs ") + ", ".join(_rid(i) for i in ids)

def summarize(rows, th, c):
    """Plain-language findings from the numbers; fixed rules, no language model. Each item: {level: info|warn, text}."""
    out = []; n = len(rows)
    done = [r for r in rows if r["status"] == "done"]; failed = [r for r in rows if r["status"] == "failed"]
    parts = ["%d of %d runs finished" % (len(done), n)]
    if failed: parts.append("%d failed" % len(failed))
    if c.get("running"): parts.append("%d running" % c["running"])
    if c.get("queued"): parts.append("%d waiting" % c["queued"])
    out.append({"level": "info", "text": ", ".join(parts) + "."})
    for r in failed[:5]:
        out.append({"level": "warn", "text": "Run %s failed: %s" % (_rid(r["run"]), (r.get("reason") or "no reason recorded")[:140])})
    if len(failed) > 5: out.append({"level": "warn", "text": "%d more failed runs not listed here." % (len(failed) - 5)})
    if done:
        ns = [r for r in done if r["flags"]["not_settled"]]
        if ns: out.append({"level": "warn", "text": "%d of %d finished runs did not settle (the iteration limit was reached before the energy stopped changing): %s." % (len(ns), len(done), _runs_text([r["run"] for r in ns])[0:120])})
        else: out.append({"level": "info", "text": "All %d finished runs settled." % len(done)})
        known = [r for r in done if r["flags"]["no_fold"] is not None]
        if len(known) < len(done): out.append({"level": "warn", "text": "No fold metric for %d finished run(s) (the table cell is empty because it could not be computed)." % (len(done) - len(known))})
        if known:
            folded = [r for r in known if not r["flags"]["no_fold"]]
            if not folded: out.append({"level": "warn", "text": "No run shows a fold above the threshold (turning angle across the sheet of at least %g rad)." % th})
            else:
                dirs = {}
                for r in folded: dirs[r["fold_direction"] or "?"] = dirs.get(r["fold_direction"] or "?", 0) + 1
                out.append({"level": "info", "text": "%d of %d finished runs show a fold above the threshold of %g rad (%s)." % (len(folded), len(known), th, ", ".join("%s %d" % kv for kv in sorted(dirs.items())))})
    ip = [r for r in rows if r["flags"]["interpenetration"]]
    for r in ip[:6]:
        out.append({"level": "warn", "text": "Interpenetration in run %s: closest approach of two yarns is %.3g of the contact distance (below %g)." % (_rid(r["run"]), r["min_dist_over_2R_end"], MIN_DIST_FLAG)})
    if len(ip) > 6: out.append({"level": "warn", "text": "Interpenetration also in %d more runs." % (len(ip) - 6)})
    intr = [r for r in rows if r.get("interrupted")]
    if intr: out.append({"level": "info", "text": "%s survived an interruption (power cut or crash) and was continued." % _runs_text([r["run"] for r in intr]).capitalize()})
    return out

@router.get("/api/campaigns/{name}")
def campaign_detail(name: str):
    d = _cdir(name); man = _manifest(d); spec = _rjson(d / "spec.json") or {}
    th = _num((spec.get("thresholds") or {}).get("fold_min_turn_rad"))
    th = th if th is not None else 0.0
    pcols, mcols, rows = _read_results(d, man, th, spec); ov = overview(d)
    return {"overview": ov, "params": pcols, "metrics": mcols, "runs": rows, "eta": _eta(rows), "summary": summarize(rows, th, ov["counts"]),
            "thresholds": {"fold_min_turn_rad": th, "min_dist_over_2R": MIN_DIST_FLAG}, "tightness": _tight().campaign_info([r.get("stitch_density") for r in rows]),
            "varied": [{"name": v.get("name"), "scale": v.get("scale"), "min": v.get("min"), "max": v.get("max")} for v in spec.get("vary") or []]}

@router.get("/api/campaigns/{name}/runs/{rid}")
def run_detail(name: str, rid: str):
    d = _cdir(name); man = _manifest(d)
    run = next((r for r in man["runs"] if r["run"] == rid), None) if RUN_RE.match(rid) else None
    if run is None: raise HTTPException(404, "no such run")
    rd = d / "runs" / rid; st = _rjson(rd / "status.json") or {}; met = _rjson(rd / "metrics.json") or {}
    atts = st.get("attempts") or []
    spec = _rjson(d / "spec.json") or {}
    th = _num((spec.get("thresholds") or {}).get("fold_min_turn_rad")) or 0.0
    _, _, rows = _read_results(d, man, th, spec); row = next((x for x in rows if x["run"] == rid), {})
    return {"run": rid, "campaign": name, "status": st.get("status", "queued"), "stage": st.get("stage", ""), "reason": st.get("reason", ""), "exit_code": st.get("exit_code"),
            "wall_s": st.get("wall_s"), "interrupted": st.get("interrupted_count", 0), "attempts": len(atts), "last_command": " ".join(atts[-1].get("cmd") or []) if atts and isinstance(atts[-1], dict) else "",
            "params": run.get("params", {}), "metrics": {k: (None if v == "" else v) for k, v in met.items()}, "flags": row.get("flags", {}),
            "nominal": {"stitch_density": row.get("stitch_density"), "tf_rel": row.get("tf_rel")},
            "history": (st.get("history") or [])[-8:], "log_tail": _tail(rd / "relax.log", 40)}


# ------------------------------------------------------------------ actions
@router.post("/api/campaigns/{name}/start")
def start(name: str):
    d = _cdir(name); man = _manifest(d)
    spec_p = d / "spec.json"
    if not spec_p.exists(): raise HTTPException(409, "spec.json is missing")
    if man.get("spec_sha1") != hashlib.sha1(spec_p.read_bytes()).hexdigest():
        raise HTTPException(409, "spec.json changed after the plan was made, so the runner would refuse to start. Nothing was started; this needs the command line (--ignore-spec-change).")
    why = runner_busy()
    if why: raise HTTPException(409, "Not started: " + why)
    if _any_web_job(): raise HTTPException(409, "Not started: a simulation started from the web page is running. Wait for it to finish.")
    ov = overview(d)
    if not ov["can_start"]: raise HTTPException(409, "Not started: no queued runs are left in this campaign.")
    with _start_lock:
        if time.time() - _last_start.get(name, 0) < START_GUARD_S: raise HTTPException(409, "Not started: a start for this campaign was requested a moment ago. Reload in a few seconds.")
        _last_start[name] = time.time()
    py = str(PROJECT_PY) if PROJECT_PY.exists() else sys.executable                       # campaign.py starts build_knit.py and relax_job.py with ITS OWN sys.executable, so use the project venv, not the web service's
    cmd = [py, str(SCRIPTS / "campaign.py"), "run", name, "--detach"]          # the same command as the README step 3
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(SCRIPTS))
    except Exception as e:
        raise HTTPException(500, "could not run the start command: %s" % e)
    if p.returncode != 0:
        raise HTTPException(500, "the start command failed (exit %s): %s" % (p.returncode, (p.stderr or p.stdout).strip()[-300:]))
    return {"started": name, "message": p.stdout.strip(), "command": " ".join(cmd)}

@router.post("/api/campaigns/{name}/stop")
def stop(name: str):
    d = _cdir(name); _manifest(d)
    alive, _ = _runner(d)
    if not (alive or _job_alive(d, _rjson(d / "manifest.json")["runs"])):
        raise HTTPException(409, "Nothing to pause: no runner of this campaign is alive.")
    p = d / "STOP"
    if p.exists(): return {"stop": name, "message": "A pause was already requested; the run in progress is finishing."}
    with open(p, "w") as f: f.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")                  # exactly what campaign.py stop writes; nothing is killed
    return {"stop": name, "message": "Pause requested. The run in progress finishes, then the runner stops. Nothing is killed."}


# ------------------------------------------------------------------ pure shell runs (scripts/shell_campaign.py; design of 2026-11-21)
def _sc(): return _load("shell_campaign", SCRIPTS / "shell_campaign.py")

def _shell(name):
    sc = _sc()
    try: c = sc.Camp(name)
    except ValueError: raise HTTPException(404, "unknown shell campaign")
    if not c.exists(): raise HTTPException(404, "unknown shell campaign")
    return sc, c

@router.get("/api/shell")
def shell_list():
    sc = _sc(); root = Path(sc.SHELL_ROOT); out = []
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir() and NAME_RE.match(p.name) and (p / "spec.json").exists()):
            st = sc.Camp(d.name).status(); out.append({"name": d.name, "control": st["control"], "runner_alive": st["runner_alive"], "state": (st["runner"] or {}).get("state", ""), "counts": st["counts"]})
    return {"campaigns": out}

@router.get("/api/shell/{name}")
def shell_detail(name: str):
    sc, c = _shell(name); st = c.status(); rows = c.rows(); sco = _load("shell_score", SCRIPTS / "shell_score.py")
    for r in rows: r["score"] = sco.score(r)          # similarity out of 100 where a measured reference exists (Pattern 1), else None
    tg = _tight()
    for r in rows: tr = tg.record(r); r["stitch_density"], r["tf_rel"] = tr["stitch_density"], tr["tf_rel"]          # display only, computed on read (Ne fixed at 12); nothing is written to the runs
    st["tightness"] = tg.campaign_info([r["stitch_density"] for r in rows], c.policy())
    st["n_rows"] = len(rows); st["rows"] = rows[-1500:]
    return st

def _shell_guard(name):
    why = _yarn_busy()
    if why: raise HTTPException(409, "Not started: " + why)
    if _any_web_job(): raise HTTPException(409, "Not started: a simulation started from the web page is running. Wait for it to finish.")

@router.post("/api/shell/{name}/start")
def shell_start(name: str):
    """Start (or, if the runner is alive, continue): control = run; starts the detached runner (the FROZEN copy's campaign.py shell-run --detach, own session, SIGHUP ignored, project interpreter) when none is alive."""
    sc, c = _shell(name)
    if c.runner_alive():
        c.set_control("run", why="page: start"); return {"started": name, "message": "The runner is already alive; control set to run."}
    _shell_guard(name)
    script = Path(c.code_dir()) / "campaign.py"          # the runner always starts from the campaign's FROZEN copy of the code, so a later git pull cannot change it
    if not script.is_file(): raise HTTPException(409, "Not started: this shell campaign has no frozen code folder (code/). Run  campaign.py shell-freeze %s  on the server first." % name)
    with _start_lock:
        if time.time() - _last_start.get("shell:" + name, 0) < START_GUARD_S: raise HTTPException(409, "Not started: a start for this campaign was requested a moment ago. Reload in a few seconds.")
        _last_start["shell:" + name] = time.time()
    c.set_control("run", why="page: start")
    py = str(PROJECT_PY) if PROJECT_PY.exists() else sys.executable
    cmd = [py, str(script), "shell-run", name, "--detach"]
    try: p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(SCRIPTS))
    except Exception as e: raise HTTPException(500, "could not run the start command: %s" % e)
    if p.returncode != 0: raise HTTPException(500, "the start command failed (exit %s): %s" % (p.returncode, (p.stderr or p.stdout).strip()[-300:]))
    return {"started": name, "message": p.stdout.strip(), "command": " ".join(cmd)}

@router.post("/api/shell/{name}/pause")
def shell_pause(name: str):
    sc, c = _shell(name); c.set_control("pause", why="page: pause")
    return {"control": "pause", "message": "Pause requested: the run in progress finishes, then the runner waits. Nothing is killed."}

@router.post("/api/shell/{name}/continue")
def shell_continue(name: str):
    sc, c = _shell(name)
    if not c.runner_alive(): raise HTTPException(409, "No runner of this campaign is alive: use Start.")
    c.set_control("run", why="page: continue"); return {"control": "run", "message": "Continuing."}

@router.post("/api/shell/{name}/stop")
def shell_stop(name: str):
    sc, c = _shell(name); c.set_control("stop", why="page: stop after current run")
    return {"control": "stop", "message": "Stop requested: the run in progress finishes, then the runner exits. Nothing is killed."}

class ShellDelete(BaseModel):
    ids: list[str]
    confirm: str = ""

@router.post("/api/shell/{name}/delete")
def shell_delete(name: str, b: ShellDelete, request: Request):
    """Remove shell.npz and shell.bin of the chosen finished runs (settings, metrics and the index line stay). The confirmation text must be 'DELETE <n>' with n the number of runs."""
    sc, c = _shell(name)
    try: prev = sc.delete_heavy(c, b.ids, dry=True)
    except ValueError as e: raise HTTPException(422, str(e))
    if b.confirm != "DELETE %d" % len(prev["ids"]): raise HTTPException(422, "Type exactly DELETE %d to confirm (the number of runs selected)." % len(prev["ids"]))
    try: return sc.delete_heavy(c, b.ids, client=(request.client.host if request.client else ""))
    except ValueError as e: raise HTTPException(422, str(e))

@router.get("/api/shell/{name}/delete-preview")
def shell_delete_preview(name: str, ids: str = ""):
    sc, c = _shell(name)
    try: return sc.delete_heavy(c, [i for i in ids.split(",") if i], dry=True)
    except ValueError as e: raise HTTPException(422, str(e))


# ------------------------------------------------------------------ pure shell runs: one run (detail, 3D mesh, strain colours) and the pages
@router.get("/api/shell/{name}/runs/{rid}")
def shell_run(name: str, rid: str):
    sc, c = _shell(name)
    try: d = c.run_detail(rid)
    except ValueError as e: raise HTTPException(404, str(e))
    tg = _tight(); d["tightness"] = dict(tg.record(d.get("settings") or {}), label_stitch_density=tg.LABEL_S, label_tf_rel=tg.LABEL_TF, note=tg.NOTE)          # display only, computed on read
    d["tightness"]["constant_note"] = tg.campaign_info([tg.record(q.get("settings") or {})["stitch_density"] for q in c.queue_items()], c.policy())["constant_note"]
    return d

@router.get("/api/shell/{name}/runs/{rid}/mesh")
def shell_mesh(name: str, rid: str):
    """The folded sheet as the KMSH v1 mesh of the viewer (shell.bin)."""
    sc, c = _shell(name)
    try: f = Path(c.run_path(rid)) / "shell.bin"
    except ValueError as e: raise HTTPException(404, str(e))
    if not f.is_file(): raise HTTPException(404, "the 3D data of this run was deleted")
    return FileResponse(f, media_type="application/octet-stream", headers={"Content-Encoding": "identity"})

@router.get("/api/shell/{name}/runs/{rid}/colors")
def shell_colors(name: str, rid: str, field: str = "area_strain", clip: float = 98.0):
    """Uint8 RGB per vertex (same order as the mesh) plus the legend in X-Legend, as /api/results/<id>/colors."""
    sc, c = _shell(name)
    if not (50.0 <= clip <= 100.0): raise HTTPException(422, "clip must be 50 to 100")
    try: f = Path(c.run_path(rid)) / "shell.npz"
    except ValueError as e: raise HTTPException(404, str(e))
    if not f.is_file(): raise HTTPException(404, "the 3D data of this run was deleted")
    ss = _load("shell_strain", SCRIPTS / "shell_strain.py")
    try: X, W, C, sub = ss.load(f); rgb, legend = ss.colours(field, ss.vertex_field(X, W, C, sub, field), clip)
    except ValueError as e: raise HTTPException(422, str(e))
    return Response(rgb.tobytes(), media_type="application/octet-stream", headers={"Content-Encoding": "identity", "X-Count": str(len(rgb)), "X-Legend": json.dumps(legend)})

@router.get("/api/shell/{name}/rules")
def shell_rules(name: str):
    """What the offline rule job (scripts/shell_rules.py) last wrote; nothing is computed here."""
    sc, c = _shell(name); r = _rjson(Path(c.d) / "analysis" / "rules.json")
    if r is None: return {"status": "not run yet", "min_runs": 100, "n_finished": sum(1 for x in c.index_last().values() if x.get("status") == "done"), "label": "learned from the shell, not confirmed on real fabric"}
    return r

def _page(name, media):
    return Response((HERE / name).read_text(encoding="utf8"), media_type=media, headers={"Cache-Control": "no-cache"})

@router.get("/shell", response_class=HTMLResponse)
def page_shell(): return _page("shell.html", "text/html; charset=utf-8")

@router.get("/shell.js")
def page_shell_js(): return _page("shell.js", "application/javascript")

@router.get("/shellview", response_class=HTMLResponse)
def page_shellview(): return _page("shellview.html", "text/html; charset=utf-8")

@router.get("/shellview.js")
def page_shellview_js(): return _page("shellview.js", "application/javascript")


# ------------------------------------------------------------------ datasets (read only) and the column help
@router.get("/api/datasets")
def datasets():
    cd = _load("check_dataset_cli", SCRIPTS / "check_dataset.py"); out = []
    if DATA.is_dir():
        for d in sorted(p for p in DATA.iterdir() if p.is_dir() and (p / "meta.json").exists()):
            info = cd.check_dataset(str(d)); meta = _rjson(d / "meta.json") or {}
            rows = None
            try:
                with open(d / "data.csv", newline="", encoding="utf8") as fh: rows = max(0, sum(1 for _ in csv.reader(fh)) - 1)
            except OSError: pass
            out.append({"id": d.name, "kind": meta.get("kind"), "role": meta.get("role"), "provenance": meta.get("provenance"), "source": meta.get("source"),
                        "fabric": (meta.get("fabric") or {}).get("name"), "rows": rows, "usable": None if info.get("usable") in (None, "-") else int(info["usable"]),
                        "errors": info["errors"], "warnings": info["warnings"], "notes": meta.get("notes", "")})
    return out

@router.get("/api/campaign-help")
def campaign_help():
    """The 'results.csv columns' section of campaigns/README.md as data (one source of truth for the page)."""
    try: lines = (CAMPAIGNS / "README.md").read_text(encoding="utf8").splitlines()
    except OSError: return {"intro": "", "rows": [], "caveats": ""}
    i = next((k for k, l in enumerate(lines) if l.strip() == "## results.csv columns"), None)
    if i is None: return {"intro": "", "rows": [], "caveats": ""}
    sec = []
    for l in lines[i + 1:]:
        if l.startswith("## "): break
        sec.append(l)
    intro, rows, cav = [], [], []
    for l in sec:
        if l.startswith("|"):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[0] != "Column" and not set(cells[0]) <= set("-"): rows.append([cells[0], " | ".join(cells[1:])])
        elif l.strip(): (cav if rows else intro).append(l.strip())
    return {"intro": " ".join(intro), "rows": rows, "caveats": " ".join(cav)}


# ------------------------------------------------------------------ the page (re-read each time, like index.html: editing needs no restart)
@router.get("/campaigns", response_class=HTMLResponse)
def page():
    return (HERE / "campaigns.html").read_text(encoding="utf8")

@router.get("/campaigns.js")
def page_js():
    return Response((HERE / "campaigns.js").read_text(encoding="utf8"), media_type="application/javascript")

@router.get("/chartkit.js")
def page_chartkit():
    """The scatter-chart renderer shared by the Campaigns page and the pure-shell page (webui/chartkit.js)."""
    return Response((HERE / "chartkit.js").read_text(encoding="utf8"), media_type="application/javascript", headers={"Cache-Control": "no-cache"})

@router.get("/theme.css")
def page_theme():
    """The shared stylesheet of the pages (webui/theme.css): the one place where the colours, sizes and spacing are defined."""
    return Response((HERE / "theme.css").read_text(encoding="utf8"), media_type="text/css", headers={"Cache-Control": "no-cache"})
