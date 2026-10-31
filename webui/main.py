import json, math, os, re, signal, subprocess, sys, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SCRIPTS = HERE.parent / "scripts"
ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")
RUNNING = ("released", "relaxing")
JOBS = {}                                    # result id -> subprocess.Popen (one at a time)

# Sourced inputs offered to the form. Everything else the solver needs (mu) is entered by the user.
CONFIG = {
    "yarn_E_MPa": 3700.0,
    "yarn_E_source": "Kharkova, Kononova, Krasnikovs, Eiduks, Machanovskis, Dzelzitis (2011), 'Elastic properties of cotton fabric based "
                     "polymer composites', Engineering for Rural Development, Jelgava, 26-27 May 2011: experimentally obtained cotton yarn "
                     "elastic modulus 3.7 GPa (yarn tension tests; yarn modelled as a homogeneous elastic rod).",
    "kc_ratio": 30.0,                        # contact stiffness as a multiple of EA/R. Numerical choice, tested on one fabric only.
    "perturb": 0.02, "outer": 40, "maxiter": 300,
}

def _alive(rid):
    p = JOBS.get(rid)
    return p is not None and p.poll() is None

def _any_running():
    return any(p.poll() is None for p in JOBS.values())

def _read_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _sim_summary(d):
    s = _read_json(d / "state.json")
    if not s:
        return None
    st = s.get("state")
    if st in RUNNING and not _alive(d.name):
        st = "interrupted"                   # the job process is gone but never reached a final state
    t = d / "relaxed_tube.obj"
    return {"state": st, "has_relaxed": t.exists() and (d / "relaxed_axis.csv").exists(),
            "relaxed_at": int(t.stat().st_mtime) if t.exists() else 0}

@app.get("/api/config")
def config():
    return CONFIG

@app.get("/api/results")
def list_results():
    out = []
    for d in sorted((p for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith(".")),
                    key=lambda p: p.name, reverse=True):
        m = _read_json(d / "meta.json")
        if m is None:
            continue                         # no meta.json yet = still being written
        st = m.get("structure", {})
        out.append({"id": d.name, "created": m.get("created"), "name": m.get("name") or m.get("label", ""),
                    "label": m.get("label", ""), "units": st.get("units", ""), "R": m.get("R"),
                    "info": m.get("info"), "settings": m.get("settings", {}), "report": m.get("report", ""),
                    "sim": _sim_summary(d)})
    return out

def _file(rid, pattern, media, variant="baseline"):
    if not ID_RE.match(rid) or variant not in ("baseline", "relaxed"):
        raise HTTPException(404)
    hits = sorted(h for h in (RESULTS / rid).glob(pattern) if h.name.startswith("relaxed_") == (variant == "relaxed"))
    if not hits:
        raise HTTPException(404)
    return FileResponse(hits[0], media_type=media)

@app.get("/api/results/{rid}/tube")
def tube(rid: str, v: str = "baseline"):
    return _file(rid, "*_tube.obj", "text/plain", v)

@app.get("/api/results/{rid}/axis")
def axis(rid: str, v: str = "baseline"):
    return _file(rid, "*_axis.csv", "text/plain", v)

# ---------------------------------------------------------------- simulation
class SimBody(BaseModel):
    E: float = Field(gt=0, description="yarn modulus: MPa (N/mm^2) for mm results, any positive number for R-unit results")
    mu: float = Field(ge=0, le=2, description="yarn-yarn friction coefficient")
    kc_ratio: float = Field(gt=0, le=10000)
    perturb: float = Field(0.02, ge=0, le=0.2)
    outer: int = Field(40, ge=1, le=500)
    maxiter: int = Field(300, ge=10, le=3000)

def _folder(rid):
    if not ID_RE.match(rid) or not (RESULTS / rid / "meta.json").exists():
        raise HTTPException(404, "unknown result")
    return RESULTS / rid

@app.post("/api/results/{rid}/simulate")
def simulate(rid: str, b: SimBody):
    d = _folder(rid)
    if _any_running():
        raise HTTPException(409, "a simulation is already running")
    meta = _read_json(d / "meta.json")
    R = meta.get("R")
    if not R:
        raise HTTPException(422, "this result has no yarn radius R in meta.json (built by an older pipeline)")
    kc = b.kc_ratio * b.E * math.pi * R          # kc = ratio * EA / R, with EA = E pi R^2
    for f in ("relaxed_axis.csv", "relaxed_tube.obj", "state.json"):
        (d / f).unlink(missing_ok=True)          # never leave a previous run's files next to a new one
    cmd = [sys.executable, str(SCRIPTS / "relax_job.py"), "--folder", str(d), "--E", repr(b.E), "--mu", repr(b.mu),
           "--k-contact", repr(kc), "--perturb", repr(b.perturb), "--outer", str(b.outer), "--maxiter", str(b.maxiter)]
    log = open(d / "job.log", "w")
    JOBS[rid] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SCRIPTS), start_new_session=True)
    return {"started": rid, "k_contact": kc}

@app.get("/api/results/{rid}/state")
def state(rid: str):
    d = _folder(rid)
    s = _read_json(d / "state.json") or {}
    sim = _sim_summary(d)
    log = ""
    try:
        log = "\n".join((d / "job.log").read_text().splitlines()[-12:])
    except Exception:
        pass
    el = s.get("energy_log") or []
    outer = (s.get("params") or {}).get("outer_max")
    return {"state": sim["state"] if sim else "none", "running": _alive(rid), "busy": _any_running(),
            "iter": (el[-1]["iter"] + 1) if el else 0, "outer": outer, "last": el[-1] if el else None,
            "summary": s.get("summary"), "has_relaxed": bool(sim and sim["has_relaxed"]),
            "relaxed_at": sim["relaxed_at"] if sim else 0, "log": log}

@app.post("/api/results/{rid}/cancel")
def cancel(rid: str):
    d = _folder(rid)
    p = JOBS.get(rid)
    if p is None or p.poll() is not None:
        raise HTTPException(409, "no running simulation for this result")
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=10)
    except Exception:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    s = _read_json(d / "state.json") or {}
    s["state"] = "cancelled"
    s.setdefault("history", []).append({"state": "cancelled", "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": "cancelled from the UI"})
    tmp = d / "state.json.tmp"; tmp.write_text(json.dumps(s, indent=1)); os.replace(tmp, d / "state.json")
    return {"cancelled": rid}

@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text()      # re-read each time: editing index.html needs no restart
