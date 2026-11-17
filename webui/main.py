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
    "tension_cN": 10.0,
    "tension_source": "Input yarn tensions of 10, 20 and 30 cN were used for cotton, polyester and polyamide threads on a circular knitting machine with passive "
                      "yarn feeding (Control of the stability circular knitting process with passive yarn feeding, Mechanika, Kaunas University of Technology). "
                      "Experimental settings in that study, not a certified standard; the lowest is offered as the default.",
    "straight": 0.0,                         # straightening factor, 0 = off. No sourced value: a modelling factor.
    "kc_ratio": 30.0,                        # contact stiffness as a multiple of EA/R. Numerical choice, tested on one fabric only.
    "perturb": 0.02, "outer": 40, "maxiter": 300,
    "shell_w": 1.0,                          # hold on the shell fold, x EA/R per vertex. Numerical choice, not tested on the real solver.
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

def _matrix_text(m):
    M = m.get("matrix")
    return ("%d x %d stitches per unit" % (len(M[0]), len(M))) if M and M[0] else ""

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
                    "matrix_text": _matrix_text(m), "n_sims": len(m.get("simulations") or []), "sim": _sim_summary(d)})
    return out

def _file(rid, pattern, media, variant="baseline"):
    if not ID_RE.match(rid) or variant not in ("baseline", "relaxed"):
        raise HTTPException(404)
    hits = sorted(h for h in (RESULTS / rid).glob(pattern) if h.name.startswith("relaxed_") == (variant == "relaxed"))
    if not hits:
        raise HTTPException(404)
    return FileResponse(hits[0], media_type=media)

def _ensure_baseline_tube(d):
    """Results built with build_knit --no-tube have no baseline tube mesh. Build it once from fab_axis.csv (same tube writer as the pipeline, 8 sides) and cache it as fab_tube.obj."""
    if any(h for h in list(d.glob("*_tube.obj")) + list(d.glob("*_tube.bin")) if not h.name.startswith("relaxed_")):
        return
    meta, ax = _read_json(d / "meta.json") or {}, d / "fab_axis.csv"
    if not meta.get("R") or not ax.exists():
        raise HTTPException(404, "no baseline mesh and no fab_axis.csv to build one from")
    import numpy as np
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from build_knit import write_obj
    A = np.loadtxt(ax, delimiter=",", skiprows=1)
    lines = [A[A[:, 0] == c][:, 1:] for c in np.unique(A[:, 0])]
    tmp = d / "fab_tmp.obj"
    write_obj(str(tmp), lines, meta["R"], 8, "baseline tube of %s" % d.name)
    if tmp.with_suffix(".bin").exists():
        os.replace(tmp.with_suffix(".bin"), d / "fab_tube.bin")      # the writer also emits the binary mesh: keep it, no second conversion
    os.replace(tmp, d / "fab_tube.obj")

@app.get("/api/results/{rid}/tube")
def tube(rid: str, v: str = "baseline"):
    if ID_RE.match(rid) and v == "baseline" and (RESULTS / rid).is_dir():
        _ensure_baseline_tube(RESULTS / rid)
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
    curl: float = Field(0.0, ge=0, le=0.5, description="stored curl, turning angle (rad) per vertex")
    curl_aniso: bool = False
    ref_scale: float = Field(0.0, ge=0, le=1, description="stress-free curvature = ref_scale x baseline loop curvature")
    tension_cN: float = Field(0.0, ge=0, le=200)
    creep_tau_h: float = Field(0.0, ge=0)
    hours: float = Field(0.0, ge=0)
    shell: bool = False
    shell_kappa: str = Field('guess', pattern='^(guess|tile)$', description='shell curvature: fixed guess (Niu et al.) or from the periodic yarn tile')
    kappa_scale: float = Field(1.0, ge=-5, le=5, description='calibration factor on the shell curvature (negative flips the fold)')
    tile_ref: float = Field(0.5, ge=0, le=1, description='tile initial rest-curvature scale (unsourced)')
    stretch_scale: float = Field(1.0, gt=0, le=1, description='multiplies EA only; below 1 lets rest length migrate more')
    seed_only: bool = False
    straight: float = Field(0.0, ge=0, le=100, description='straightening factor: extra pull toward a straight yarn, length unchanged (0 = off)')
    shell_frame: bool = True
    bend_scale: float = Field(1.0, gt=0, le=1, description='multiplies the bending stiffness EI only (spun yarn is far softer than a solid rod)')
    length_ratio: float = Field(1.0, ge=0.5, le=1.0, description='rest length ratio eta (<1: yarn knitted tighter than the built geometry)')
    shell_w: float = Field(1.0, ge=0, le=1000, description="hold on the shell fold, x EA/R per vertex (0 = start from the fold only)")

class TileBody(BaseModel):
    E: float = Field(gt=0); kc_ratio: float = Field(30.0, gt=0, le=10000)
    ref_scale: float = Field(0.5, ge=0, le=1, description="initial rest curvature = ref_scale x baseline loop curvature")
    reset: float = Field(0.0, ge=0, le=1, description="fraction of the way the rest shape moves to the current state per reset (Sperl et al. 7.2)")
    rounds: int = Field(3, ge=1, le=12); maxiter: int = Field(1500, ge=50, le=6000)
    perturb: float = Field(0.02, ge=0, le=0.2); delta: float = Field(0.05, gt=0, le=0.5)
    strain: str = Field("0.05,0.1,0.2", pattern=r"^[0-9.]+(,[0-9.]+){0,7}$")
    do: str = Field("rest,curl,poisson", pattern=r"^(rest|curl|poisson)(,(rest|curl|poisson))*$")

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
    if b.creep_tau_h > 0 and b.hours <= 0:
        raise HTTPException(422, "creep needs a simulated duration (hours) above 0")
    if b.tension_cN > 0 and (meta.get("structure") or {}).get("units") != "mm":
        raise HTTPException(422, "feeder tension needs a result built in mm; set it to 0 for R-unit results")
    if b.seed_only and not b.shell:
        raise HTTPException(422, "shell-only needs the shell option ticked")
    if b.shell and not meta.get("matrix"):
        raise HTTPException(422, "the shell option needs the stitch matrix in meta.json (result built by an older pipeline)")
    kc = b.kc_ratio * b.E * math.pi * R          # kc = ratio * EA / R, with EA = E pi R^2
    for f in ("relaxed_axis.csv", "relaxed_tube.obj", "relaxed_tube.bin", "state.json", "job.json", "shell.npz"):
        (d / f).unlink(missing_ok=True)          # never leave a previous run's files next to a new one
    cmd = [sys.executable, str(SCRIPTS / "relax_job.py"), "--folder", str(d), "--E", repr(b.E), "--mu", repr(b.mu),
           "--k-contact", repr(kc), "--perturb", repr(b.perturb), "--outer", str(b.outer), "--maxiter", str(b.maxiter),
           "--curl", repr(b.curl), "--tension", repr(b.tension_cN), "--creep-tau", repr(b.creep_tau_h), "--hours", repr(b.hours)] + (["--curl-aniso"] if b.curl_aniso else []) + ["--ref-scale", repr(b.ref_scale)]
    if b.shell:
        cmd += ["--shell", "--shell-w", repr(b.shell_w), "--shell-kappa", b.shell_kappa, "--kappa-scale", repr(b.kappa_scale), "--tile-ref", repr(b.tile_ref)]
        if b.seed_only: cmd += ["--seed-only"]
        if not b.shell_frame: cmd += ["--no-shell-frame"]
    cmd += ["--stretch-scale", repr(b.stretch_scale), "--bend-scale", repr(b.bend_scale), "--straight", repr(b.straight), "--length-ratio", repr(b.length_ratio)]
    log = open(d / "job.log", "w")
    (d / "job.json").write_text(json.dumps({"started": time.time()}))          # before the job starts, so it can always read it
    JOBS[rid] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SCRIPTS), start_new_session=True)
    return {"started": rid, "k_contact": kc}

def _tile_alive(rid):
    p = JOBS.get(rid + "#tile")
    return p is not None and p.poll() is None

@app.post("/api/results/{rid}/tile")
def tile_start(rid: str, b: TileBody):
    d = _folder(rid)
    if _any_running():
        raise HTTPException(409, "a job is already running")
    meta = _read_json(d / "meta.json") or {}
    if not (meta.get("matrix") and meta.get("W") and meta.get("C") and meta.get("R")):
        raise HTTPException(422, "meta.json has no matrix, W, C and R (result built by an older pipeline)")
    for f in ("tile_result.json", "tile_axis.csv", "tile_state.npz", "tile.log"):
        (d / f).unlink(missing_ok=True)
    cmd = [sys.executable, str(SCRIPTS / "tile_sim.py"), "--folder", str(d), "--E", repr(b.E), "--kc-ratio", repr(b.kc_ratio), "--ref-scale", repr(b.ref_scale),
           "--reset", repr(b.reset), "--rounds", str(b.rounds), "--maxiter", str(b.maxiter), "--perturb", repr(b.perturb), "--delta", repr(b.delta),
           "--strain", b.strain, "--do", b.do]
    log = open(d / "tile.log", "w")
    (d / "tile_job.json").write_text(json.dumps({"started": time.time()}))
    JOBS[rid + "#tile"] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SCRIPTS), start_new_session=True)
    return {"started": rid}

@app.get("/api/results/{rid}/tile")
def tile_state(rid: str):
    d = _folder(rid)
    res = _read_json(d / "tile_result.json")
    jj = _read_json(d / "tile_job.json") or {}
    alive = _tile_alive(rid)
    try:
        log = "\n".join((d / "tile.log").read_text().splitlines()[-14:])
    except Exception:
        log = ""
    status = (res or {}).get("status")
    if status == "running" and not alive:
        status = "interrupted"
    if res is None and not alive and (d / "tile_job.json").exists():
        status = "failed"                    # the job ended before writing anything: the log tail says why
    elapsed = (res or {}).get("seconds") if not alive else ((time.time() - jj["started"]) if jj.get("started") else None)
    return {"running": alive, "busy": _any_running(), "status": status, "elapsed": elapsed, "log": log, "result": res}

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
    jj = _read_json(d / "job.json") or {}
    sims = (_read_json(d / "meta.json") or {}).get("simulations") or []
    seconds = sims[-1].get("seconds") if sims else None
    elapsed = (time.time() - jj["started"]) if (_alive(rid) and jj.get("started")) else seconds
    return {"elapsed": elapsed, "seconds": seconds, "state": sim["state"] if sim else "none", "running": _alive(rid), "busy": _any_running(),
            "iter": (el[-1]["iter"] + 1) if el else 0, "outer": outer, "last": el[-1] if el else None,
            "summary": s.get("summary"), "has_relaxed": bool(sim and sim["has_relaxed"]),
            "relaxed_at": sim["relaxed_at"] if sim else 0, "log": log}

@app.get("/api/results/{rid}/log")
def sim_log(rid: str):
    return (_read_json(_folder(rid) / "meta.json") or {}).get("simulations") or []

def _obj_to_bin(obj, binp):
    """Legacy results only: convert fab_tube.obj to the compact binary mesh once and cache it."""
    import numpy as np
    V, F = [], []
    with open(obj) as f:
        for line in f:
            if line.startswith("v "): V.append(line[2:].split())
            elif line.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in line[2:].split()[:3]])
    V = np.asarray(V, dtype="<f4"); F = np.asarray(F, dtype="<u4"); tmp = str(binp) + ".tmp"
    with open(tmp, "wb") as o:
        o.write(b"KMSH"); np.array([1, len(V), len(F)], dtype="<u4").tofile(o); V.tofile(o); F.tofile(o)
    os.replace(tmp, binp)

@app.get("/api/results/{rid}/mesh")
def mesh(rid: str, v: str = "baseline"):
    if not ID_RE.match(rid) or v not in ("baseline", "relaxed"):
        raise HTTPException(404)
    d = RESULTS / rid; rel = (v == "relaxed")
    if not rel and d.is_dir():
        _ensure_baseline_tube(d)
    cand = sorted(h for h in d.glob("*_tube.bin") if h.name.startswith("relaxed_") == rel)
    if not cand:
        objs = sorted(h for h in d.glob("*_tube.obj") if h.name.startswith("relaxed_") == rel)
        if not objs: raise HTTPException(404)
        _obj_to_bin(objs[0], objs[0].with_suffix(".bin")); cand = [objs[0].with_suffix(".bin")]
    return FileResponse(cand[0], media_type="application/octet-stream", headers={"Content-Encoding": "identity"})   # float data barely compresses: skip gzip

@app.post("/api/results/{rid}/cancel")
def cancel(rid: str):
    d = _folder(rid)
    p = JOBS.get(rid)
    if p is None or p.poll() is not None:
        p = JOBS.get(rid + "#tile")
        if p is None or p.poll() is not None:
            raise HTTPException(409, "no running simulation for this result")
        try:
            os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=10)
        except Exception:
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: pass
        return {"cancelled": rid}
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