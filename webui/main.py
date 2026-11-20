import base64, json, logging, math, os, re, shutil, signal, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SCRIPTS = HERE.parent / "scripts"
ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")
RUNNING = ("loaded", "armed", "released", "relaxing")      # every state a live job can be in; "loaded" and "armed" last only while the tile / shell stages run
STAGE_FILE = "job_stage.json"
CKPT_FILES = ("checkpoint.npz", "checkpoint.json")
PRESET_FILE = HERE / "presets.json"
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
    "ramp": 0, "ramp_iter": 200,             # growth stages for runs without the shell (0 = off), L-BFGS iterations per stage. Numerical choices, not sourced.
    "seed_stitch": 0.0,                      # out-of-plane seed by stitch type, x R (0 = off). A modelling choice.
    "stall_tol": 1e-6,                       # early stop on a stalled outer loop (0 = off). Numerical choice; a run stopped this way is reported as settled with the reason in state.json.
    "kc_ratio": 30.0,                        # contact stiffness as a multiple of EA/R. Numerical choice, tested on one fabric only.
    "perturb": 0.02, "outer": 40, "maxiter": 300,
    "shell_w": 1.0,                          # hold on the shell fold, x EA/R per vertex. Numerical choice, not tested on the real solver.
}

def _alive(rid):
    p = JOBS.get(rid)
    return p is not None and p.poll() is None

def _any_running():
    return any(p.poll() is None for p in JOBS.values())

sys.path.insert(0, str(HERE)); import campaign_api          # Campaigns page and its API (webui/campaign_api.py)
campaign_api.setup(app, _any_running)

def _read_json(path):
    try:
        return json.loads(path.read_text(), parse_constant=lambda c: None)      # Infinity / NaN (e.g. min axis distance of a loop with no neighbours) -> null; JSON responses reject them and /state returned 500
    except Exception:
        return None

def _sim_summary(d):
    s = _read_json(d / "state.json")
    if not s:
        return None
    st = s.get("state")
    if st in RUNNING and not _alive(d.name):
        st = "interrupted"                   # the job process is gone but never reached a final state
    t = next((p for p in (d / "relaxed_tube.bin", d / "relaxed_tube.obj") if p.exists()), None)      # the .bin is the one the viewer reads; the text OBJ is optional
    return {"state": st, "has_relaxed": t is not None and (d / "relaxed_axis.csv").exists(),
            "relaxed_at": int(t.stat().st_mtime) if t is not None else 0}

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
    hits = sorted(h for h in (RESULTS / rid).glob(pattern) if h.name.startswith("relaxed_") == (variant == "relaxed") and ".view." not in h.name)
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
    os.replace(tmp.with_suffix(".bin"), d / "fab_tube.bin")            # the binary mesh is what the viewer reads
    if tmp.exists():
        os.replace(tmp, d / "fab_tube.obj")                            # the text OBJ exists only for small meshes

def _bin_to_obj_lines(binp):
    """Stream a KMSH binary mesh as OBJ text, 100000 vertices at a time (large meshes never sit in memory as text)."""
    import numpy as np
    with open(binp, "rb") as f:
        head = f.read(16); nv, nf = (int(x) for x in np.frombuffer(head[8:16], "<u4"))
        V = np.fromfile(f, "<f4", nv * 3).reshape(-1, 3); F = np.fromfile(f, "<u4", nf * 3).reshape(-1, 3) + 1
    yield "# %s (from the binary mesh)\n" % binp.name
    for a in range(0, nv, 100000):
        yield "".join("v %.5f %.5f %.5f\n" % tuple(r) for r in V[a:a + 100000])
    for a in range(0, nf, 100000):
        yield "".join("f %d %d %d\n" % tuple(r) for r in F[a:a + 100000])

@app.get("/api/results/{rid}/tube")
def tube(rid: str, v: str = "baseline"):
    """OBJ text of the tube. Large meshes have no stored OBJ; it is streamed from the .bin on request (the web viewer itself uses /mesh)."""
    if not ID_RE.match(rid) or v not in ("baseline", "relaxed"):
        raise HTTPException(404)
    d = RESULTS / rid
    if v == "baseline" and d.is_dir():
        _ensure_baseline_tube(d)
    rel = (v == "relaxed")
    objs = sorted(h for h in d.glob("*_tube.obj") if h.name.startswith("relaxed_") == rel)
    if objs:
        return FileResponse(objs[0], media_type="text/plain")
    bins = sorted(h for h in d.glob("*_tube.bin") if h.name.startswith("relaxed_") == rel)
    if not bins:
        raise HTTPException(404)
    return StreamingResponse(_bin_to_obj_lines(bins[0]), media_type="text/plain")

@app.get("/api/results/{rid}/axis")
def axis(rid: str, v: str = "baseline", view: int = 0):
    """Axis CSV. view=1: a thinned copy for drawing (cached as <name>.view.csv): the files are sampled about 0.2 R apart, which a line drawing does not need."""
    if not view:
        return _file(rid, "*_axis.csv", "text/plain", v)
    if not ID_RE.match(rid) or v not in ("baseline", "relaxed"):
        raise HTTPException(404)
    d = RESULTS / rid
    hits = sorted(h for h in d.glob("*_axis.csv") if h.name.startswith("relaxed_") == (v == "relaxed") and not h.name.endswith(".view.csv"))
    if not hits:
        raise HTTPException(404)
    src = hits[0]; out = src.with_name(src.stem + ".view.csv")
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        import numpy as np
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        from build_knit import decimate_curve
        R = (_read_json(d / "meta.json") or {}).get("R") or 1.0
        A = np.loadtxt(src, delimiter=",", skiprows=1)
        rows = [np.c_[np.full(len(c), k), c] for k in np.unique(A[:, 0]) for c in [decimate_curve(A[A[:, 0] == k][:, 1:], R, 1.0, 8.0)]]
        tmp = out.with_suffix(".tmp"); np.savetxt(tmp, np.vstack(rows), delimiter=",", header="course,x,y,z", comments="", fmt=["%d", "%.5f", "%.5f", "%.5f"]); os.replace(tmp, out)
    return FileResponse(out, media_type="text/plain")

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
    ramp: int = Field(0, ge=0, le=40, description="no shell: grow the stress-free state from the built loop shape to the target in this many stages (0 = off)")
    ramp_iter: int = Field(200, ge=10, le=3000, description="L-BFGS iterations per growth stage")
    seed_stitch: float = Field(0.0, ge=-1, le=1, description="no shell: out-of-plane seed, front loops +a R, back loops -a R (0 = off)")
    fold_search: int = Field(0, ge=0, le=200, description="no shell: accepted large-scale bending steps searched by the yarn energy (0 = off)")
    fold_iter: int = Field(200, ge=10, le=5000, description="relaxation iterations after each trial bend")
    fold_delta: float = Field(0.0, ge=0, le=1, description="first bend step in 1/mm (0 = 0.25 / sheet size)")
    macro: bool = Field(False, description="no shell: the optimiser also moves the three large-scale bending modes (same energy, same minimum)")
    warp_kx: float = Field(0.0, ge=-1, le=1, description="no shell: start from the sheet bent along the courses, curvature in 1/mm (0 = flat start)")
    warp_ky: float = Field(0.0, ge=-1, le=1, description="no shell: same along the wales")
    ea: float = Field(0.0, ge=0, le=1e6, description='measured axial stiffness EA in N (0 = solid rod from E)')
    ei: float = Field(0.0, ge=0, le=1e6, description='measured bending stiffness EI in N mm^2 (0 = solid rod from E)')
    stretch_law: str = Field('', max_length=400, pattern=r'^([0-9.eE+-]+:[0-9.eE+-]+(,[0-9.eE+-]+:[0-9.eE+-]+){0,19})?$', description="measured stretch curve 'strain:cN,...'")
    stall_tol: float = Field(0.0, ge=0, le=0.01, description="early stop: relative energy change per outer round below this for 3 rounds (0 = off)")

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

def _law_points(s):
    """'strain:cN,...' -> [(strain, N)]; ValueError unless both columns are positive and strictly increasing (same rule as relax_state.build_law)."""
    p = [(float(a), 0.01*float(c)) for a, c in (t.split(':') for t in s.split(','))]
    if p[0][0] <= 0 or p[0][1] <= 0 or any(q[0] <= o[0] or q[1] <= o[1] for o, q in zip(p, p[1:])): raise ValueError('strain and force must be positive and strictly increasing')
    return p

def _sim_cmd(d, b, R):
    EA_in = b.ea or (_law_points(b.stretch_law)[0][1]/_law_points(b.stretch_law)[0][0] if b.stretch_law else None)
    kc = b.kc_ratio * (EA_in if EA_in else b.E * math.pi * R**2) / R          # kc = ratio * EA / R; EA = E pi R^2 unless EA or a stretch law is entered (then the entered EA)
    cmd = [sys.executable, str(SCRIPTS / "relax_job.py"), "--folder", str(d), "--E", repr(b.E), "--mu", repr(b.mu),
           "--k-contact", repr(kc), "--perturb", repr(b.perturb), "--outer", str(b.outer), "--maxiter", str(b.maxiter),
           "--curl", repr(b.curl), "--tension", repr(b.tension_cN), "--creep-tau", repr(b.creep_tau_h), "--hours", repr(b.hours)] + (["--curl-aniso"] if b.curl_aniso else []) + ["--ref-scale", repr(b.ref_scale)]
    if b.shell:
        cmd += ["--shell", "--shell-w", repr(b.shell_w), "--shell-kappa", b.shell_kappa, "--kappa-scale", repr(b.kappa_scale), "--tile-ref", repr(b.tile_ref)]
        if b.seed_only: cmd += ["--seed-only"]
        if not b.shell_frame: cmd += ["--no-shell-frame"]
    cmd += ["--stretch-scale", repr(b.stretch_scale), "--bend-scale", repr(b.bend_scale), "--straight", repr(b.straight), "--length-ratio", repr(b.length_ratio),
            "--ramp", str(b.ramp), "--ramp-iter", str(b.ramp_iter), "--seed-stitch", repr(b.seed_stitch), "--stall-tol", repr(b.stall_tol)]
    if b.macro: cmd += ["--macro"]
    if b.fold_search: cmd += ["--fold-search", str(b.fold_search), "--fold-iter", str(b.fold_iter), "--fold-delta", repr(b.fold_delta)]
    if b.warp_kx: cmd += ["--warp-kx", repr(b.warp_kx)]
    if b.warp_ky: cmd += ["--warp-ky", repr(b.warp_ky)]
    if b.ea: cmd += ["--ea", repr(b.ea)]
    if b.ei: cmd += ["--ei", repr(b.ei)]
    if b.stretch_law: cmd += ["--stretch-law", b.stretch_law]
    return cmd, kc

def _boot_id():
    try: return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError: return ""

def _launch(rid, d, cmd, resume=False):
    log = open(d / "job.log", "a" if resume else "w")
    (d / "job.json").write_text(json.dumps({"started": time.time(), "boot_id": _boot_id()}))          # before the job starts, so it can always read it; boot_id as in campaign runner.lock
    JOBS[rid] = subprocess.Popen(cmd + (["--resume"] if resume else []), stdout=log, stderr=subprocess.STDOUT, cwd=str(SCRIPTS), start_new_session=True)

@app.post("/api/results/{rid}/simulate")
def simulate(rid: str, b: SimBody):
    d = _folder(rid)
    if _any_running():
        raise HTTPException(409, "a simulation is already running")
    busy = campaign_api.runner_busy()
    if busy:
        raise HTTPException(409, busy)
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
    if (b.macro or b.fold_search or b.warp_kx or b.warp_ky) and b.shell:
        raise HTTPException(422, "the macro modes and the bent start are for runs without the shell: untick the shell")
    if b.ramp and b.shell:
        raise HTTPException(422, "growth stages are for runs without the shell: untick the shell or set the stages to 0")
    if b.seed_stitch and (b.shell or not meta.get("matrix")):
        raise HTTPException(422, "the stitch seed needs the stitch matrix and is for runs without the shell")
    if (b.ea or b.ei or b.stretch_law) and (meta.get("structure") or {}).get("units") != "mm":
        raise HTTPException(422, "EA, EI and the stretch curve are in N and mm: they need a result built in mm; leave them 0/empty for R-unit results")
    if b.ea and b.stretch_law:
        raise HTTPException(422, "give either EA or a stretch curve (the curve's first point already fixes EA)")
    if (b.ea or b.stretch_law) and b.stretch_scale != 1.0:
        raise HTTPException(422, "the axial stiffness scale does not apply together with EA or a stretch curve: set it to 1")
    if b.ei and b.bend_scale != 1.0:
        raise HTTPException(422, "the bending stiffness scale does not apply together with EI: set it to 1")
    if b.stretch_law:
        try: _law_points(b.stretch_law)
        except (ValueError, ZeroDivisionError) as e: raise HTTPException(422, "stretch curve: %s (format strain:cN,strain:cN with both rising)" % e)
    cmd, kc = _sim_cmd(d, b, R)
    for f in ("relaxed_axis.csv", "relaxed_axis.view.csv", "relaxed_tube.obj", "relaxed_tube.bin", "state.json", "job.json", STAGE_FILE, "shell.npz") + CKPT_FILES:
        (d / f).unlink(missing_ok=True)          # never leave a previous run's files next to a new one
    (d / "job_request.json").write_text(json.dumps(b.model_dump()))      # lets a power cut be resumed with exactly these settings
    _preset_write("_last", b.model_dump())
    sl = d / "settings_log.json"; hist = _read_json(sl) or []          # every setting of every Simulate press, shown in the UI log
    hist.append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "settings": b.model_dump()}); sl.write_text(json.dumps(hist, indent=1))
    _launch(rid, d, cmd)
    return {"started": rid, "k_contact": kc}

def _resumable(d):
    s = _read_json(d / "state.json") or {}
    return (s.get("state") in RUNNING and not _alive(d.name) and (d / "checkpoint.npz").exists() and (d / "job_request.json").exists())

def _auto_block(d):
    """Why a resumable folder must NOT be resumed by itself at startup ('' = it may). Only a job that was running when the machine rebooted qualifies: the boot id recorded in job.json
    must exist and differ from the current one. A job killed while the machine stayed up, a cancelled job and a job without a recorded boot id are left for the Resume button."""
    s = _read_json(d / "state.json") or {}
    sims = (_read_json(d / "meta.json") or {}).get("simulations") or []
    if s.get("state") == "cancelled" or (sims and sims[-1].get("status") == "cancelled"): return "it was cancelled"
    rec, now = (_read_json(d / "job.json") or {}).get("boot_id"), _boot_id()
    if not rec: return "no boot id recorded for the job"
    if not now: return "the current boot id cannot be read"
    if rec == now: return "the job was started in the current boot, so the machine did not reboot"
    return ""

def _resume(rid):
    d = _folder(rid); meta = _read_json(d / "meta.json") or {}
    b = SimBody(**(_read_json(d / "job_request.json") or {}))
    cmd, _ = _sim_cmd(d, b, meta["R"]); _launch(rid, d, cmd, resume=True)

@app.post("/api/results/{rid}/resume")
def resume(rid: str):
    d = _folder(rid)
    if _any_running():
        raise HTTPException(409, "a simulation is already running")
    busy = campaign_api.runner_busy()
    if busy:
        raise HTTPException(409, busy)
    if not _resumable(d):
        raise HTTPException(409, "nothing to resume: no checkpoint for an interrupted run")
    _resume(rid)
    return {"resumed": rid}

@app.on_event("startup")
def auto_resume():
    """After a power cut or reboot: continue the most recent interrupted run from its checkpoint (AUX_AUTORESUME=0 turns this off)."""
    if os.environ.get("AUX_AUTORESUME", "1") == "0":
        return
    busy = campaign_api.runner_busy()
    if busy:
        logging.getLogger("uvicorn.error").warning("auto-resume skipped: %s", busy)      # a second solver job next to a campaign run must not start by itself; resume by hand later
        return
    for d in sorted((p for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name, reverse=True):
        if not _resumable(d):
            continue
        why = _auto_block(d)
        if why:
            logging.getLogger("uvicorn.error").warning("auto-resume: %s left alone (%s); the Resume button still works", d.name, why)
            continue
        _resume(d.name); break

def _tile_alive(rid):
    p = JOBS.get(rid + "#tile")
    return p is not None and p.poll() is None

@app.post("/api/results/{rid}/tile")
def tile_start(rid: str, b: TileBody):
    d = _folder(rid)
    if _any_running():
        raise HTTPException(409, "a job is already running")
    busy = campaign_api.runner_busy()
    if busy:
        raise HTTPException(409, busy)
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
            "relaxed_at": sim["relaxed_at"] if sim else 0, "log": log,
            "checkpoint": _read_json(d / "checkpoint.json"), "resumable": _resumable(d),
            "stage": (_read_json(d / STAGE_FILE) or {}).get("stage") if _alive(rid) else None}

@app.get("/api/results/{rid}/log")
def sim_log(rid: str):
    """All runs of this result, oldest first. A run still marked running whose job is gone (power cut, killed) is reported as interrupted."""
    d = _folder(rid); sims = (_read_json(d / "meta.json") or {}).get("simulations") or []
    for i, r in enumerate(sims):
        if r.get("status") not in ("settled", "failed", "cancelled", "interrupted") and not (i == len(sims) - 1 and _alive(rid)):
            r["status"] = "interrupted"
    hist = _read_json(d / "settings_log.json") or []
    for r in sims:
        prev = [h for h in hist if h["t"] <= (r.get("started") or "")]          # the latest Simulate press at or before this run started
        if prev: r["ui_settings"] = prev[-1]["settings"]
    return sims

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
    mp = d / "meta.json"; m = _read_json(mp) or {}
    if m.get("simulations") and m["simulations"][-1].get("status") not in ("settled", "failed", "cancelled"):
        m["simulations"][-1].update(status="cancelled", finished=time.strftime("%Y-%m-%dT%H:%M:%S")); tmpm = d / "meta.json.tmp"; tmpm.write_text(json.dumps(m, indent=1)); os.replace(tmpm, mp)
    (d / STAGE_FILE).unlink(missing_ok=True)
    s = _read_json(d / "state.json") or {}
    s["state"] = "cancelled"
    s.setdefault("history", []).append({"state": "cancelled", "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": "cancelled from the UI"})
    tmp = d / "state.json.tmp"; tmp.write_text(json.dumps(s, indent=1)); os.replace(tmp, d / "state.json")
    return {"cancelled": rid}

@app.delete("/api/results/{rid}")
def delete_result(rid: str):
    d = _folder(rid)
    if _alive(rid) or _tile_alive(rid):
        raise HTTPException(409, "cannot delete while a job is running on this result")
    shutil.rmtree(d)
    JOBS.pop(rid, None); JOBS.pop(rid + "#tile", None)
    return {"deleted": rid}

# ---------------------------------------------------------------- presets (saved settings)
def _presets():
    return _read_json(PRESET_FILE) or {}

def _preset_write(name, data):
    P = _presets(); P[name] = data
    tmp = PRESET_FILE.with_suffix(".tmp"); tmp.write_text(json.dumps(P, indent=1)); os.replace(tmp, PRESET_FILE)

PRESET_RE = re.compile(r"^[\w .()+-]{1,60}$")

@app.get("/api/presets")
def presets_list():
    return _presets()                         # "_last" = settings of the most recent Simulate press

@app.put("/api/presets/{name}")
def presets_save(name: str, b: SimBody):
    if not PRESET_RE.match(name) or name.startswith("_"):
        raise HTTPException(422, "preset name: letters, digits, space . ( ) + - _ , max 60, not starting with _")
    _preset_write(name, b.model_dump()); return {"saved": name}

@app.delete("/api/presets/{name}")
def presets_delete(name: str):
    P = _presets()
    if name not in P or name.startswith("_"):
        raise HTTPException(404)
    del P[name]; tmp = PRESET_FILE.with_suffix(".tmp"); tmp.write_text(json.dumps(P, indent=1)); os.replace(tmp, PRESET_FILE); return {"deleted": name}

# ---------------------------------------------------------------- naming
class RenameBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)

@app.post("/api/results/{rid}/rename")
def rename(rid: str, b: RenameBody):
    d = _folder(rid)
    if _alive(rid):
        raise HTTPException(409, "cannot rename while its simulation is running")
    mp = d / "meta.json"; m = _read_json(mp) or {}
    m["name"] = b.name.strip(); tmp = d / "meta.json.tmp"; tmp.write_text(json.dumps(m, indent=1)); os.replace(tmp, mp)
    return {"id": rid, "name": m["name"]}

# ---------------------------------------------------------------- new fabric: photo -> matrix, build
RED_BGR = (87.0, 87.0, 255.0); GREEN_BGR = (87.0, 217.0, 126.0)      # measured reference colours of the grid editor export (from crop_classify.py)

def _decode(data):
    import numpy as np
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None: return img.astype(float)
    except ImportError:
        pass
    try:
        from PIL import Image; import io
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=float)[:, :, ::-1]      # to BGR
    except Exception:
        return None

@app.post("/api/classify")
async def classify(request: Request, rows: int, cols: int):
    """Body = raw image bytes. Same method as crop_classify.py: coloured-cell bounding box (saturation > 30), equal cells, mean colour of the inner 70% of each cell,
    nearest of the two reference colours (1 = red = front, 2 = green = back). Returns the matrix, the grid box and how sure each cell is."""
    import numpy as np
    if not (1 <= rows <= 200 and 1 <= cols <= 200):
        raise HTTPException(422, "rows and columns must be 1 to 200")
    data = await request.body()
    if len(data) > 25_000_000: raise HTTPException(413, "image too large (25 MB max)")
    img = _decode(data)
    if img is None: raise HTTPException(422, "could not read the image")
    mx, mn = img.max(2), img.min(2); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9) * 255.0, 0.0)
    ys, xs = np.where(sat > 30)
    if ys.size == 0: raise HTTPException(422, "no coloured grid found in the image")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    ch, cw = (y1 - y0) / rows, (x1 - x0) / cols
    if ch <= 1 or cw <= 1: raise HTTPException(422, "image too small for that many rows and columns")
    red, green = np.array(RED_BGR), np.array(GREEN_BGR); M, conf = [], []
    for r in range(rows):
        row, crow = [], []
        for c in range(cols):
            ya, yb = int(y0 + r*ch + 0.15*ch), int(y0 + (r+1)*ch - 0.15*ch); xa, xb = int(x0 + c*cw + 0.15*cw), int(x0 + (c+1)*cw - 0.15*cw)
            cell = img[max(0, ya):max(ya+1, yb), max(0, xa):max(xa+1, xb)]
            m = cell.reshape(-1, 3).mean(0); dr, dg = np.linalg.norm(m - red), np.linalg.norm(m - green)
            row.append(1 if dr < dg else 2); crow.append(round(float(abs(dr - dg) / max(dr + dg, 1e-9)), 3))      # 0 = ambiguous, 1 = clear
        M.append(row); conf.append(crow)
    low = [[r, c] for r in range(rows) for c in range(cols) if conf[r][c] < 0.25]
    return {"rows": rows, "cols": cols, "matrix": M, "confidence": conf, "low_confidence": low, "box": [x0, y0, x1, y1], "size": [int(img.shape[1]), int(img.shape[0])]}

class BuildBody(BaseModel):
    name: str = Field(min_length=1, max_length=80, description="model name, chosen by the user")
    matrix: str = Field(pattern=r"^[12 ]+(/[12 ]+)*$", description="rows separated by /, digits 1 front, 2 back; first row = bottom course")
    tiles: int = Field(4, ge=1, le=40)
    ne: float = Field(10.0, gt=0, le=200); cpi: float = Field(gt=0, le=200); wpi: float = Field(gt=0, le=200)

BUILD = {"proc": None, "id": None, "name": None, "tmp": None, "done": None, "error": ""}

def _unique_id():
    base = datetime.now().strftime("%Y%m%d-%H%M%S"); rid, n = base, 1
    while (RESULTS / rid).exists() or (RESULTS / (".build-" + rid)).exists():
        n += 1; rid = "%s-%d" % (base, n)
    return rid

def _build_watch(p, tmp, rid):
    p.wait()
    if p.returncode == 0 and (tmp / "meta.json").exists():
        os.rename(tmp, RESULTS / rid); BUILD["done"] = True      # atomic: the list never shows a half-built result
    else:
        BUILD["done"] = False; BUILD["error"] = "build failed (exit %s); see the log below" % p.returncode

@app.post("/api/build")
def build(b: BuildBody):
    if BUILD["proc"] is not None and BUILD["proc"].poll() is None:
        raise HTTPException(409, "a build is already running")
    rows = [r.replace(" ", "") for r in b.matrix.split("/")]
    if len({len(r) for r in rows}) != 1 or not rows[0]:
        raise HTTPException(422, "the matrix rows must all have the same length")
    rid = _unique_id(); tmp = RESULTS / (".build-" + rid)
    cmd = [sys.executable, str(SCRIPTS / "build_knit.py"), "--matrix", "/".join(" ".join(r) for r in rows), "--tiles", str(b.tiles), "--ne", repr(b.ne), "--cpi", repr(b.cpi), "--wpi", repr(b.wpi),
           "--out", str(tmp), "--no-tube", "--name", b.name.strip(), "--id", rid]       # id = creation date and time; the folder carries the same id; the name is the user's
    tmp.mkdir(parents=True, exist_ok=True)
    log = open(tmp / "build.log", "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SCRIPTS), start_new_session=True)
    BUILD.update(proc=p, id=rid, name=b.name.strip(), tmp=tmp, done=None, error="")
    threading.Thread(target=_build_watch, args=(p, tmp, rid), daemon=True).start()
    return {"id": rid}

@app.get("/api/build/status")
def build_status():
    p = BUILD["proc"]
    if p is None: return {"running": False, "id": None}
    d = RESULTS / BUILD["id"] if BUILD["done"] else BUILD["tmp"]
    try: log = "\n".join((d / "build.log").read_text().splitlines()[-8:])
    except Exception: log = ""
    return {"running": p.poll() is None, "id": BUILD["id"], "name": BUILD["name"], "done": BUILD["done"], "error": BUILD["error"], "log": log}

# ---------------------------------------------------------------- server status strip
@app.get("/api/status")
def sys_status():
    out = {"load": os.getloadavg()[0] if hasattr(os, "getloadavg") else None, "sim_running": _any_running(), "classify": True}
    try:
        import psutil
        v = psutil.virtual_memory(); du = psutil.disk_usage("/")
        out.update(cpu=psutil.cpu_percent(interval=None), ram_pct=v.percent, disk_free_gb=round(du.free / 2**30, 1), uptime_h=round((time.time() - psutil.boot_time()) / 3600, 1))
        t = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
        for k in ("coretemp", "k10temp"):
            if t.get(k): out["temp_c"] = round(t[k][0].current, 1); break
    except Exception:
        pass
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text()      # re-read each time: editing index.html needs no restart