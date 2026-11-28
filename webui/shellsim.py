"""Single shell runs from the Simulator page (2026-11-28, K: "New" -> Shell instead of yarn mesh -> matrix, CPI, WPI). One pure-shell run of scripts/campaigns/shell_run.py with the
shell2 settings (campaigns/shell2_policy.json 'essential' + 'no_effect': plate bending, FIRE 2000 steps per growth step, stability check), so a single run is made exactly like a shell2 run;
only the pattern, CPI, WPI and tiles are chosen. It is NOT part of any campaign: it never enters the rules, the proposer or the analysis.
Folder: webui/results_shell/<id>/ (id = creation date and time): request.json (name, matrix as entered, settings), settings.json, job.json (pid, boot id, process start), shell.log,
and the files of shell_run.py (shell.npz, shell.bin, metrics.json last = complete). One run at a time, and none while a yarn simulation of the web page is running.
Routes: POST /api/shellsim, GET /api/shellsim, GET /api/shellsim/{id}, /mesh, /colors, POST /api/shellsim/{id}/cancel."""
import json, os, re, subprocess, sys, time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
ROOT = HERE.parent
SCRIPTS = ROOT / "scripts"
DIR = Path(os.environ.get("AUX_SHELLSIM_DIR") or HERE / "results_shell")
POLICY = ROOT / "campaigns" / "shell2_policy.json"
ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}$")
MAX_STITCHES = 3600            # the shell2 limit (campaigns/shell2_policy.json ranges.max_stitches): larger sheets take long on a CPU
router = APIRouter()
_api = None                    # campaign_api (module), set by setup(): process checks and the module loader
_busy = lambda: False          # True while a yarn job of the web page runs


def setup(app, campaign_api, any_running):
    global _api, _busy
    _api, _busy = campaign_api, any_running
    app.include_router(router)


def _pc(): return _api._load("platform_compat", SCRIPTS / "campaigns" / "platform_compat.py")
def _rj(p, d=None):
    try: return json.loads(Path(p).read_text(encoding="utf8"))
    except (OSError, ValueError): return d


def _alive(job):
    try: return bool(job and _api._cm().alive(job))
    except Exception: return False


def _state(d):
    if (d / "metrics.json").is_file(): return "done"
    job = _rj(d / "job.json")
    if _alive(job): return "running"
    if job and job.get("cancelled"): return "cancelled"
    return "failed" if job else "queued"


def _run_dir(rid):
    if not isinstance(rid, str) or not ID_RE.match(rid): raise HTTPException(404, "unknown shell run")
    d = DIR / rid
    if not d.is_dir() or d.is_symlink(): raise HTTPException(404, "unknown shell run")
    return d


def _running():
    return [d.name for d in DIR.glob("*") if d.is_dir() and ID_RE.match(d.name) and _state(d) == "running"] if DIR.is_dir() else []


class ShellBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    matrix: str = Field(..., description="rows separated by '/', digits 1 (knit) and 2 (purl) separated by spaces; row 1 = bottom course (as /api/build)")
    cpi: float = Field(..., gt=4, le=80)
    wpi: float = Field(..., gt=4, le=80)
    tiles: int = Field(4, ge=1, le=8)


def _settings(b):
    rows = [[int(v) for v in r.split()] for r in b.matrix.strip().split("/") if r.strip()]
    if not rows or any(len(r) != len(rows[0]) for r in rows) or any(v not in (1, 2) for r in rows for v in r): raise HTTPException(422, "the matrix must be rows of equal length with digits 1 and 2")
    if not (2 <= len(rows) <= 60 and 2 <= len(rows[0]) <= 60): raise HTTPException(422, "each side of the matrix must be 2 to 60 stitches")
    n = len(rows) * len(rows[0]) * b.tiles ** 2
    if n > MAX_STITCHES: raise HTTPException(422, "%d stitches after tiling: the limit is %d (fewer tiles or a smaller matrix)" % (n, MAX_STITCHES))
    pol = _rj(POLICY, {}) or {}
    s = dict(pol.get("essential") or {}); s.update(pol.get("no_effect") or {}); s.update(matrix=rows, cpi=float(b.cpi), wpi=float(b.wpi), tiles=int(b.tiles), fold_min_turn_rad=0.2)
    missing = [k for k in ("kappa_scale", "h_scale", "nu", "springs", "sub", "steps", "iter", "seed") if k not in s]
    if missing: raise HTTPException(500, "campaigns/shell2_policy.json has no %s" % ", ".join(missing))
    return s, n


@router.post("/api/shellsim")
def shellsim_start(b: ShellBody):
    s, n = _settings(b)
    if _running(): raise HTTPException(409, "a shell run is already running; wait for it to finish")
    if _busy(): raise HTTPException(409, "a yarn simulation of this page is running; wait for it to finish")
    rid = time.strftime("%Y%m%d-%H%M%S")
    while (DIR / rid).exists(): time.sleep(1); rid = time.strftime("%Y%m%d-%H%M%S")
    d = DIR / rid; d.mkdir(parents=True)
    (d / "request.json").write_text(json.dumps(dict(name=b.name.strip(), created=time.strftime("%Y-%m-%dT%H:%M:%S"), stitches=n, settings=s), indent=1), encoding="utf8")
    (d / "settings.json").write_text(json.dumps(s), encoding="utf8")
    pc = _pc(); py = str(_api.PROJECT_PY) if _api.PROJECT_PY.exists() else sys.executable
    cmd = ([] if os.name == "nt" else ["nice", "-n", "10"]) + [py, str(SCRIPTS / "campaigns" / "shell_run.py"), "--settings", str(d / "settings.json"), "--out", str(d)]
    log = open(d / "shell.log", "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SCRIPTS / "campaigns"), **pc.detach_kwargs()); log.close()
    pi = pc.proc_info(p.pid)
    (d / "job.json").write_text(json.dumps(dict(pid=p.pid, boot_id=_api._cm().boot_id(), start_ticks=pi[0] if pi else None, started=time.strftime("%Y-%m-%dT%H:%M:%S"), cmd=cmd)), encoding="utf8")
    return {"id": rid, "stitches": n, "message": "shell run %s started (%d stitches)" % (rid, n)}


def _summary(d):
    rq = _rj(d / "request.json", {}) or {}; s = rq.get("settings") or {}; m = _rj(d / "metrics.json", {}) or {}
    return dict(id=d.name, name=rq.get("name", d.name), created=rq.get("created"), state=_state(d), stitches=rq.get("stitches"), cpi=s.get("cpi"), wpi=s.get("wpi"), tiles=s.get("tiles"),
                matrix="%dx%d" % (len(s["matrix"]), len(s["matrix"][0])) if s.get("matrix") else "", seconds=m.get("solver_seconds"), fold_direction=m.get("fold_direction"))


@router.get("/api/shellsim")
def shellsim_list():
    if not DIR.is_dir(): return []
    return [_summary(d) for d in sorted((d for d in DIR.iterdir() if d.is_dir() and ID_RE.match(d.name)), reverse=True)]


@router.get("/api/shellsim/{rid}")
def shellsim_detail(rid: str):
    d = _run_dir(rid); rq = _rj(d / "request.json", {}) or {}
    try: tail = [l.rstrip() for l in open(d / "shell.log", errors="replace")][-25:]
    except OSError: tail = []
    g = [l for l in tail if l.strip().startswith("growth")]
    return dict(_summary(d), run=rid, label=rq.get("name", ""), source="simulator", settings=rq.get("settings") or {}, metrics=_rj(d / "metrics.json", {}) or {}, log_tail=tail,
                progress=dict(growth_steps_done=len(g), steps=(rq.get("settings") or {}).get("steps")), files={"shell.bin": (d / "shell.bin").is_file()})


@router.get("/api/shellsim/{rid}/mesh")
def shellsim_mesh(rid: str):
    f = _run_dir(rid) / "shell.bin"
    if not f.is_file(): raise HTTPException(404, "not finished yet")
    return FileResponse(f, media_type="application/octet-stream")


@router.get("/api/shellsim/{rid}/colors")
def shellsim_colors(rid: str, field: str = "area_strain", clip: float = 98.0):
    if not (50.0 <= clip <= 100.0): raise HTTPException(422, "clip must be 50 to 100")
    f = _run_dir(rid) / "shell.npz"
    if not f.is_file(): raise HTTPException(404, "not finished yet")
    _api._sc(); ss = _api._load("shell_strain", SCRIPTS / "shell_model" / "shell_strain.py")          # _sc() first: it puts the script folders on sys.path
    try: X, W, C, sub = ss.load(f); rgb, legend = ss.colours(field, ss.vertex_field(X, W, C, sub, field), clip)
    except ValueError as e: raise HTTPException(422, str(e))
    return Response(rgb.tobytes(), media_type="application/octet-stream", headers={"X-Count": str(len(rgb)), "X-Legend": json.dumps(legend)})


@router.delete("/api/shellsim/{rid}")
def shellsim_delete(rid: str):
    """Remove one simulator shell run (K 2026-11-28: the trash can in the Simulator list, as for yarn results). Refused while it runs. Only a folder directly inside DIR."""
    d = _run_dir(rid)
    if _state(d) == "running": raise HTTPException(409, "this shell run is still running: cancel it first")
    if d.resolve().parent != DIR.resolve(): raise HTTPException(404, "unknown shell run")
    import shutil; shutil.rmtree(d)
    return {"deleted": rid}


@router.post("/api/shellsim/{rid}/cancel")
def shellsim_cancel(rid: str):
    d = _run_dir(rid); job = _rj(d / "job.json")
    if not _alive(job): raise HTTPException(409, "this shell run is not running")
    _pc().kill_tree(job["pid"]); job["cancelled"] = time.strftime("%Y-%m-%dT%H:%M:%S"); (d / "job.json").write_text(json.dumps(job), encoding="utf8")
    return {"cancelled": rid}
