"""Read-only support for viewing campaign runs (campaigns/<name>/runs/<id>) and for per-vertex fields on any result folder.

Nothing here writes into a run folder. Generated files (baseline tube, axis view copies) go to campaigns/<name>/cache/<id>/.
A run is addressed in URLs as  run~<campaign>~<id>  (the viewer link is  /?run=<campaign>/<id>); the usual result ids are unchanged.

Fields (all computed from files the run already has):
  looplen  yarn arc length of each loop divided by the as-built mean loop length   (loop = one stitch of one course; column = floor(x / W) of the as-built vertex, the same rule relax_job.py uses)
  strain   axial strain of the relaxed fabric = segment length / rest length - 1, per segment, shown per vertex (mean of the two adjacent segments)
The saved outputs do NOT contain the rest length of each segment after relaxation (it lives only in checkpoint.npz, which relax_job.py deletes at the end). The rest length is therefore
reconstructed: as-built chord length of the resampled yarn x rest_length_ratio (from length_ratio, tension and the stretch law in meta.json), exactly as relax_job.py sets it at the start.
Rest-length migration during the relaxation is not included (see the notes returned with every field). A file relaxed_l0.csv (course,seg,l0), if a later relax_job.py writes one, is used instead.
"""
import json, math, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
from fastapi import APIRouter, HTTPException, Response

HERE = Path(__file__).parent
ROOT = HERE.parent
CAMPAIGNS = Path(os.environ.get("CAMPAIGN_DIR") or ROOT / "campaigns")
SCRIPTS = Path(os.environ.get("CAMPAIGN_SCRIPTS") or ROOT / "scripts")
RESULTS = HERE / "results"
ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")
RUN_ID_RE = re.compile(r"^run~([0-9A-Za-z_-]{1,64})~([0-9]{1,6})$")
CLIP_PERCENTILE = 98.0        # UNSOURCED: the strain colour range is +-(this percentile of |strain|); a display choice, adjustable with ?clip=
router = APIRouter()


def setup(app, results_dir):
    global RESULTS
    RESULTS = Path(results_dir)
    app.include_router(router)


# ------------------------------------------------------------------ path safety
def is_run(rid): return bool(RUN_ID_RE.match(rid))

def run_dir(rid):
    """The run folder for run~<campaign>~<id>. The resolved path must be exactly campaigns/<campaign>/runs/<id> (a symlink or a name that escapes is refused)."""
    m = RUN_ID_RE.match(rid)
    if not m: raise HTTPException(404, "unknown run")
    name, num = m.groups()
    root = CAMPAIGNS.resolve()
    cdir = (CAMPAIGNS / name).resolve()
    if cdir.parent != root or not cdir.is_dir(): raise HTTPException(404, "unknown campaign")
    rd = (cdir / "runs" / num).resolve()
    if rd.parent != cdir / "runs" or not rd.is_dir(): raise HTTPException(404, "unknown run")
    return rd

def view_dir(rid):
    """Folder a read-only view endpoint may read: a run folder, or webui/results/<id> as before."""
    if is_run(rid): return run_dir(rid)
    if not ID_RE.match(rid) or not (RESULTS / rid).is_dir(): raise HTTPException(404, "unknown result")
    return RESULTS / rid

def cache_dir(rid):
    name, num = RUN_ID_RE.match(rid).groups()
    c = CAMPAIGNS / name / "cache" / num
    c.mkdir(parents=True, exist_ok=True)
    return c

def _rjson(p, default=None):
    try: return json.loads(Path(p).read_text(), parse_constant=lambda c: None)
    except Exception: return default


# ------------------------------------------------------------------ tube on demand (never into the run folder)
_locks, _glock = {}, threading.Lock()

def _progress_path(c, name): return c / ("progress_%s.json" % name)

def ensure_tube(rid, d, rel):
    """Folder that holds the tube mesh of this variant. The run keeps the relaxed_tube.bin that relax_job.py wrote; anything else is generated, niced, into the cache folder."""
    if rel and (d / "relaxed_tube.bin").exists(): return d
    name = ("relaxed" if rel else "fab") + "_tube.bin"
    src = d / ("relaxed_axis.csv" if rel else "fab_axis.csv")
    if not src.exists(): raise HTTPException(404, "no %s to build the tube from" % src.name)
    c = cache_dir(rid); out = c / name
    def fresh(): return out.exists() and out.stat().st_size > 16 and out.stat().st_mtime >= src.stat().st_mtime
    if fresh(): return c
    with _glock: lk = _locks.setdefault(str(out), threading.Lock())
    with lk:
        if fresh(): return c                                  # another request built it while this one waited
        meta = _rjson(d / "meta.json") or {}
        R = meta.get("R")
        if not R: raise HTTPException(422, "meta.json has no R")
        sides = int(((meta.get("tube") or {}).get("sides")) or 8)      # build_knit.py --sides default is 8; a --no-tube build records no tube entry
        nice = shutil.which("nice")
        cmd = ([nice, "-n", "19"] if nice else []) + [sys.executable, str(HERE / "runview.py"), "gen", str(src), str(out), repr(float(R)), str(sides), str(_progress_path(c, name))]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not fresh():
            raise HTTPException(500, "tube generation failed: " + (p.stderr or p.stdout).strip()[-300:])
    return c

def _gen(src, out, R, sides, prog):
    """Child process: the same tube writer as build_knit.py (write_obj: decimation, rotation-minimising frames, sweep_tube), progress in a JSON file."""
    sys.path.insert(0, str(SCRIPTS))
    import build_knit as bk
    t0 = time.time(); out = Path(out)
    def note(stage, done=0, total=0, finished=False, error=""):
        tmp = prog + ".tmp"
        Path(tmp).write_text(json.dumps(dict(stage=stage, done=done, total=total, pct=(100.0 * done / total if total else 0.0), elapsed_s=round(time.time() - t0, 2), finished=finished, error=error)))
        os.replace(tmp, prog)
    note("reading axis")
    A = np.loadtxt(src, delimiter=",", skiprows=1)
    curves = [A[A[:, 0] == c][:, 1:] for c in np.unique(A[:, 0])]
    total, cnt = len(curves), [0]
    orig = bk.sweep_tube
    def counted(curve, R_, sides_):
        cnt[0] += 1
        if cnt[0] % 4 == 0 or cnt[0] == total: note("sweeping tubes", cnt[0], total)
        return orig(curve, R_, sides_)
    bk.sweep_tube = counted
    tmp_obj = out.with_name(".gen_" + out.stem + ".obj")
    nv, nf = bk.write_obj(str(tmp_obj), curves, float(R), int(sides), "tube from %s" % Path(src).name, obj_max=0)      # obj_max=0: binary mesh only
    os.replace(tmp_obj.with_suffix(".bin"), out)
    note("done", total, total, True)
    print("wrote %s: %d vertices, %d triangles, %.1f s" % (out, nv, nf, time.time() - t0))


# ------------------------------------------------------------------ geometry helpers
def resample(P, ds):
    """Same as relax_state.resample (checked equal in the tests)."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.linspace(0, s[-1], max(3, int(round(s[-1] / ds)) + 1))
    return np.c_[[np.interp(u, s, P[:, k]) for k in range(3)]].T

def decim_idx(P, R, max_step=0.8, max_turn_deg=6.0):
    """Indices that build_knit.decimate_curve keeps (same algorithm; the tests compare it with the original)."""
    P = np.asarray(P, float)
    if len(P) < 3: return np.arange(len(P))
    keep = [0]; cmax = np.cos(np.radians(max_turn_deg)); d0 = None
    for i in range(1, len(P) - 1):
        arc = np.linalg.norm(P[i] - P[keep[-1]])
        if d0 is None: d0 = (P[i] - P[i - 1]); d0 = d0 / (np.linalg.norm(d0) + 1e-30)
        di = P[i + 1] - P[i]; di = di / (np.linalg.norm(di) + 1e-30)
        if arc >= max_step * R or float(d0 @ di) < cmax: keep.append(i); d0 = None
    keep.append(len(P) - 1)
    return np.array(keep)

def read_axis(path):
    A = np.loadtxt(path, delimiter=",", skiprows=1)
    cs = np.unique(A[:, 0]).astype(int)
    return cs, [A[A[:, 0] == c][:, 1:] for c in cs]


# ------------------------------------------------------------------ fields
class Fields:
    pass

def _rest_ratio(meta, R):
    """rest_length_ratio exactly as relax_job.py computes it: length_ratio / (1 + strain under the feeder tension). Returns (ratio, note, exact)."""
    sims = meta.get("simulations") or []
    p = (sims[-1].get("params") if sims else None) or {}
    if not p: return 1.0, "meta.json has no simulation parameters: rest length taken as the as-built length (ratio 1)", False
    lr = float(p.get("length_ratio") or 1.0); T = float(p.get("tension_cN") or 0.0)
    law, ea = p.get("stretch_law") or "", float(p.get("ea") or 0.0)
    if law:
        pts = [(float(a), 0.01 * float(c)) for a, c in (t.split(":") for t in law.split(","))]
        s0 = pts[0][1] / pts[0][0]; K = np.array([0.0] + [a for a, _ in pts]); Tn = np.array([0.0] + [f / s0 for _, f in pts])
        t = 0.01 * T / s0
        slope = (K[-1] - K[-2]) / (Tn[-1] - Tn[-2]) if len(K) > 1 else 1.0
        eps = float(np.interp(t, Tn, K)) if t <= Tn[-1] else float(K[-1] + (t - Tn[-1]) * slope)
    else:
        EA = ea if ea else float(p.get("E") or 0.0) * math.pi * R * R * float(p.get("stretch_scale") or 1.0)
        eps = 0.01 * T / EA if EA else 0.0
    ratio = lr / (1.0 + eps)
    exact = not (p.get("ramp") or 0)
    return ratio, "rest length = as-built chord x %.5f (length_ratio %g, tension %g cN, strain under tension %.5f)%s" % (ratio, lr, T, eps, "" if exact else "; growth stages (ramp) were used, so this is not exact"), exact

_FCACHE = {}
LOOP_HALF_MIN = 0.01     # UNSOURCED display choice: the loop-length colour range is 1 +- h with h = max(clip-th percentile of |ratio - 1|, 0.01), so a field that is flat within 1 % shows one mid colour instead of noise

def _cum_at(cum, p):
    i = np.minimum(np.floor(p).astype(int), len(cum) - 2)
    return cum[i] + (p - i) * (cum[i + 1] - cum[i])

def _loop_sum(v, beta):
    """Sum of a per-segment quantity over each loop. beta = loop boundaries in segment units (fractional): the segments that straddle a boundary are split in proportion, so the result is not
    quantised to whole segments (the old whole-segment rule moved +-1 segment, about +-2.4 % of a loop, between neighbouring loops)."""
    a = _cum_at(np.r_[0.0, np.cumsum(v)], beta)
    return a[1:] - a[:-1]

def _boundaries(P, W, ncol):
    """Fine-curve sample indices of the ncol + 1 loop boundaries of one course, from the build itself: build_knit.py builds every stitch from control points that start and end exactly on x = k W
    (16 control points per stitch) and the Catmull-Rom curve passes through its control points, so the samples with x / W an integer are the loop boundaries. None if that does not give ncol + 1."""
    w = P[:, 0] / W; b = np.where(np.abs(w - np.round(w)) < 1e-6)[0]
    if len(b) != ncol + 1 or b[0] != 0 or b[-1] != len(P) - 1 or not np.array_equal(np.round(w[b]).astype(int), np.arange(ncol + 1)): return None
    return b

def fields(d):
    """Loop and strain data of one folder, cached while its files do not change."""
    d = Path(d)
    files = [d / n for n in ("meta.json", "fab_axis.csv", "relaxed_axis.csv", "relaxed_l0.csv")]
    key = (str(d), "v3") + tuple((f.stat().st_mtime_ns if f.exists() else 0) for f in files)
    if key in _FCACHE: return _FCACHE[key]
    meta = _rjson(d / "meta.json") or {}
    st = meta.get("structure") or {}; R = meta.get("R"); W = st.get("W") or meta.get("W"); C = st.get("C") or meta.get("C")
    M = meta.get("matrix"); tiles = meta.get("tiles")
    if not (R and W and C and M and tiles) or not (d / "fab_axis.csv").exists(): raise HTTPException(409, "this folder has no R, W, C, matrix and fab_axis.csv: no loop data")
    f = Fields(); f.d = d; f.meta = meta; f.R, f.W, f.C = float(R), float(W), float(C)
    f.ncol, f.nrow = len(M[0]) * int(tiles), len(M) * int(tiles); f.units = st.get("units") or ("mm" if meta.get("ne") else "R-units")
    cs, F = read_axis(d / "fab_axis.csv"); f.courses = cs; f.F = F
    f.X0 = [resample(P, f.R) for P in F]
    f.has_relaxed = (d / "relaxed_axis.csv").exists()
    f.Xr = None
    if f.has_relaxed:
        cr, Xr = read_axis(d / "relaxed_axis.csv")
        if list(cr) == list(cs) and all(len(a) == len(b) for a, b in zip(Xr, f.X0)): f.Xr = Xr
    # loop boundaries: exact from the build where possible. beta = boundary position in segment units of the resampled curve (the solver's vertex i of a course is the as-built vertex i).
    f.bidx, f.beta, f.vid0, f.vidF, fallback = [], [], [], [], 0
    for P, X in zip(F, f.X0):
        b = _boundaries(P, f.W, f.ncol); n = len(X); s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        if b is not None:
            beta = s_[b] / (s_[-1] / (n - 1)); vF = np.searchsorted(b, np.arange(len(P)), side="right") - 1
        else:        # fallback (older or foreign builds): the first crossing of x = k W along the curve; counted and reported
            fallback += 1; xm = np.maximum.accumulate(X[:, 0]); beta = np.interp(np.arange(f.ncol + 1) * f.W, xm, np.arange(n), right=n - 1); vF = np.clip((P[:, 0] // f.W).astype(int), 0, f.ncol - 1)
        beta = np.maximum.accumulate(np.clip(beta, 0, n - 1)); beta[0], beta[-1] = 0.0, n - 1.0
        f.bidx.append(b); f.beta.append(beta); f.vid0.append(np.clip(np.searchsorted(beta, np.arange(n), side="right") - 1, 0, f.ncol - 1)); f.vidF.append(np.clip(vF, 0, f.ncol - 1))
    f.identity = "loop boundaries from the build (control points on x = k W)" if not fallback else "loop boundaries from the first crossing of x = k W (%d of %d courses had no exact build boundaries)" % (fallback, len(cs))
    f.l_asb = [np.linalg.norm(np.diff(X, axis=0), axis=1) for X in f.X0]
    nc = len(cs)
    arc = lambda L: np.array([_loop_sum(L[i], f.beta[i]) for i in range(nc)])
    f.span = np.array([np.diff(b) for b in f.beta])                        # segments (fractional) per loop
    f.arc_asb = arc(f.l_asb)
    f.mean_asb = float(f.arc_asb[f.arc_asb > 0].mean())
    f.rest_src, f.rest_note, f.rest_exact = "reconstructed", "", False
    f.l_rel = f.arc_rel = f.l0 = f.strain = f.strain_grid = f.rest_sum = None
    f.ratio, _n, _e = _rest_ratio(meta, f.R); f.rest0_mean = f.ratio * f.mean_asb      # the as-built rest length of a loop is ratio x its as-built chord length
    if f.Xr is not None:
        f.l_rel = [np.linalg.norm(np.diff(X, axis=0), axis=1) for X in f.Xr]; f.arc_rel = arc(f.l_rel)
        l0file = d / "relaxed_l0.csv"
        if l0file.exists():
            T = np.loadtxt(l0file, delimiter=",", skiprows=1); f.l0 = [T[T[:, 0] == c][:, 2] for c in cs]
            f.rest_src, f.rest_note, f.rest_exact = "saved", "rest lengths read from relaxed_l0.csv", True
            if any(len(a) != len(b) for a, b in zip(f.l0, f.l_rel)): raise HTTPException(409, "relaxed_l0.csv does not match relaxed_axis.csv")
        else:
            f.rest_note = _n; f.l0 = [l * f.ratio for l in f.l_asb]
        f.rest_sum = arc(f.l0)
        f.strain = [lr / l0 - 1.0 for lr, l0 in zip(f.l_rel, f.l0)]
        f.strain_grid = np.array([_loop_sum(f.strain[i], f.beta[i]) / np.maximum(f.span[i], 1e-12) for i in range(nc)])
    if len(_FCACHE) >= 3: _FCACHE.pop(next(iter(_FCACHE)))
    _FCACHE[key] = f
    return f

def _vertex_values(f, field, variant):
    """(values per source vertex in the file order of the variant's axis csv, the field actually used).
    loop    = restlen when the run has saved rest lengths and the relaxed state is asked for, otherwise looplen
    looplen = yarn ARC LENGTH of each loop in the shown geometry / as-built mean loop length (geometric: does not show migration)
    restlen = SUM OF THE REST LENGTHS l0 of each loop / as-built mean rest length of a loop (needs relaxed_l0.csv): this is the migrated length
    strain  = axial strain of the relaxed fabric, per vertex"""
    if field == "loop": field = "restlen" if (variant == "relaxed" and f.rest_src == "saved") else "looplen"
    if field in ("looplen", "restlen"):
        if variant == "relaxed":
            if f.Xr is None: raise HTTPException(404, "no relaxed axis for this folder")
            if field == "restlen":
                if f.rest_src != "saved": raise HTTPException(404, "the rest lengths after the relaxation are not saved for this run (no relaxed_l0.csv)")
                grid = f.rest_sum / f.rest0_mean
            else: grid = f.arc_rel / f.mean_asb
            ids = f.vid0
        else:
            if field == "restlen": raise HTTPException(404, "the as-built rest length is the as-built loop length times a constant: use the loop length")
            grid, ids = f.arc_asb / f.mean_asb, f.vidF
        return np.concatenate([grid[i][ids[i]] for i in range(len(f.courses))]), field
    if field == "strain":
        if variant != "relaxed" or f.strain is None: raise HTTPException(404, "strain is defined for the relaxed fabric only")
        out = []
        for s_ in f.strain:
            v = np.empty(len(s_) + 1); v[0] = s_[0]; v[-1] = s_[-1]; v[1:-1] = (s_[:-1] + s_[1:]) / 2.0; out.append(v)
        return np.concatenate(out), "strain"
    raise HTTPException(422, "field must be loop, looplen, restlen or strain")


# ------------------------------------------------------------------ colour maps
# loop length: yellow (shortest) to dark blue (longest), a reversed-viridis style ramp. Stops: viridis yellow, light green, teal, blue-green, blue, dark blue.
LOOP_STOPS = np.array([(253, 231, 37), (122, 209, 81), (34, 168, 132), (42, 120, 142), (65, 68, 135), (28, 31, 102)], float)
STRAIN_NEUTRAL, STRAIN_RED, STRAIN_GREEN = np.array((235, 235, 235), float), np.array((200, 40, 40), float), np.array((30, 150, 70), float)

def ramp(stops, t):
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1); i = np.minimum(t.astype(int), len(stops) - 2); u = (t - i)[:, None]
    return stops[i] * (1 - u) + stops[i + 1] * u

LOOP_NOTES = {"looplen": "geometric arc length - does not show migration", "restlen": "rest length l0 of each loop - shows migration"}

def colour_values(field, vals, flip, clip):
    """(uint8 RGB per value, legend dict). Loop fields: the range is 1 +- h, h = max(clip-th percentile of |value - 1|, LOOP_HALF_MIN): yellow = shorter than the as-built mean, dark blue = longer,
    so a few outlier loops cannot compress everything else into one colour (the old min-to-max range did)."""
    if field in ("looplen", "restlen"):
        h = max(float(np.percentile(np.abs(vals - 1.0), clip)), LOOP_HALF_MIN); lo, hi = 1.0 - h, 1.0 + h
        t = (vals - lo) / (2 * h)
        lab = "loop yarn length (relaxed geometry)" if field == "looplen" else "loop rest length l0"
        return ramp(LOOP_STOPS, t).round().astype(np.uint8), dict(vmin=lo, vmax=hi, centre=1.0, half_width=h, min_half_width=LOOP_HALF_MIN, data_min=float(vals.min()), data_max=float(vals.max()), clip_percentile=clip,
                                                                  stops=LOOP_STOPS.astype(int).tolist(), label=lab + " / as-built mean", note=LOOP_NOTES[field], field=field, ticks=[lo, 1.0, hi], end_labels=["shorter", "longer"])
    c = float(np.percentile(np.abs(vals), clip)) or 1e-12
    t = np.clip(vals / c, -1, 1)[:, None]
    pos, neg = (STRAIN_GREEN, STRAIN_RED) if flip else (STRAIN_RED, STRAIN_GREEN)       # default: tension red, compression green
    rgb = np.where(t >= 0, STRAIN_NEUTRAL + (pos - STRAIN_NEUTRAL) * t, STRAIN_NEUTRAL + (neg - STRAIN_NEUTRAL) * (-t))
    return rgb.round().astype(np.uint8), dict(vmin=-c, vmax=c, tension=("green" if flip else "red"), compression=("red" if flip else "green"), neutral=STRAIN_NEUTRAL.astype(int).tolist(),
                                              stops=[neg.astype(int).tolist(), STRAIN_NEUTRAL.astype(int).tolist(), pos.astype(int).tolist()], end_labels=["compression", "tension"],
                                              label="axial strain of the relaxed fabric", ticks=[-c, 0.0, c], clip_percentile=clip)


# ------------------------------------------------------------------ vertex layout of the served geometry
def _served_mesh_file(rid, d, variant):
    rel = variant == "relaxed"
    if is_run(rid):
        base = d if (rel and (d / "relaxed_tube.bin").exists()) else cache_dir(rid)
        return base / ("relaxed_tube.bin" if rel else "fab_tube.bin")
    hits = sorted(h for h in d.glob("*_tube.bin") if h.name.startswith("relaxed_") == rel)
    if not hits: raise HTTPException(404, "no tube mesh yet; open the view first")
    return hits[0]

def tube_layout(curves, R, nv):
    """Source-vertex index (into the concatenated curves) of every vertex of a mesh written by build_knit.write_obj: per course n rings of `sides` vertices, then the two end-cap vertices.
    `sides` is solved from the vertex count."""
    offs = np.cumsum([0] + [len(c) for c in curves]); kept = []
    for c, o in zip(curves, offs):
        k = decim_idx(c, R); P = c[k]
        k = k[np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]]
        kept.append(k + o)
    rings = sum(len(k) for k in kept)
    if rings == 0 or (nv - 2 * len(curves)) % rings: raise HTTPException(409, "the mesh vertex count does not fit this axis (%d vertices, %d rings)" % (nv, rings))
    sides = (nv - 2 * len(curves)) // rings
    if sides < 3: raise HTTPException(409, "the mesh vertex count does not fit this axis")
    idx = np.concatenate([np.concatenate([np.repeat(k, sides), [k[0], k[-1]]]) for k in kept])
    return idx, sides

def _check_layout(V, idx, pts, R):
    """Sanity check: the vertices must lie one yarn radius from their source point (guards against a mesh that was built from another axis)."""
    n = len(idx); sel = np.linspace(0, n - 1, min(n, 400)).astype(int)
    dist = np.linalg.norm(V[sel] - pts[idx[sel]], axis=1)
    if not np.all(np.isfinite(dist)) or float(np.median(np.abs(dist - R))) > 0.25 * R: raise HTTPException(409, "the mesh does not match this axis (vertices are not one yarn radius from the yarn centre)")

def colour_bytes(rid, field, variant, kind, flip, clip):
    d = view_dir(rid); f = fields(d); vals, field = _vertex_values(f, field, variant)
    rgb, legend = colour_values(field, vals, flip, clip)
    curves = f.Xr if variant == "relaxed" else f.F
    pts = np.vstack(curves)
    if kind == "tube":
        mf = _served_mesh_file(rid, d, variant)
        if not mf.exists(): raise HTTPException(404, "no tube mesh yet; open the view first")
        with open(mf, "rb") as fh:
            head = fh.read(16); nv = int(np.frombuffer(head[8:12], "<u4")[0])
        idx, sides = tube_layout(curves, f.R, nv)
        V = np.memmap(mf, dtype="<f4", mode="r", offset=16, shape=(nv, 3))
        _check_layout(V, idx, pts, f.R)
        return rgb[idx].tobytes(), legend
    if kind == "axis":
        keeps, o = [], 0
        for c in curves: keeps.append(decim_idx(c, f.R, 1.0, 8.0) + o); o += len(c)
        return rgb[np.concatenate(keeps)].tobytes(), legend
    raise HTTPException(422, "kind must be tube or axis")


# ------------------------------------------------------------------ endpoints (read only)
def _hist(vals, bins=60):
    v = vals[np.isfinite(vals)]
    if not len(v): return dict(edges=[], counts=[])
    c, e = np.histogram(v, bins=bins)
    return dict(edges=e.tolist(), counts=c.tolist())

def _stats(v):
    v = v[np.isfinite(v)]
    return dict(n=int(len(v)), min=float(v.min()), max=float(v.max()), mean=float(v.mean()), median=float(np.median(v)), p02=float(np.percentile(v, 2)), p98=float(np.percentile(v, 98))) if len(v) else dict(n=0)

def _grid(a): return [[(None if not np.isfinite(x) else float(x)) for x in row] for row in a]

@router.get("/api/results/{rid}/info")
def info(rid: str):
    """A result-like record for a run, so the viewer list and info panel work unchanged."""
    d = view_dir(rid)
    m = _rjson(d / "meta.json") or {}; st = _rjson(d / "state.json") or {}; stat = _rjson(d / "status.json") or {}
    params = stat.get("params") or {}
    has_rel = (d / "relaxed_axis.csv").exists()
    name = "%s run %s" % tuple(RUN_ID_RE.match(rid).groups()) if is_run(rid) else m.get("name", rid)
    lines = ["Run viewer: read-only view of campaign run %s" % name, "Run parameters: %s" % json.dumps(params)] + list(m.get("info") or [])
    return dict(id=rid, created=m.get("created"), name=name, label=name, units=(m.get("structure") or {}).get("units", ""), R=m.get("R"), info=lines, settings=params, report="",
                matrix_text=("%d x %d stitches per unit" % (len(m["matrix"][0]), len(m["matrix"]))) if m.get("matrix") else "", n_sims=len(m.get("simulations") or []),
                sim=dict(state=st.get("state") or "none", has_relaxed=has_rel, relaxed_at=int((d / "relaxed_axis.csv").stat().st_mtime) if has_rel else 0),
                campaign=RUN_ID_RE.match(rid).group(1) if is_run(rid) else None, run=RUN_ID_RE.match(rid).group(2) if is_run(rid) else None)

@router.get("/api/results/{rid}/field-info")
def field_info(rid: str, field: str = "loop", variant: str = "baseline", flip: int = 0, clip: float = CLIP_PERCENTILE):
    d = view_dir(rid); f = fields(d); vals, fu = _vertex_values(f, field, variant)
    _, legend = colour_values(fu, vals, bool(flip), clip)
    notes = []
    if fu in ("looplen", "restlen"):
        notes.append(LOOP_NOTES[fu].capitalize() + ".")
        notes.append("Loop = one stitch of one course; identity: %s. Lengths are chord sums of the solver's resampled yarn, with the two segments at a loop boundary split in proportion; divided by the as-built mean over all %d loops "
                     "(%.4g%s)." % (f.identity, int((f.arc_asb > 0).sum()), f.mean_asb if fu == "looplen" else f.rest0_mean, "" if fu == "looplen" else ", as-built rest length = chord x %.5f" % f.ratio))
        notes.append("UNSOURCED display choice: colour range = 1 +- h, h = max(%g-th percentile of |value - 1|, %g); here h = %.4g, data range %.4f to %.4f." % (clip, LOOP_HALF_MIN, legend["half_width"], legend["data_min"], legend["data_max"]))
        if fu == "looplen" and variant == "relaxed":
            if f.strain is not None: notes.append("The geometric arc length follows the rest length only through l = l0 (1 + strain), so it can show migration larger than the elastic strain (here up to about %.2f %%) and nothing smaller." % (100 * float(np.abs(np.concatenate(f.strain)).max())))
            if f.rest_src != "saved": notes.append("This run has no relaxed_l0.csv (runs made before commit c325e07 did not save it). Its colouring cannot be used as evidence for or against loop-length migration.")
    else:
        notes += ["Axial strain of the relaxed fabric = segment length / rest length - 1. It is not stretch-test strain.", f.rest_note,
                  "Rest-length migration during the relaxation is %s." % ("included (saved rest lengths)" if f.rest_src == "saved" else "NOT included: after relaxation the rest lengths are not saved"),
                  "UNSOURCED display choice: colour range = +-%g-th percentile of |strain|." % clip]
    return dict(field=fu, requested=field, variant=variant, legend=legend, mean_loop_asbuilt=f.mean_asb, n_loops=int((f.arc_asb > 0).sum()), rest_source=f.rest_src, rest_exact=f.rest_exact, identity=f.identity, notes=[n for n in notes if n])

@router.get("/api/results/{rid}/colors")
def colors(rid: str, field: str = "loop", variant: str = "baseline", kind: str = "tube", flip: int = 0, clip: float = CLIP_PERCENTILE):
    if variant not in ("baseline", "relaxed") or not (50.0 <= clip <= 100.0): raise HTTPException(422, "bad variant or clip")
    data, legend = colour_bytes(rid, field, variant, kind, bool(flip), clip)
    return Response(data, media_type="application/octet-stream", headers={"Content-Encoding": "identity", "X-Count": str(len(data) // 3), "X-Legend": json.dumps(legend)})

@router.get("/api/results/{rid}/loops")
def loops(rid: str):
    d = view_dir(rid); f = fields(d)
    out = dict(ncourse=len(f.courses), ncol=f.ncol, mean_loop_asbuilt=f.mean_asb, units=f.units, asbuilt=_grid(np.where(f.arc_asb > 0, f.arc_asb, np.nan)),
               relaxed=_grid(np.where(f.arc_rel > 0, f.arc_rel, np.nan)) if f.arc_rel is not None else None,
               ratio_asbuilt=_grid(np.where(f.arc_asb > 0, f.arc_asb / f.mean_asb, np.nan)), ratio_relaxed=_grid(np.where(f.arc_rel > 0, f.arc_rel / f.mean_asb, np.nan)) if f.arc_rel is not None else None,
               rest_ratio=(_grid(np.where(f.rest_sum > 0, f.rest_sum / f.rest0_mean, np.nan)) if f.rest_sum is not None and f.rest_src == "saved" else None), rest_source=f.rest_src, identity=f.identity,
               note="Loop = one stitch of one course (%s). Lengths are chord sums of the solver's resampled yarn, the two segments at a loop boundary split in proportion." % f.identity)
    return out

@router.get("/api/results/{rid}/strain")
def strain(rid: str, step: int = 0):
    """Axial strain of the relaxed fabric. `step` is reserved for a later stretch-test state (0 = the relaxed fabric); other values are refused."""
    if step != 0: raise HTTPException(400, "stretch-test steps are not available: only the relaxed fabric (step 0)")
    d = view_dir(rid); f = fields(d)
    if f.strain is None: raise HTTPException(404, "no relaxed axis for this folder")
    allv = np.concatenate(f.strain)
    return dict(label="axial strain of the relaxed fabric", step=0, units="dimensionless", stats=_stats(allv), hist=_hist(allv, 80), ncourse=len(f.courses), ncol=f.ncol, grid=_grid(f.strain_grid),
                rest_source=f.rest_src, rest_exact=f.rest_exact, notes=[f.rest_note, "Rest-length migration during the relaxation is %s." % ("included" if f.rest_src == "saved" else "not included (rest lengths after relaxation are not saved)")])

@router.get("/api/results/{rid}/fold")
def fold(rid: str):
    """Height profile of the relaxed fabric along each direction: the cell-mean heights after removing a best-fit plane (campaigns/README.md, fold_amplitude_mm), averaged across the other direction."""
    d = view_dir(rid); f = fields(d)
    if f.Xr is None: raise HTTPException(404, "no relaxed axis for this folder")
    P = np.vstack(f.Xr); x, y, z = P[:, 0], P[:, 1], P[:, 2]; xc, yc = x - x.mean(), y - y.mean()
    A = np.c_[xc, yc, np.ones_like(xc)]; res = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    xe = np.linspace(x.min(), x.max(), f.ncol + 1); ye = np.linspace(y.min(), y.max(), f.nrow + 1)
    cnt, _, _ = np.histogram2d(y, x, bins=[ye, xe])
    if (cnt == 0).mean() > 0.10: raise HTTPException(409, "too many empty cells to form a height profile")
    s, _, _ = np.histogram2d(y, x, bins=[ye, xe], weights=res); g = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    px, py = np.nanmean(g, axis=0), np.nanmean(g, axis=1)
    cam = d.parent.parent if is_run(rid) else None
    spec = _rjson(cam / "spec.json") if cam else None; met = _rjson(d / "metrics.json") or {}
    return dict(units=f.units, x_mm=((xe[:-1] + xe[1:]) / 2).tolist(), y_mm=((ye[:-1] + ye[1:]) / 2).tolist(), profile_course=[None if not np.isfinite(v) else float(v) for v in px],
                profile_wale=[None if not np.isfinite(v) else float(v) for v in py], direction=met.get("fold_direction"), amplitude=met.get("fold_amplitude_mm"), local_amplitude=met.get("fold_local_amplitude_mm"),
                turn_x_rad=met.get("fold_turn_x_rad"), turn_y_rad=met.get("fold_turn_y_rad"), threshold_rad=((spec or {}).get("thresholds") or {}).get("fold_min_turn_rad"),
                threshold_note="The fold threshold is a classification choice written in the campaign's spec.json, not a physical constant (UNSOURCED).",
                note="profile_course = mean height along x (courses direction), profile_wale = mean height along y (wales direction); heights after removing a best-fit plane.")

@router.get("/api/results/{rid}/tube-status")
def tube_status(rid: str):
    view_dir(rid)                                         # unknown names and paths are refused (404)
    if not is_run(rid): return dict(stages=[])
    c = CAMPAIGNS / RUN_ID_RE.match(rid).group(1) / "cache" / RUN_ID_RE.match(rid).group(2)
    out = []
    for p in sorted(c.glob("progress_*.json")) if c.is_dir() else []:
        j = _rjson(p) or {}; j["file"] = p.stem.replace("progress_", ""); out.append(j)
    sizes = {p.name: p.stat().st_size for p in c.glob("*") if p.is_file() and not p.name.startswith("progress_")} if c.is_dir() else {}
    return dict(stages=out, cache_files=sizes)


# ------------------------------------------------------------------ pages (static files, re-read on every request like index.html)
from fastapi.responses import HTMLResponse

def _static(name, media):
    return Response((HERE / name).read_text(encoding="utf8"), media_type=media, headers={"Cache-Control": "no-cache"})

@router.get("/strain", response_class=HTMLResponse)
def page_strain(): return _static("strain.html", "text/html; charset=utf-8")

@router.get("/run", response_class=HTMLResponse)
def page_run(): return _static("run.html", "text/html; charset=utf-8")

@router.get("/strain.js")
def js_strain(): return _static("strain.js", "application/javascript")

@router.get("/run.js")
def js_run(): return _static("run.js", "application/javascript")

@router.get("/viewer_core.js")
def js_core(): return _static("viewer_core.js", "application/javascript")


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "gen":
        try: _gen(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
        except Exception as e:
            try: Path(sys.argv[6]).write_text(json.dumps(dict(stage="failed", finished=True, error=str(e))))
            except Exception: pass
            raise
    else: sys.exit("usage: runview.py gen <axis.csv> <out.bin> <R> <sides> <progress.json>")
