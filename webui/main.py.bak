import json, re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")

@app.get("/api/results")
def list_results():
    out = []
    for d in sorted((p for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith(".")),
                    key=lambda p: p.name, reverse=True):
        try:
            m = json.loads((d / "meta.json").read_text())
        except Exception:
            continue                      # no meta.json yet = still being written
        out.append({"id": d.name, "created": m.get("created"), "label": m.get("label", ""),
                    "settings": m.get("settings", {}), "report": m.get("report", "")})
    return out

def _file(rid, pattern, media):
    if not ID_RE.match(rid):
        raise HTTPException(404)
    hits = sorted((RESULTS / rid).glob(pattern))
    if not hits:
        raise HTTPException(404)
    return FileResponse(hits[0], media_type=media)

@app.get("/api/results/{rid}/tube")
def tube(rid: str):
    return _file(rid, "*_tube.obj", "text/plain")

@app.get("/api/results/{rid}/axis")
def axis(rid: str):
    return _file(rid, "*_axis.csv", "text/plain")

@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text()      # re-read each time: editing index.html needs no restart
