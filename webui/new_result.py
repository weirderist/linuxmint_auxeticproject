#!/usr/bin/env python3
"""Create one timestamped result folder by running catmull_axis.py (stdlib only).

Folder: ~/auxetic_project/webui/results/<YYYYMMDD-HHMMSS>/
holding fab_axis.csv, fab_tube.obj, meta.json.

meta.json is written last, so the web page never lists a half-finished result.
The Telegram bot's future `simulate` command can call this script with the
user's settings.
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path.home() / "auxetic_project"

ap = argparse.ArgumentParser()
ap.add_argument("--cpi", type=float, default=14.0)
ap.add_argument("--wpi", type=float, default=18.0)
ap.add_argument("--l", type=float, default=3.2)
ap.add_argument("--d", type=float, default=0.2869)
ap.add_argument("--zb", type=float, default=0.0)
ap.add_argument("--matrix", default="FFFFBBBB,FFFFBBBB,BBBBFFFF,BBBBFFFF")
ap.add_argument("--label", default="")
a = ap.parse_args()

now = datetime.now().astimezone()
rid = now.strftime("%Y%m%d-%H%M%S")
d = BASE / "webui" / "results" / rid
d.mkdir(parents=True, exist_ok=True)

cmd = [
    "/usr/bin/python3", str(BASE / "scripts" / "catmull_axis.py"),
    "--cpi", str(a.cpi),
    "--wpi", str(a.wpi),
    "--l", str(a.l),
    "--d", str(a.d),
    "--zb", str(a.zb),
    "--matrix", a.matrix,
    "--out", str(d / "fab"),
]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
if p.returncode != 0:
    print(p.stdout + p.stderr)
    shutil.rmtree(d, ignore_errors=True)
    sys.exit(1)

meta = {
    "created": now.isoformat(timespec="seconds"),
    "label": a.label,
    "report": p.stdout,
    "settings": {
        "cpi": a.cpi,
        "wpi": a.wpi,
        "l": a.l,
        "d": a.d,
        "zb": a.zb,
        "matrix": a.matrix,
    },
}
(d / "meta.json").write_text(json.dumps(meta, indent=1))
print(rid)