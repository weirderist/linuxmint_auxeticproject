#!/usr/bin/env python3
"""publish_result.py - build a fabric with build_knit.py and publish it to the web UI. Replaces new_result.py (which called the legacy catmull_axis.py).
Usage: python3 publish_result.py --matrix "1 1 2 2 1/2 1 1 2 2" --tiles 4 --ne 10 --cpi 14 --wpi 18 --name "my fabric"
Every argument is passed to build_knit.py unchanged, except --out, which is set here. The build goes into a hidden folder
(webui/results/.build-<n>, never listed by the UI) and is renamed to webui/results/<id>/ only when complete, where <id> is the id build_knit.py wrote
into meta.json. The rename is atomic, so the viewer never sees a half-written result. Prints the id.
"""
import json, os, re, shutil, subprocess, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
RES = BASE / "webui" / "results"; RES.mkdir(parents=True, exist_ok=True)
args = sys.argv[1:]
if any(a == "--out" or a.startswith("--out=") for a in args): sys.exit("error: --out is set by this script")
tmp = RES / (".build-%d" % os.getpid())
p = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "build_knit.py"), *args, "--out", str(tmp)])
if p.returncode != 0 or not (tmp / "meta.json").exists():
    shutil.rmtree(tmp, ignore_errors=True); sys.exit("build failed; nothing published")
rid = json.loads((tmp / "meta.json").read_text())["id"]
if not re.match(r"^[0-9A-Za-z_-]{1,64}$", rid):
    shutil.rmtree(tmp, ignore_errors=True); sys.exit("id %r is not usable as a folder name (letters, digits, - and _ only, max 64)" % rid)
if (RES / rid).exists():
    shutil.rmtree(tmp, ignore_errors=True); sys.exit("a result with id %s already exists" % rid)
os.rename(tmp, RES / rid)
print(rid)
