import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)          # repository root: the tests run from any folder
import re, json, sys, pathlib, esprima, ast
W = pathlib.Path(_os.path.join(ROOT_STR, "webui")); bad = 0
def parse(body, typ, label):
    global bad
    try:
        if typ == "importmap": json.loads(body)
        elif typ == "module": esprima.parseModule(body)
        else: esprima.parseScript(body)
        print("OK   ", label, "(%d chars)" % len(body))
    except Exception as e: bad += 1; print("ERROR", label, "->", str(e)[:200])
for f in ("index.html", "campaigns.html", "strain.html", "run.html"):
    t = (W / f).read_text(encoding="utf8")
    blocks = list(re.finditer(r"<script([^>]*)>(.*?)</script>", t, re.S | re.I))
    if not blocks: print("(no script block in %s)" % f)
    for n, m in enumerate(blocks, 1):
        a, body = m.group(1), m.group(2); typ = (re.search(r'type="([^"]*)"', a) or [0, "text/javascript"])[1]; src = re.search(r'src="([^"]*)"', a)
        label = "%s block #%d (%s, %s)" % (f, n, typ, "src=" + src[1] if src else "inline")
        if src:
            if src[1].startswith(("http", "//")): print("SKIP ", label, "remote"); continue
            body = (W / src[1].lstrip("/")).read_text(encoding="utf8")
        parse(body, typ, label)
for f, typ in (("campaigns.js", "script"), ("strain.js", "module"), ("viewer_core.js", "module"), ("run.js", "script")):
    parse((W / f).read_text(encoding="utf8"), typ, f + " (standalone, " + typ + ")")
for f in ("main.py", "campaign_api.py", "runview.py"): ast.parse((W / f).read_text(encoding="utf8")); print("OK    %s (python)" % f)
for f in ("relax_job.py", "relax_state.py", "loop_grade.py", "campaign.py"): ast.parse((W.parent / "scripts" / f).read_text(encoding="utf8")); print("OK    scripts/%s (python)" % f)
print("ALL OK" if not bad else "FAILURES")
sys.exit(bad)
