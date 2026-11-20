"""Creates the (git-ignored) fixtures the tests use under smoke_test/. Run from anywhere:  python tests/make_fixtures.py

1. smoke_test/campaigns/smoke1        a synthetic 6-run campaign (tests/make_smoke.py)
2. smoke_test/campaigns/fid_copy/runs/0020   a REAL single-jersey build (build_knit.py, matrix "1", 12 x 12 loops, Ne 12, CPI 22, WPI 16): needs numpy and scipy, about a minute
3. smoke_test/campaigns/fid_copy/runs/0001 and 0003   copies of two finished fidelity runs: they must be copied from the server; the commands are printed if they are missing
Nothing here touches campaigns/ or scripts/ (build_knit.py is run with its output and cell cache inside smoke_test/).
"""
import os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; SM = ROOT / "smoke_test"; FID = SM / "campaigns" / "fid_copy"
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

subprocess.run([sys.executable, str(ROOT / "tests" / "make_smoke.py")], check=True, env=env, stdout=subprocess.DEVNULL)
print("smoke1 campaign ready")

j = SM / "jersey_build12"
if not (j / "meta.json").exists():
    j.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_knit.py"), "--matrix", "1", "--tiles", "12", "--ne", "12", "--cpi", "22", "--wpi", "16", "--out", str(j), "--cache", str(SM / "cell_cache"),
                    "--no-tube", "--name", "jersey12", "--id", "jersey12"], check=True, env=env, stdout=subprocess.DEVNULL)
r20 = FID / "runs" / "0020"; r20.mkdir(parents=True, exist_ok=True)
for fn in ("fab_axis.csv", "fab_nodes.csv", "meta.json"): shutil.copy(j / fn, r20 / fn)
print("single-jersey run 0020 ready")

missing = [r for r in ("0001", "0003") if not (FID / "runs" / r / "relaxed_axis.csv").exists()]
if missing:
    print("\nCopy these finished fidelity runs from the server (read-only copies; run in a shell with ssh access):")
    files = "fab_axis.csv fab_nodes.csv meta.json metrics.json relaxed_axis.csv relaxed_tube.bin state.json status.json"
    print('  mkdir -p smoke_test/campaigns/fid_copy/runs/%s' % " smoke_test/campaigns/fid_copy/runs/".join(missing))
    for r in missing: print('  for f in %s; do scp server:auxetic_project/campaigns/fidelity/runs/%s/$f smoke_test/campaigns/fid_copy/runs/%s/; done' % (files, r, r))
    print('  for f in spec.json manifest.json; do scp server:auxetic_project/campaigns/fidelity/$f smoke_test/campaigns/fid_copy/; done')
    print("\nThen run  python tests/make_fixtures.py  once more (it is repeatable).")
    sys.exit(1)
print("all fixtures present")
