"""Writes tests/golden_loop_grade.json: SHA-256 fingerprints of the starting rest lengths that scripts/yarn_model/loop_grade.py produces for fixed fixtures and settings.
Run ONCE before a refactor of loop_grade.py (python tests/make_golden_loop_grade.py); tests/test_slip_ext.py then proves the refactor changed no number.
Needs the fixtures of tests/make_fixtures.py. Regenerating it on purpose is a statement that the field itself was meant to change."""
import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)
import hashlib, json, shutil, sys
from pathlib import Path
P = Path(ROOT_STR); SM = P / "smoke_test"; FID = SM / "campaigns" / "fid_copy" / "runs"
sys.path.insert(0, str(P / "scripts")); import scriptdirs
import numpy as np
import loop_grade as lg, relax_state as rs

CASES = [("0020", 0.02, "both", 0.2), ("0001", 0.02, "both", 0.2), ("0001", 0.02, "boundary", 0.2), ("0001", 0.02, "tm", 0.2), ("0003", 0.02, "both", 0.2), ("0001", 0.05, "both", 0.5)]

def fingerprint(src, a, target, s_min):
    d = SM / "grade_tmp" / ("golden_%s_%s_%s" % (src, target, a)); d.mkdir(parents=True, exist_ok=True)
    for fn in ("fab_axis.csv", "meta.json"): shutil.copy(FID / src / fn, d / fn)
    meta = json.loads((d / "meta.json").read_text()); sim = rs.load(str(d), rs.Params(rest_length_ratio=0.9488))
    rec = lg.apply_start_grade(sim, str(d), meta, a, target, s_min)
    h = hashlib.sha256()
    for y in sim.yarns: h.update(np.ascontiguousarray(y.l0, dtype="<f8").tobytes())
    return h.hexdigest(), rec["factor_min"], rec["factor_max"]

if __name__ == "__main__":
    out = {}
    for src, a, target, s_min in CASES:
        h, fmin, fmax = fingerprint(src, a, target, s_min); out["%s|%s|%s|%s" % (src, a, target, s_min)] = dict(sha256=h, factor_min=fmin, factor_max=fmax)
        print(src, a, target, s_min, h[:16], fmin, fmax)
    (P / "tests" / "golden_loop_grade.json").write_text(json.dumps(out, indent=1))
    print("wrote tests/golden_loop_grade.json")
