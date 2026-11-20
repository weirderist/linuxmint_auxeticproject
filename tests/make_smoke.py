import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)          # repository root: the tests run from any folder
"""Build a synthetic throw-away campaign under smoke_test/ (never touches campaigns/)."""
import json, os, sys, hashlib
P = _os.path.join(ROOT_STR)
SM = os.path.join(P, "smoke_test", "campaigns"); os.environ["CAMPAIGN_DIR"] = SM
os.environ["CAMPAIGN_ROOT"] = os.path.join(P, "smoke_test"); os.environ["CAMPAIGN_SCRIPTS"] = os.path.join(P, "scripts")
sys.path.insert(0, os.path.join(P, "scripts"))
import campaign as cm
name = "smoke1"; d = os.path.join(SM, name); os.makedirs(d, exist_ok=True)
spec = json.load(open(os.path.join(P, "campaigns", "selftest", "spec.json"))); spec["name"] = name
spec["design"]["n_runs"] = 6
json.dump(spec, open(os.path.join(d, "spec.json"), "w"), indent=1)
pts = cm.design_points(spec)
runs = [dict(run="%04d" % (i + 1), params=p) for i, p in enumerate(pts)]
json.dump(dict(name=name, created=cm.now(), spec_sha1=cm.spec_hash(name), n_runs=6, runs=runs), open(os.path.join(d, "manifest.json"), "w"), indent=1)
M = {  # run -> (status, exit, wall, metrics)
 "0001": ("done", 0, 100.0, dict(relax_state="settled", rounds_used=5, rel_dE_last=-1e-7, min_dist_over_2R_end=1.0, fold_turn_x_rad=0.5, fold_turn_y_rad=0.1, fold_direction="course", fold_amplitude_mm=1.2, fold_local_amplitude_mm=0.3)),
 "0002": ("done", 2, 120.0, dict(relax_state="failed", rounds_used=40, rel_dE_last=-0.01, min_dist_over_2R_end=0.99, fold_turn_x_rad=0.02, fold_turn_y_rad=0.01, fold_direction="none", fold_amplitude_mm=0.05, fold_local_amplitude_mm=0.01)),
 "0003": ("done", 2, 140.0, dict(relax_state="failed", rounds_used=40, rel_dE_last=-0.02, min_dist_over_2R_end=0.91, fold_turn_x_rad=0.03, fold_turn_y_rad=0.02, fold_direction="none", fold_amplitude_mm=0.07, fold_local_amplitude_mm=0.02)),
 "0004": ("failed", 137, 30.0, {}), "0005": ("running", None, None, {}), "0006": ("queued", None, None, {})}
for rid, (stt, ex, wall, met) in M.items():
    rd = os.path.join(d, "runs", rid); os.makedirs(rd, exist_ok=True)
    st = dict(run=rid, status=stt, params=runs[int(rid) - 1]["params"], stage="relax" if stt == "running" else "", attempts=[], interrupted_count=1 if rid == "0002" else 0, exit_code=ex, wall_s=wall,
              reason="timeout after 301 s (limit 300 s)" if stt == "failed" else "", history=[dict(t=cm.now(), event="queued")])
    if stt == "running": st["attempts"] = [dict(n=1, pid=4242, boot_id="", start_ticks="1", t_start=0, cmd=["python", "relax_job.py", "--folder", rd])]
    json.dump(st, open(os.path.join(rd, "status.json"), "w"), indent=1)
    if met:
        met.update(exit_code=ex, wall_s=wall); json.dump(met, open(os.path.join(rd, "metrics.json"), "w"))
    if stt != "queued": open(os.path.join(rd, "relax.log"), "w").write("\n".join("log line %d" % i for i in range(100)) + "\n")
cm.write_results(name)
print(open(os.path.join(d, "results.csv")).read()[:600])
