import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)          # repository root: the tests run from any folder
import sys, json, math, types, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, _os.path.join(ROOT_STR, "scripts")); sys.path.insert(0, _os.path.join(ROOT_STR, "webui"))
import relax_state as rs, runview as rv
base = _os.path.join(ROOT_STR, "smoke_test", "campaigns", "fid_copy", "runs", "0001")
f = rv.fields(base); p = f.meta["simulations"][-1]["params"]; mu, step = p["mu"], 0.2
pts = [(float(a), 0.01 * float(c)) for a, c in (t.split(":") for t in p["stretch_law"].split(","))]
rs.LAW = rs.build_law([a for a, _ in pts], [b for _, b in pts]); EA = rs.LAW["EA"]
print("EA from the stretch law = %.1f N; mu = %.2f; slip_step_max = %.2f; segment rest length ~ %.4f mm (R = %.4f mm)" % (EA, mu, step, np.mean(np.concatenate(f.l0)), f.R))
print("capstan factor exp(mu*phi): phi=0.5 -> %.2f, 1.0 -> %.2f, 1.5 -> %.2f, 2.0 -> %.2f, pi -> %.2f" % tuple(math.exp(mu * x) for x in (0.5, 1, 1.5, 2, math.pi)))
def mk(X, l0):
    ys = [types.SimpleNamespace(X=x.copy(), l0=l.copy()) for x, l in zip(X, l0)]; return types.SimpleNamespace(yarns=ys, R=f.R)
def pass_(sim, label):
    cons = rs.build_contacts(sim); n = tr = slack = 0; ds = []; phis = []; ratios = []
    for c in cons:
        for yi, i, phi in ((c.ya, c.ia, c.phi_a), (c.yb, c.ib, c.phi_b)):
            y = sim.yarns[yi]
            if i < 1 or i > len(y.l0) - 1: continue
            la, lb = np.linalg.norm(y.X[i] - y.X[i - 1]), np.linalg.norm(y.X[i + 1] - y.X[i]); n += 1; phis.append(phi)
            Ta, Tb = rs.tension(la, y.l0[i - 1], EA), rs.tension(lb, y.l0[i], EA); hi, lo = max(Ta, Tb), min(Ta, Tb)
            if lo <= 0 < hi: slack += 1
            if lo > 0: ratios.append(hi / lo)
            d = rs.slip_at_node(la, y.l0[i - 1], lb, y.l0[i], EA, mu, phi, step)
            if d: tr += 1; ds.append(abs(d))
    ds = np.array(ds) if ds else np.zeros(1); phis = np.array(phis)
    print("%s: %d contact nodes (%d contacts); wrap angle phi: median %.2f rad, p10 %.2f, p90 %.2f" % (label, n, len(cons), np.median(phis), np.percentile(phis, 10), np.percentile(phis, 90)))
    print("   nodes where the capstan rule triggers a transfer: %d (%.1f %%); of all nodes %d have one side slack (T<=0) and the other in tension (threshold 0: ANY tension triggers)" % (tr, 100 * tr / max(n, 1), slack))
    if ratios: print("   tension ratio T_hi/T_lo at nodes with both sides in tension: median %.3f, p99 %.3f, max %.2f" % (np.median(ratios), np.percentile(ratios, 99), max(ratios)))
    print("   transfer per node in ONE pass: max %.2e mm, median of the non-zero %.2e mm = %.3f %% of a segment; cap would be %.2e mm (so max is %.1f %% of the cap)" % (ds.max(), np.median(ds), 100 * np.median(ds) / np.mean(np.concatenate(f.l0)), step * np.mean(np.concatenate(f.l0)), 100 * ds.max() / (step * np.mean(np.concatenate(f.l0)))))
# (a) the as-built flat start: l0 = chord x ratio everywhere, positions as built -> uniform tension
pass_(mk(f.X0, [l * f.ratio for l in f.l_asb]), "AS-BUILT START of the fidelity fabric (uniform ratio %.4f => uniform strain %+.2f %%)" % (f.ratio, 100 * (1 / f.ratio - 1)))
# (b) the single jersey as-built start (real build, run 0020)
J = rv.fields(_os.path.join(ROOT_STR, "smoke_test", "campaigns", "fid_copy", "runs", "0020")); fj = f
jx = mk(J.X0, [l * f.ratio for l in J.l_asb]); pass_(jx, "AS-BUILT START of the SINGLE JERSEY (12x12), same ratio")
# (c) the saved relaxed state of the old run 0001 (rest lengths reconstructed, migration not included): how big is the imbalance the rule would act on?
pass_(mk(f.Xr, f.l0), "RELAXED state of old run 0001 (pattern 1, did not settle; rest lengths reconstructed)")
# direct response of the rule to a given strain imbalance at one node (phi = 1.5, EA from the law)
print("\nOne node, two segments of rest length 0.1245 mm, phi = 1.5 rad (factor %.2f):" % math.exp(mu * 1.5))
for ea_, eb_ in ((0.005, 0.0), (0.005, -0.002), (0.01, 0.0), (0.02, 0.0), (0.05, 0.0), (0.0013, 0.0013), (0.0013, 0.0)):
    l0 = 0.1245; d = rs.slip_at_node(l0 * (1 + ea_), l0, l0 * (1 + eb_), l0, EA, mu, 1.5, step)
    print("   strains %+.2f %% | %+.2f %% -> transfer %.3e mm (%.3f %% of l0; equalising both would need %.3e mm)" % (100 * ea_, 100 * eb_, d, 100 * abs(d) / l0, l0 * abs(ea_ - eb_) / 2))
