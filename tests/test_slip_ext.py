"""Tests for the opt-in slip extensions (scripts/yarn_model/slip_ext.py), the per-round diagnostics (scripts/yarn_model/round_diag.py), the tightness record and the wiring in relax_job.py / relax_state.py,
WITHOUT the solver and without JAX. Run from the repository root:  python tests/test_slip_ext.py   (needs the fixtures of tests/make_fixtures.py)."""
import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)
import ast, csv, json, math, shutil, sys
from pathlib import Path
P = Path(ROOT_STR); SM = P / "smoke_test"; FID = SM / "campaigns" / "fid_copy" / "runs"
import os
os.environ["CAMPAIGN_DIR"] = str(SM / "campaigns"); os.environ["CAMPAIGN_SCRIPTS"] = str(P / "scripts")
for r in ("0001", "0020"):
    if not (FID / r).exists(): sys.exit("fixtures missing: run  python tests/make_fixtures.py  (see tests/README.md)")
sys.path.insert(0, str(P / "scripts")); import scriptdirs; sys.path.insert(0, str(P / "tests"))
import numpy as np
import loop_grade as lg, relax_state as rs, slip_ext as se, round_diag as rd, campaign as cm
import make_golden_loop_grade as golden
ok = True
def check(label, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + label, "" if cond else extra)

EA, MU, STEP, RATIO = 80.0, 0.49, 0.2, 0.9488
def scratch(name, src):
    d = SM / "grade_tmp" / name; d.mkdir(parents=True, exist_ok=True)
    for fn in ("fab_axis.csv", "meta.json"): shutil.copy(FID / src / fn, d / fn)
    return d, json.loads((d / "meta.json").read_text())
def make_sim(name, src, outer=3, ratio=RATIO):
    d, meta = scratch(name, src); p = rs.Params(mu=MU, EA=EA, EI=1.0, GJ=0.0, k_contact=18329.7, outer_max=outer, rest_length_ratio=ratio, slip_step_max=STEP)
    sim = rs.load(str(d), p); return sim, d, meta
def copy_sim(sim):
    import copy
    s = copy.copy(sim); s.yarns = [rs.Yarn(y.course, y.X.copy(), y.l0.copy(), y.theta.copy(), y.pinned.copy(), y.s) for y in sim.yarns]; s.contacts = sim.contacts; return s

# ---------------------------------------------------------------- 1. the loop-grade refactor changed no number
gold = json.loads((P / "tests" / "golden_loop_grade.json").read_text()); same = True
for k, v in gold.items():
    src, a, target, s_min = k.split("|")
    if not (FID / src).exists(): continue
    h, fmin, fmax = golden.fingerprint(src, float(a), target, float(s_min)); same &= (h == v["sha256"] and fmin == v["factor_min"] and fmax == v["factor_max"])
check("loop_grade.py after the refactor gives bit-identical starting rest lengths (SHA-256 of l0 for %d fixed cases)" % len(gold), same)

# ---------------------------------------------------------------- 2. slip_at_node_ext
rng = np.random.default_rng(1); worst = 0; n_equal = 0; n_nz = 0
for _ in range(2500):
    l0a, l0b = 0.1237*rng.uniform(0.8, 1.2, 2); la, lb = l0a*(1 + rng.uniform(-0.02, 0.08)), l0b*(1 + rng.uniform(-0.02, 0.08)); phi = rng.uniform(0, math.pi)
    d0 = rs.slip_at_node(la, l0a, lb, l0b, EA, MU, phi, STEP); d1 = se.slip_at_node_ext(la, l0a, lb, l0b, EA, MU, phi, STEP, rs.tension)
    n_equal += d0 == d1; n_nz += d0 != 0
check("slip_at_node_ext with no offsets and no extra limit returns EXACTLY what relax_state.slip_at_node returns (2500 random nodes, %d of them slipping)" % n_nz, n_equal == 2500 and n_nz > 200, (n_equal, n_nz))
la, l0a, lb, l0b, phi = 0.1237*1.005, 0.1237, 0.1237, 0.1237, 1.5
d_base = se.slip_at_node_ext(la, l0a, lb, l0b, EA, MU, phi, STEP, rs.tension); ds = [se.slip_at_node_ext(la, l0a, lb, l0b, EA, MU, phi, STEP, rs.tension, extra=e) for e in (0.0, 0.1, 0.3, 0.6, 5.0)]
check("a larger load-dependent limit (extra = mu_n F_n) moves less rest length, and a large one holds the contact: %s" % ["%.2e" % x for x in ds], all(a >= b - 1e-18 for a, b in zip(ds, ds[1:])) and ds[0] > ds[2] > 0 and ds[-1] == 0.0)
Ta = rs.tension(la, l0a + ds[2], EA); Tb = rs.tension(lb, l0b - ds[2], EA)
check("after a transfer with an extra limit the contact sits exactly on the new limit: T_hi = max(T_lo,0) exp(mu phi) + extra", abs(Ta - (max(Tb, 0)*math.exp(MU*phi) + 0.3)) < 1e-7, (Ta, Tb))
du = se.slip_at_node_ext(0.1237*1.003, 0.1237, 0.1237*1.003, 0.1237, EA, MU, phi, STEP, rs.tension)
dd = se.slip_at_node_ext(0.1237*1.003, 0.1237, 0.1237*1.003, 0.1237, EA, MU, phi, STEP, rs.tension, off_a=-1.0, off_b=1.0); dp = se.slip_at_node_ext(0.1237*1.003, 0.1237, 0.1237*1.003, 0.1237, EA, MU, phi, STEP, rs.tension, off_a=1.0, off_b=-1.0)
check("equal strains: no transfer without a drive; a drive offset moves rest length TOWARD the segment that looks more tense, and the opposite offset the other way (a +%.2e, b %.2e)" % (dd, dp), du == 0.0 and dd < 0 < dp)

# ---------------------------------------------------------------- 3. SlipExt: strength, offsets, normal force
G = [np.array([0.0, 0.0, 1.0, 1.0])]
x = se.SlipExt(mu_n=0.0, drive=0.02, decay_rounds=0.0, Gseg=G, kc=1000.0, R=0.13)
check("drive strength without decay is constant", all(x.strength(k) == 0.02 for k in (0, 1, 10, 500)))
xd = se.SlipExt(drive=0.02, decay_rounds=10.0, Gseg=G)
check("drive strength with a decay N = 10 rounds falls as exp(-round / N)", all(abs(xd.strength(k) - 0.02*math.exp(-k/10)) < 1e-15 for k in (0, 1, 7, 30)) and xd.strength(0) == 0.02)
oa, ob = x.offsets(0, 2, EA, 0)
check("offsets at the node between a segment in a loop with G = 0 and one with G = 1: -0.5 e EA and +0.5 e EA (toward the higher G); none inside a loop; reversed for a negative drive",
      abs(oa + 0.5*0.02*EA) < 1e-15 and abs(ob - 0.5*0.02*EA) < 1e-15 and x.offsets(0, 1, EA, 0) == (0.0, 0.0) and se.SlipExt(drive=-0.02, Gseg=G).offsets(0, 2, EA, 0) == (-oa, -ob))
check("F_n = kc x overlap: 0 for a contact at or beyond 2R, kc x (2R - dist) inside", x.normal_force(0.3) == 0.0 and abs(x.normal_force(0.25) - 1000.0*0.01) < 1e-9)
for bad, kw in (("negative mu_n", dict(mu_n=-1)), ("negative decay", dict(decay_rounds=-1)), ("drive without a field", dict(drive=0.01))):
    try: se.SlipExt(**kw); raised = False
    except ValueError: raised = True
    check("SlipExt refuses %s" % bad, raised)

# ---------------------------------------------------------------- 4. relax_state.migrate with and without the hooks, on a real relaxed state
simA, dA, metaA = make_sim("mig_relaxed", "0001"); R = metaA["R"]
Ax = np.loadtxt(FID / "0001" / "relaxed_axis.csv", delimiter=",", skiprows=1)
for y in simA.yarns: y.X = Ax[Ax[:, 0] == y.course][:, 1:].copy()                     # the saved relaxed positions; l0 stays the uniform start (chord of the as-built x ratio)
simA.contacts = rs.build_contacts(simA)
def old_migrate(sim):             # the function exactly as it was before the hooks were added
    mx = 0.0; p = sim.params
    for c in sim.contacts:
        for yi, i, phi in ((c.ya, c.ia, c.phi_a), (c.yb, c.ib, c.phi_b)):
            y = sim.yarns[yi]
            if i < 1 or i > len(y.l0)-1: continue
            la, lb = np.linalg.norm(y.X[i]-y.X[i-1]), np.linalg.norm(y.X[i+1]-y.X[i])
            d = rs.slip_at_node(la, y.l0[i-1], lb, y.l0[i], p.EA, p.mu, phi, p.slip_step_max)
            if d: y.l0[i-1] += d; y.l0[i] -= d; mx = max(mx, abs(d))
    return mx
s_old, s_new = copy_sim(simA), copy_sim(simA); m_old = old_migrate(s_old); m_new = rs.migrate(s_new)
check("relax_state.migrate with no extension and no statistics equals the old function exactly (same l0 on every segment of every course, same return value)", m_old == m_new and all(np.array_equal(a.l0, b.l0) for a, b in zip(s_old.yarns, s_new.yarns)) and m_old > 0)
s_st = copy_sim(simA); st = dict(n=0, total=0.0, max=0.0); calls = []; orig = rs.slip_at_node
rs.slip_at_node = lambda *a, **k: calls.append(orig(*a, **k)) or calls[-1]
rs.migrate(s_st, st); rs.slip_at_node = orig
check("the slip statistics are exact: number of transfers %d, sum %.3e, largest %.3e equal the values returned by the slip rule" % (st["n"], st["total"], st["max"]), st["n"] == sum(1 for d in calls if d) and abs(st["total"] - sum(abs(d) for d in calls)) < 1e-15 and st["max"] == max(abs(d) for d in calls), st)
s_z = copy_sim(simA); s_z.slip_ext = se.SlipExt(); rs.migrate(s_z)
check("with an extension object that has every constant 0 the result is identical to no extension", all(np.array_equal(a.l0, b.l0) for a, b in zip(s_old.yarns, s_z.yarns)))
s_m = copy_sim(simA); s_m.slip_ext = se.SlipExt(mu_n=1.0, kc=18329.7, R=R); st_m = dict(n=0, total=0.0, max=0.0); rs.migrate(s_m, st_m)
s_h = copy_sim(simA); s_h.slip_ext = se.SlipExt(mu_n=1e6, kc=18329.7, R=R); st_h = dict(n=0, total=0.0, max=0.0); rs.migrate(s_h, st_h)
check("mu_n > 0 on the same saved state: fewer / smaller transfers (%d -> %d slips, total %.2e -> %.2e)" % (st["n"], st_m["n"], st["total"], st_m["total"]), st_m["n"] <= st["n"] and st_m["total"] < st["total"])
loaded = [c for c in simA.contacts if c.dist < 2 * R]; unloaded = [c for c in simA.contacts if c.dist >= 2 * R]
def run_subset(contacts, ext):
    s = copy_sim(simA); s.contacts = contacts; s.slip_ext = ext; stt = dict(n=0, total=0.0, max=0.0); rs.migrate(s, stt); return stt, s
s_l, _ = run_subset(loaded, se.SlipExt(mu_n=1e6, kc=18329.7, R=R)); s_l0, _ = run_subset(loaded, None); s_u1, _ = run_subset(unloaded, se.SlipExt(mu_n=1e6, kc=18329.7, R=R)); s_u0, _ = run_subset(unloaded, None)
check("a huge mu_n holds EVERY loaded contact (overlap > 0: %d contacts, %d slips without mu_n, %d with) and does not touch unloaded contacts (F_n = 0: %d contacts, %d slips both ways)" % (len(loaded), s_l0["n"], s_l["n"], len(unloaded), s_u1["n"]),
      s_l0["n"] > 0 and s_l["n"] == 0 and s_u0["n"] == s_u1["n"] and s_u0["total"] == s_u1["total"])
check("every slip pass conserves the rest length of every yarn (transfers are between neighbouring segments of one yarn)", all(abs(a.l0.sum() - b.l0.sum()) < 1e-12 for a, b in zip(s_new.yarns, simA.yarns)) and all(abs(a.l0.sum() - b.l0.sum()) < 1e-12 for a, b in zip(s_m.yarns, simA.yarns)))

# ---------------------------------------------------------------- 5. the persistent drive
fA = lg.segment_field(simA, str(dA), metaA, "both", 0.2); GsA = lg.drive_field(fA); mG = float(np.mean(np.concatenate(GsA)))
def gain_of(sim_after, sim_before, Gs): return sum(float(((a.l0 - b.l0)*(g - np.mean(np.concatenate(Gs)))).sum()) for a, b, g in zip(sim_after.yarns, sim_before.yarns, Gs))
s_dr = copy_sim(simA); s_dr.slip_ext = se.SlipExt(drive=0.01, Gseg=GsA, kc=18329.7, R=R); st_dr = dict(n=0, total=0.0, max=0.0); rs.migrate(s_dr, st_dr)
s_dn = copy_sim(simA); s_dn.slip_ext = se.SlipExt(drive=-0.01, Gseg=GsA, kc=18329.7, R=R); rs.migrate(s_dn)
check("on the saved relaxed state (strains a few tenths of a percent) the drive e = 0.01 changes the transfers (%d -> %d slips, total %.2e -> %.2e) and rest length moves toward the higher-G loops (gain %.2e > baseline %.2e); a negative drive moves it the other way (%.2e)" % (st["n"], st_dr["n"], st["total"], st_dr["total"], gain_of(s_dr, simA, GsA), gain_of(s_new, simA, GsA), gain_of(s_dn, simA, GsA)),
      gain_of(s_dr, simA, GsA) > gain_of(s_new, simA, GsA) and gain_of(s_dn, simA, GsA) < gain_of(s_new, simA, GsA) and st_dr["total"] != st["total"])
# uniform START: friction decides. With the campaign's pre-strain (+5.4 %) the lower-tension side carries 4.3 N, so exp(mu phi) T_lo (about 1.9 N more) exceeds the drive offset (0.4 N): nothing moves.
simJ, dJ, metaJ = make_sim("drive_jersey", "0020", outer=4, ratio=1.0); rs.release(simJ)                 # ratio 1: strain-free uniform start, T = 0 on both sides
simP, dP, metaP = make_sim("drive_jersey_pre", "0020", outer=4); rs.release(simP)                           # ratio 0.9488: uniform +5.4 % pre-strain
fldJ = lg.segment_field(simJ, str(dJ), metaJ, "both", 0.2); Gs = lg.drive_field(fldJ); fldP = lg.segment_field(simP, str(dP), metaP, "both", 0.2)
for lab_, sm, ff in (("strain-free", simJ, fldJ), ("pre-strained +5.4 %", simP, fldP)):
    s0 = copy_sim(sm); st0 = dict(n=0, total=0.0, max=0.0); rs.migrate(s0, st0)
    check("uniform %s start, no drive: the slip rule does nothing (0 transfers at %d contacts), as the probe found" % (lab_, len(sm.contacts)), st0["n"] == 0 and all(np.array_equal(a.l0, b.l0) for a, b in zip(s0.yarns, sm.yarns)))
sj1 = copy_sim(simJ); sj1.slip_ext = se.SlipExt(drive=0.01, Gseg=Gs, kc=18329.7, R=metaJ["R"]); st1 = dict(n=0, total=0.0, max=0.0); rs.migrate(sj1, st1)
dl = [a.l0 - b.l0 for a, b in zip(sj1.yarns, simJ.yarns)]; gain = gain_of(sj1, simJ, Gs)
check("strain-free uniform start WITH the drive (e = 0.01): the slip rule now acts (%d transfers, total %.2e) and rest length moves toward the higher-G loops (G-weighted gain %.2e > 0)" % (st1["n"], st1["total"], gain), st1["n"] > 0 and gain > 0)
inner = [abs(d[j]) for d, g in zip(dl, Gs) for j in range(1, len(d) - 1) if g[j - 1] == g[j] == g[j + 1]]; moved = [abs(d[j]) for d, g in zip(dl, Gs) for j in range(1, len(d) - 1) if not (g[j - 1] == g[j] == g[j + 1])]
check("...only where the field changes: %d segments whose neighbouring nodes have equal G (a plateau of the field) did not change at all (largest %.1e); %d segments on a slope did (largest %.1e, %d of them changed)" % (len(inner), max(inner or [0]), len(moved), max(moved), sum(1 for m in moved if m > 0)), max(inner or [0]) == 0.0 and sum(1 for m in moved if m > 0) > 100)
touched = sum(1 for d in dl for v in d if v != 0); contacts_total = len(simJ.contacts)
check("the smooth drive reaches many contacts: %d segments changed in one pass at %d contacts (the piecewise field of the grade reached almost none)" % (touched, contacts_total), touched > 200)
sp = copy_sim(simP); sp.slip_ext = se.SlipExt(drive=0.01, Gseg=lg.drive_field(fldP), kc=18329.7, R=metaP["R"]); stp = dict(n=0, total=0.0, max=0.0); rs.migrate(sp, stp)
sp2 = copy_sim(simP); sp2.slip_ext = se.SlipExt(drive=0.1, Gseg=lg.drive_field(fldP), kc=18329.7, R=metaP["R"]); stp2 = dict(n=0, total=0.0, max=0.0); rs.migrate(sp2, stp2)
check("pre-strained uniform start: a drive of e = 0.01 stays inside the friction limit (%d transfers) but the largest allowed e = 0.1 exceeds it (%d transfers): the drive has to beat exp(mu phi) T_lo, so it acts only where the strain has relaxed" % (stp["n"], stp2["n"]), stp["n"] == 0 and stp2["n"] > 0)
neg = copy_sim(simJ); neg.slip_ext = se.SlipExt(drive=-0.01, Gseg=Gs, kc=18329.7, R=metaJ["R"]); rs.migrate(neg)
check("a negative drive moves rest length the other way (G-weighted gain %.2e < 0)" % gain_of(neg, simJ, Gs), gain_of(neg, simJ, Gs) < 0)
# whole outer loop with a no-op relaxation: run() sets sim.round_index, passes the statistics and applies the decaying drive
def run_rounds(decay, rounds=4):
    s = copy_sim(simJ); s.params = rs.Params(mu=MU, EA=EA, EI=1.0, GJ=0.0, k_contact=18329.7, outer_max=rounds, rest_length_ratio=1.0, slip_step_max=STEP, tol_len=0.0); s.folder = str(dJ)          # tol_len 0: no early settle in this test
    s.slip_ext = se.SlipExt(drive=0.02, decay_rounds=decay, Gseg=Gs, kc=18329.7, R=metaJ["R"]); s.slip_stats = None; out = []
    s.state = rs.S.RELEASED
    def relax_fn(sim): sim.slip_stats = dict(n=0, total=0.0, max=0.0)
    rs.run(s, relax_fn, lambda sim: 1.0, lambda k, E, dmax: out.append((k, dict(s.slip_stats), s.slip_ext.strength(k), s.round_index)))
    return out, s
(r0, s0_end), (r1, s1_end) = run_rounds(0.0), run_rounds(2.0)
check("rs.run sets sim.round_index every round and hands the statistics to migrate (round_index %s, transfers per round %s)" % ([r[3] for r in r0], [r[1]["n"] for r in r0]), [r[3] for r in r0] == [0, 1, 2, 3] and r0[0][1]["n"] > 0 and all(r[1]["total"] >= 0 for r in r0))
g0, g1 = gain_of(s0_end, simJ, Gs), gain_of(s1_end, simJ, Gs)
check("with decay N = 2 the drive strength of rounds 0..3 is e exp(-k/2) (%s); with the positions held fixed the elastic back-stress then pushes rest length BACK (recovery: later transfers %s vs %s without decay), so the net movement toward higher G after 4 rounds is smaller than with a constant drive (%.2e < %.2e)" % ([round(r[2], 5) for r in r1], ["%.1e" % r[1]["total"] for r in r1[1:]], ["%.1e" % r[1]["total"] for r in r0[1:]], g1, g0),
      all(abs(r[2] - 0.02*math.exp(-r[0]/2)) < 1e-15 for r in r1) and 0 < g1 < g0)

# ---------------------------------------------------------------- 6. diagnostics
sD, dD, mD = make_sim("diag_jersey", "0020"); diag = rd.RoundDiag(sD, str(dD), mD)               # start state = the uniform start
rec = lg.apply_start_grade(sD, str(dD), mD, 0.02, "both", 0.2); row = diag.row(sD, 0, 1.0, 0.0, dict(n=3, total=0.5, max=0.4), 0.01)
check("diagnostics columns: one value for every documented column, in the documented order (%d columns)" % len(rd.COLUMNS), set(row) == set(rd.COLUMNS) and rd.COLUMNS[:3] == ["round", "E", "max_dl0"] and row["round"] == 1)
check("regions on a graded single jersey: the outer ring has rest length ABOVE its start (mean %.4f), the inner core BELOW (%.4f), all loops about 1 (%.4f)" % (row["rest_outer_mean"], row["rest_inner_mean"], row["rest_all_mean"]),
      row["rest_outer_mean"] > 1.004 and row["rest_inner_mean"] < 0.995 and abs(row["rest_all_mean"] - 1) < 2e-3 and row["rest_outer_min"] <= row["rest_outer_mean"] <= row["rest_outer_max"])
check("the geometric length of the loops has not changed (positions unchanged): geo_* = 1; the grade conserves the total rest length (net_l0_change %.1e)" % row["net_l0_change"], all(abs(row["geo_%s_%s" % (r, s)] - 1) < 1e-12 for r in rd.REGIONS for s in ("mean", "min", "max")) and abs(row["net_l0_change"]) < 1e-12)
check("share of the length change stored in l0 is between 0 and 1 (%.3f; the rest-length change is %s of the elastic strain at this uniform pre-strain), slip statistics and drive strength are written through" % (row["share_l0"], "small" if row["share_l0"] < 0.5 else "large"),
      0 <= row["share_l0"] <= 1 and row["n_slips"] == 3 and row["transferred_total"] == 0.5 and row["drive_strength"] == 0.01 and abs(row["mean_elastic_strain"] - (1/RATIO - 1)) < 5e-3)
sD2, dD2, mD2 = make_sim("diag_jersey2", "0020"); diag2 = rd.RoundDiag(sD2, str(dD2), mD2); r_start = diag2.row(sD2, 0, 1.0, 0.0)
check("at the start of a run share_l0 is 0 and every rest ratio is exactly 1 (nothing has moved yet)", r_start["share_l0"] == 0.0 and r_start["rest_all_mean"] == 1.0 and r_start["rest_outer_max"] == 1.0)
flat = [np.c_[np.linspace(0, 10, 50), np.full(50, y), np.zeros(50)] for y in np.linspace(0, 5, 12)]
bump = [X + np.c_[np.zeros(50), np.zeros(50), 0.4*np.exp(-((X[:, 0] - 5)**2 + (X[:, 1] - 2.5)**2))] for X in flat]
check("fold descriptor: 0 for a flat sheet, > 0 for a bumped sheet", abs(rd.fold_amplitude(flat, 10, 6)) < 1e-12 and rd.fold_amplitude(bump, 10, 6) > 0.05)
from fold_metric import fit as _fit
_P = np.vstack([np.c_[np.linspace(-5, 5, 30), np.full(30, y), 0.01*np.linspace(-5, 5, 30)**2 + 0.02*y**2] for y in np.linspace(-3, 3, 10)])
check("fold curvatures of round_diag are the fold_metric.fit curvatures (a paraboloid z = 0.01 x^2 + 0.02 y^2 gives kx 0.02, ky 0.04)", abs(rd.fold_fit(_P)["kx"] - 0.02) < 1e-9 and abs(rd.fold_fit(_P)["ky"] - 0.04) < 1e-9 and rd.fold_fit is _fit)
csvp = dD / "diagnostics_test.csv"
if csvp.exists(): csvp.rename(csvp.with_suffix(".old.%d" % int(__import__("time").time())))
rd.RoundDiag.append(str(csvp), row); rd.RoundDiag.append(str(csvp), row)
with open(csvp, newline="") as fh: rows = list(csv.reader(fh))
check("diagnostics.csv: header once (also after a resume = second append), one line per round, %d columns" % len(rd.COLUMNS), rows[0] == rd.COLUMNS and len(rows) == 3 and all(len(r) == len(rd.COLUMNS) for r in rows))

# ---------------------------------------------------------------- 7. tightness factor (recorded only)
d12, m12 = scratch("tight_jersey", "0020"); rec12 = lg.tightness_record(m12, str(d12))
exp_tex = 590.5 / m12["ne"]
check("tightness record for Ne %g: tex = 590.5 / Ne = %.3f, loop length %.3f mm, TF = sqrt(tex) / loop length = %.4f per mm (%.2f per cm)" % (m12["ne"], rec12["tex"], rec12["loop_length_mm"], rec12["tightness_factor_per_mm"], rec12["tightness_factor_per_cm"]),
      abs(rec12["tex"] - exp_tex) < 1e-9 and 5.3 < rec12["loop_length_mm"] < 5.7 and abs(rec12["tightness_factor_per_mm"] - math.sqrt(exp_tex)/rec12["loop_length_mm"]) < 1e-12 and abs(rec12["tightness_factor_per_cm"] - 10*rec12["tightness_factor_per_mm"]) < 1e-12 and "definition" in rec12)
mn = dict(m12); mn.pop("ne", None); mn.get("structure", {}).pop("ne", None)
check("no tightness record for a result without Ne (R-unit result): None", lg.tightness_record(mn, str(d12)) is None)
src = (P / "scripts" / "yarn_model" / "relax_job.py").read_text(encoding="utf8"); tree = ast.parse(src)
par = {}
for node in ast.walk(tree):
    for ch in ast.iter_child_nodes(node): par[ch] = node
def inside_if(node, test_src):
    while node in par:
        node = par[node]
        if isinstance(node, ast.If) and test_src in ast.unparse(node.test): return True
    return False
tl = [l.strip() for l in src.splitlines() if "tightness" in l.lower()]
allowed = ("tightness_record", "run_entry['tightness']", "the tightness factor is recorded in meta.json (no effect)")
check("relax_job.py uses the tightness only to RECORD it: every line that mentions it is the record, its except branch or the docstring (%d lines)" % len(tl), all(any(a in l for a in allowed) for l in tl) and sum("run_entry['tightness']" in l for l in tl) == 2 and "numpy" not in " ".join(tl), tl)

# ---------------------------------------------------------------- 8. wiring in relax_job.py (static) and defaults
adds = {a.args[0].value: a for a in ast.walk(tree) if isinstance(a, ast.Call) and getattr(a.func, "attr", "") == "add_argument" and a.args and isinstance(a.args[0], ast.Constant)}
kw = lambda flag: {k.arg: k.value for k in adds[flag].keywords}
check("new options exist with default OFF: --slip-friction-load 0.0, --slip-drive 0.0, --slip-drive-decay 0.0, --diagnostics store_true",
      all(f in adds for f in ("--slip-friction-load", "--slip-drive", "--slip-drive-decay", "--diagnostics")) and all(ast.literal_eval(kw(f)["default"]) == 0.0 for f in ("--slip-friction-load", "--slip-drive", "--slip-drive-decay"))
      and ast.literal_eval(kw("--diagnostics")["action"]) == "store_true" and "default" not in kw("--diagnostics"))
helps = {f: ast.literal_eval(kw(f)["help"]) for f in ("--slip-friction-load", "--slip-drive", "--slip-drive-decay")}
check("--help of the three constants says UNSOURCED; the drive help gives the range -0.1..0.1 and the decay unit (outer rounds)", all("UNSOURCED" in h for h in helps.values()) and "-0.1..0.1" in helps["--slip-drive"] and "outer rounds" in helps["--slip-drive-decay"])
sets = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "slip_ext" and isinstance(par.get(n), ast.Assign) and n in par[n].targets]
check("sim.slip_ext is assigned only inside 'if g.slip_friction_load or g.slip_drive or g.diagnostics:' (nothing is attached to the solver state when all options are off)", len(sets) == 1 and inside_if(sets[0], "g.slip_friction_load or g.slip_drive or g.diagnostics"))
stat = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "slip_stats" and isinstance(par.get(n), ast.Assign) and n in par[n].targets]
rowc = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "append" and "RoundDiag" in ast.dump(n)]
check("the statistics reset and the diagnostics row are written only under 'if diag:'", len(stat) == 1 and inside_if(stat[0], "diag") and len(rowc) == 1 and inside_if(rowc[0], "diag"))
check("the validation refuses a drive outside -0.1..0.1, a negative friction constant or decay, and a decay without a drive", all(s_ in src for s_ in ("--slip-drive must be between -0.1 and 0.1", "--slip-friction-load must be between 0 and 10", "--slip-drive-decay must be >= 0", "--slip-drive-decay needs --slip-drive")))
check("the record (params, slip_ext block with the UNSOURCED list) is written only when an extension is on; diagnostics adds one param flag", "if ext_rec: run_entry['params'].update(" in src and "if diag: run_entry['params']['diagnostics'] = True" in src)
rst = (P / "scripts" / "yarn_model" / "relax_state.py").read_text(encoding="utf8")
check("relax_state.py: only the hooks changed (migrate gets sim.slip_ext / stats, run sets round_index and passes the statistics); slip_at_node is untouched", "def slip_at_node(la, l0a, lb, l0b, EA, mu, phi, step_max):" in rst and "sim.round_index = k; sim.contacts = build_contacts(sim); dmax = migrate(sim, getattr(sim, 'slip_stats', None))" in rst)

# ---------------------------------------------------------------- 9. campaign.py picks the flags up; dry-run command
flags = cm.known_relax_flags()
check("campaign.py finds --slip-friction-load, --slip-drive, --slip-drive-decay and --diagnostics in relax_job.py without hand edits", {"--slip-friction-load", "--slip-drive", "--slip-drive-decay", "--diagnostics"} <= flags)
spec_dir = SM / "campaigns" / "gradespec"; spec_dir.mkdir(parents=True, exist_ok=True)
spec = json.loads((P / "campaigns" / "selftest" / "spec.json").read_text()); spec["name"] = "gradespec"; spec["relax"].update(slip_friction_load=0.5, slip_drive=0.01, slip_drive_decay=20.0, diagnostics=True)
spec["vary"].append(dict(name="slip_drive", min=0.0, max=0.05, scale="linear", integer=False)); (spec_dir / "spec.json").write_text(json.dumps(spec))
try: sp = cm.load_spec("gradespec"); loaded = True
except SystemExit as e: loaded = False; print("   load_spec:", e)
check("a campaign spec may set the four options in 'relax' and vary slip_drive", loaded)
if loaded:
    cmd = cm.relax_cmd(dict(sp["relax"], slip_drive=0.02), 0.13, "/x")
    check("the dry-run command line carries them (--slip-friction-load 0.5, --slip-drive 0.02, --slip-drive-decay 20.0, --diagnostics)", cmd[cmd.index("--slip-friction-load") + 1] == "0.5" and cmd[cmd.index("--slip-drive") + 1] == "0.02" and cmd[cmd.index("--slip-drive-decay") + 1] == "20.0" and "--diagnostics" in cmd, cmd)
    off = cm.relax_cmd({k: v for k, v in sp["relax"].items() if not k.startswith(("slip_", "diagnostics"))}, 0.13, "/x")
    check("without the keys the command line has none of the new flags (unchanged)", not any(c in off for c in ("--slip-friction-load", "--slip-drive", "--slip-drive-decay", "--diagnostics")))
print("ALL OK" if ok else "FAILURES")
