# Tests for the web UI, the run viewer and the loop-length colouring

None of these starts a server, a simulation or a campaign. The API tests use FastAPI's `TestClient`; the viewer tests read fixtures under `smoke_test/` (git-ignored). Nothing writes to `campaigns/` or `scripts/`.

## Set up once

```
python -m venv .venv-tests
.venv-tests/Scripts/pip install fastapi httpx numpy scipy esprima      # Windows; on Linux: .venv-tests/bin/pip
python tests/make_fixtures.py
```

`make_fixtures.py` builds the synthetic campaign and a real single-jersey fabric (about a minute, needs scipy). It prints the `scp` commands for the two finished fidelity runs it cannot create itself (runs 0001 and 0003 of `campaigns/fidelity`); run them, then run `make_fixtures.py` again.

## Run (from the repository root; use the venv's python)

| Command | What it checks |
|---|---|
| `python tests/test_api.py` | campaigns API: counts, summary, estimate, start/stop guards, Simulate lock, datasets, pages |
| `python tests/test_boot.py` | boot-id auto-resume rule and the /resume and /tile locks |
| `python tests/test_runview.py` | run viewer: valid run, tube on demand, path traversal, delete/rename refused, colour and strain data |
| `python tests/test_loopfix.py` | loop-length colouring: baseline uniformity on the real single jersey, rest-length mode with a known gradient, colour range |
| `python tests/test_start_interpreter.py` | the Start endpoint uses the project interpreter when it exists |
| `python tests/test_parse_scripts.py` | every script block of the four HTML pages and every script, parsed with a real JavaScript parser (esprima) and `ast` |
| `python tests/test_loop_grade.py` | the starting loop-length field (`scripts/yarn_model/loop_grade.py`) without the solver; see the section below |
| `python tests/test_slip_ext.py` | the opt-in slip extensions, the per-round diagnostics, the tightness record and their wiring in `relax_job.py` / `relax_state.py`, without the solver |
| `python tests/make_golden_loop_grade.py` | not a test: writes `golden_loop_grade.json` (fingerprints of the grade's starting rest lengths); run it ONCE before refactoring `loop_grade.py` |
| `python tests/analysis_slip_rule.py` | not a test: applies the capstan slip rule to the as-built start and a saved relaxed state (arithmetic, no solver) |

Each test prints PASS/FAIL lines and ends with `ALL OK` or `FAILURES`.

## `test_loop_grade.py`: what it proves and what it does not

Needs the fixtures: the finished fidelity runs 0001 (Pattern 1, tiles 1) and 0003 (tiles 2) and the real single-jersey build 0020, all made by `make_fixtures.py`. It works on scratch copies under `smoke_test/grade_tmp/`.

**Proves (no solver, no JAX):**
- Sections on the tiled matrix: Pattern 1 has 3, 6 and 9 sections at tiles 1, 2 and 3 (tile joins merge them); single jersey has one. The labelling and the distance to the section boundary equal SciPy's.
- Strengths on Pattern 1 (wales 6, 7, 14, 15 = 1, others 0, no course axis) and the dilution case (strength 1.0, 0.8, 0.6, 0.4, 0.0; the TM-only gain falls with dilution).
- `scripts/yarn_model/loop_grade.py` reproduces the independent prototype (`analysis_loop_grade_design.py`) in all three modes; the viewer and the field share one loop identity and one `resample` (equal to `relax_state.resample`).
- Option off: the starting rest length is the plain `chord x ratio`, bit for bit; `relax_job.py` calls the grade only inside `if g.loop_grade:` and writes the record only inside `if grade_rec:` (checked on the source with `ast`); the three options have the stated defaults and UNSOURCED help.
- Option on: the total rest length of every section is conserved to rounding (about 1e-16); single jersey is shorter in the centre and longer at the free edge by about `a`, symmetric and monotonic; on Pattern 1 the four longest wales, relative to each loop's own start, are the TM axes 6, 7, 14, 15 in modes `both` and `tm` (not in `boundary`); the shell flag changes nothing in the field; too-large grades, a missing matrix and unknown modes are refused with plain messages.
- Command line and records: `campaign.py` finds the new flags in `relax_job.py` without hand edits, a spec with the keys loads and its dry-run command line carries them; `SimBody`, `_sim_cmd`, the `job_request.json` round trip, the power-cut resume command, presets and the 422 for a result without a matrix all work; the Simulate form has the fields.
- The viewer shows a field written as `relaxed_l0.csv`: rest-length mode, lower in the centre and higher at the edge on single jersey, TM axes longest on Pattern 1; an old run without the file still gets the geometric-arc label.

**Does not prove:**
- That `relax_job.py` runs with the option (it needs JAX; the call, its arguments and the record are checked statically and through the module, never executed in the solver).
- That the solver, the shell hold, creep or migration behave sensibly from a graded start, or that the grade produces the curl it is meant to drive.
- That the field is physically right: every constant is UNSOURCED.
- The web page in a browser (the HTML and script blocks are parsed and the field ids checked, never rendered).
- Anything on the server.

## `test_slip_ext.py`: what it proves and what it does not

**Proves (no solver, no JAX):**
- `loop_grade.py` after the refactor gives bit-identical starting rest lengths for six fixed cases (SHA-256 against `golden_loop_grade.json`).
- `slip_ext.slip_at_node_ext` with no offsets and no extra limit returns exactly what `relax_state.slip_at_node` returns (2500 random nodes); a larger `mu_n F_n` moves less rest length and the contact then sits exactly on the new limit; a drive offset moves rest length toward the segment that looks more tense.
- `relax_state.migrate` with nothing set equals the old function exactly on a real saved relaxed state; the slip statistics (number, sum, largest) equal the values the slip rule returned; transfers conserve the rest length of every yarn.
- `mu_n`: fewer and smaller transfers; a huge `mu_n` holds every contact with overlap and leaves contacts without overlap alone.
- The drive: from a uniform start with no drive nothing happens (as the probe found); with the drive the slip rule acts at many contacts and rest length moves toward the higher-G loops, only where the field changes; a negative drive reverses it; at the as-built +5.4 % pre-strain a drive of 0.01 stays inside the friction limit and 0.1 does not; `rs.run` sets `round_index`, passes the statistics and applies the exponential decay.
- Diagnostics: all 30 documented columns, region means above / below 1 for the outer ring / inner core of a graded jersey, share of the change stored in l0, fold descriptor 0 for a flat sheet, header once in the CSV; tightness record (Ne 12: tex 49.208, loop length 5.489 mm, factor 1.278 per mm).
- Wiring in `relax_job.py` (checked on the source with `ast`): the four options default off, are validated, say UNSOURCED, and attach nothing to the solver state unless set; the record is written only when an extension is on; the tightness is only recorded; `campaign.py` picks the flags up and its dry-run command carries them.

**Does not prove:**
- That the extensions or the diagnostics run inside the real solver (needs JAX): in a real run the relaxation re-equilibrates positions between rounds, which the tests replace by a no-op.
- That the drive produces a persistent chain or the loop-length gradient, how the sign of the feedback comes out, or that any value of `mu_n`, `e` or `N` is right: all are UNSOURCED.
- That the fold descriptor equals `campaign.py`'s on real data (same definition, not compared).
- Anything in a browser or on the server.

## Pure shell only runs (no solver; the shell is replaced by a stub)

| Command | What it checks |
|---|---|
| `python tests/test_shell_campaign.py` | settings validation, ids, queue, duplicates, runner end to end, pause / continue / stop, per-run timeout, disk guard, power-cut recovery (quarantine, publish of finished attempts), no overwrite, delete path guard and log, descriptors, KMSH export, `shell_run.py` stub. Two tests (symlink, in-progress delete) skip on Windows |
| `python tests/test_shell_api.py` | `/api/shell/...`: list, detail, controls, start guards, delete with typed confirmation, page and script |
| `python tests/test_shell_policy_v2.py` | the shell2 policy (`campaigns/shell2_policy.json`, 2026-11-27): 1000 generated runs plus the seeds, mix 400 / 200 / 400, every block of 100 balanced and covering R and CPI x WPI, every entry valid, hinge / L-BFGS / 31-row entries refused, unknown solver rejected, shell1 settings unchanged, deterministic |
| `python tests/test_shell_leases.py` | the shared queue with leases (2026-11-27): claimed runs skipped by every other worker, concurrent claims never share a run, release only by the holder, expiry requeues, the server runner leases what it runs and skips Kaggle leases, stale server leases released at start, `import_result` (worker recorded, idempotent, refuses changed settings and foreign code), status and rows by worker |
| `python tests/test_workstation.py` | the workstation worker end to end (2026-11-27; local stand-in for ssh/scp, stub solver, the real page route): nothing claimed until allowed; specific runs first, then the queue up to the allowance; frozen code checked by sha256; upload and import with worker workstation:<host>; Cancel stops and releases the run (folder kept in quarantine); a power cut releases the run at the next start; a network outage does not stop the run; an expired lease is not renewed back |
| `python tests/test_kaggle_controller.py` | `scripts/kaggle/kaggle_controller.py` with a fake Kaggle CLI (2026-11-27): disabled claims nothing, a push claims a batch, an outage changes nothing, completion imports done runs (worker kaggle:<slot>), marks failed ones, requeues the rest and starts the next batch, a lost session and a failed push release their runs; the worker template itself runs (stub solver) and writes its manifest |
| `python tests/test_shell_policy.py` | the approved policy (`campaigns/shell_policy.json`): ESSENTIAL settings enforced at enqueue and by the runner, labelled controls, ratio / density / tiles / stitch-cap limits, slight pattern edits, the deterministic seeded plan (11 seed runs, 2 controls, 91 variants, filler) and `shell-plan --policy` |
| `python tests/test_shell_strain.py` | `scripts/shell_model/shell_strain.py`: flat sheet, uniform stretch and an isometric cylinder give the exact strains; the colour and legend contract of the shell viewer |
| (`tests/scriptdirs.py`) | not a test: puts scripts/ and its purpose folders on `sys.path`; every test that imports a script module imports it right after its `sys.path` line |
| `python tests/test_shell_target.py` | `scripts/shell_derived_target/shell_target.py`: stretch of a flat sheet, an isometric cylinder, a uniform stretch and a rigid rotation; per-course conservation; the sign of the factors |
| `venv/bin/python tests/test_shell_plate.py` | NEEDS JAX (run on the server): the plate bending term of `shell_sim.py` (`bending='plate'`, 2026-11-26) against the exact Kirchhoff plate energy for a cylinder, sphere, aligned saddle and 45-degree twist at three mesh sizes (all 0.977-0.983, mesh-independent; the old hinge term 2-10 x), the sign convention, and a free sheet relaxing to its rest curvature |
| `venv/bin/python tests/test_shell_newton.py` | NEEDS JAX (run on the server): the Newton, FIRE and FIRE+Newton solvers and the stability check of `shell_sim.py` (2026-11-27): FIRE+Newton reaches the Newton energy, no negative eigenvalue beyond the rigid modes; sparse element Hessian against the JAX Hessian-vector product (hinge and plate, with and without springs), every growth step converged, energy not above a 20000-iteration L-BFGS run, default solver unchanged |
| `python tests/test_shell_fold.py` | `scripts/shell_derived_target/shell_fold.py` (progressive fold, no JAX: a stand-in relax_solver and a stub relaxation): vertex fractions, blocked test, freezing, per-course conservation at every stage, signs of the loop change, the spacing and loop-length measurement, files and aggregation |
| `python tests/test_shell_learning.py` | pattern features, the rule job (lasso, tree, 100-run minimum, label) and the Gaussian-process proposer (tier 5, policy-enforced, fixed test against random) on synthetic finished runs |
| `python tests/test_shell_proposer_gate.py` | the proposer gate and hard cap (`scripts/shell_learning/shell_proposer.py`): closed at 799 and open at 800 finished original runs, failed / timed-out / rejected / needs_approval runs, controls and proposer runs not counted, already-queued tier-5 entries left byte-identical on disk, cap of 100 (200 existing: nothing added; 98: only 2 added), unreadable or missing files keep the gate closed. Does NOT test the runner, which is frozen code and still dispatches queued tier-5 entries (about 3.5 min) |
| `python tests/test_theme.py` | `webui/theme.css` and the Campaigns page: dark only, every colour defined once in `:root`, no colour literal or undefined variable in the page files, the stylesheet is served |
