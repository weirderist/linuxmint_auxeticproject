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
| `python tests/test_loop_grade.py` | the starting loop-length field (`scripts/loop_grade.py`) without the solver; see the section below |
| `python tests/analysis_slip_rule.py` | not a test: applies the capstan slip rule to the as-built start and a saved relaxed state (arithmetic, no solver) |

Each test prints PASS/FAIL lines and ends with `ALL OK` or `FAILURES`.

## `test_loop_grade.py`: what it proves and what it does not

Needs the fixtures: the finished fidelity runs 0001 (Pattern 1, tiles 1) and 0003 (tiles 2) and the real single-jersey build 0020, all made by `make_fixtures.py`. It works on scratch copies under `smoke_test/grade_tmp/`.

**Proves (no solver, no JAX):**
- Sections on the tiled matrix: Pattern 1 has 3, 6 and 9 sections at tiles 1, 2 and 3 (tile joins merge them); single jersey has one. The labelling and the distance to the section boundary equal SciPy's.
- Strengths on Pattern 1 (wales 6, 7, 14, 15 = 1, others 0, no course axis) and the dilution case (strength 1.0, 0.8, 0.6, 0.4, 0.0; the TM-only gain falls with dilution).
- `scripts/loop_grade.py` reproduces the independent prototype (`analysis_loop_grade_design.py`) in all three modes; the viewer and the field share one loop identity and one `resample` (equal to `relax_state.resample`).
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
