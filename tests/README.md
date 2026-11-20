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
| `python tests/analysis_slip_rule.py` | not a test: applies the capstan slip rule to the as-built start and a saved relaxed state (arithmetic, no solver) |

Each test prints PASS/FAIL lines and ends with `ALL OK` or `FAILURES`.
