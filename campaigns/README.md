# Campaigns

## Quick commands

Paste these into PowerShell on the ThinkPad. Each is one line and works without Claude Code. Replace `fidelity` with the name of a folder in `campaigns/` (`fidelity`, `fidelity_large`, `screen1`, `shell_ref`, ...).

**Once, after new campaign files were pushed to GitHub:** copy them to the server.

```
ssh server "cd ~/auxetic_project && git pull"
```

**See which campaigns exist:**

```
ssh server "ls ~/auxetic_project/campaigns"
```

**1. Plan** (fixes the list of runs; do this once per campaign, it refuses to overwrite an existing plan):

```
ssh server "cd ~/auxetic_project && venv/bin/python scripts/campaign.py plan fidelity"
```

**2. Check what would run** (starts nothing, prints every command):

```
ssh server "cd ~/auxetic_project && venv/bin/python scripts/campaign.py run fidelity --dry-run"
```

**3. Start** (detached: it keeps running after you close PowerShell or the connection drops):

```
ssh server "cd ~/auxetic_project && venv/bin/python scripts/campaign.py run fidelity --detach"
```

**4. Check status** (counts, and a table of the runs; also refreshes results.csv):

```
ssh server "cd ~/auxetic_project && venv/bin/python scripts/campaign.py status fidelity"
```

**5. Pause (stop):** the run in progress finishes, then the runner stops. Nothing is killed. This can take as long as one run (minutes to an hour for the larger fabrics).

```
ssh server "cd ~/auxetic_project && venv/bin/python scripts/campaign.py stop fidelity"
```

**6. Resume:** start it again with the same command as step 3. Finished runs are never repeated; queued runs go on. Use the same command after a power cut: a run that was cut off is detected, requeued and continued from its last checkpoint.

```
ssh server "cd ~/auxetic_project && venv/bin/python scripts/campaign.py run fidelity --detach"
```

**Is the runner alive?** (empty output means no runner and no job is running):

```
ssh server "ps -eo pid,etime,cmd | grep -E 'campaign.py|relax_job.py' | grep -v grep"
```

**Runner log** (last 20 lines):

```
ssh server "tail -n 20 ~/auxetic_project/logs/campaign_fidelity.log"
```

**Where the results are.** On the server, in `~/auxetic_project/campaigns/fidelity/results.csv` (one row per run; the details of each run are in `~/auxetic_project/campaigns/fidelity/runs/0001/` and so on; `relax.log` in that folder is the solver's own log). To read it on screen:

```
ssh server "cat ~/auxetic_project/campaigns/fidelity/results.csv"
```

To copy it to the current PowerShell folder and open it in Excel:

```
scp server:auxetic_project/campaigns/fidelity/results.csv .\fidelity_results.csv
```

**How to read results.csv.** One row per run, columns for the settings of that run first, then the measurements. An empty cell means "could not be computed", never zero.

- `status`: `done` (finished), `failed` (see the `reason` column), `queued` (not yet run), `running`. `interrupted` counts power cuts or crashes the run survived.
- `exit_code`: 0 = the solver settled, 2 = it used all its iterations without settling (both leave a usable result). Anything else is a failure.
- `rel_dE_last`: how much the energy still changed in the last round. About 0 = settled; clearly negative = still falling, so the run was stopped before it converged.
- `min_dist_over_2R_end`: closest approach of two yarns in units of the contact distance. Below about 0.97 the yarns pass into each other.
- `fold_direction` (`course`, `wale`, `both`, `none`), `fold_amplitude_mm`, `fold_local_amplitude_mm` and the two `wavelength_*` columns describe the fold. The wavelength of an almost flat sheet is noise: read it only when `fold_local_amplitude_mm` is clearly above zero. The full definitions are under "results.csv columns" below.

**WARNING: while a campaign is running, do not press Simulate in the web page.** The web page only knows about jobs it started itself. A second job at the same time would slow both down and could use too much memory. Wait until `status` shows no `running` run (or pause with step 5 and wait for the runner to stop) before using the web page. The runner itself waits whenever it finds another solver job running.

## Details

A campaign runs many `relax_job.py` simulations one after another on the server and records every result in one table. It is driven by `scripts/campaign.py` (standard library plus numpy). Run it on the server, in the project virtual environment.

```
campaigns/<name>/
  spec.json        what to run (written by hand)
  manifest.json    the numbered runs with their exact parameter values (written by `plan`)
  results.csv      one row per run: parameters and metrics (rewritten after every run)
  runner.lock      held (flock) by the running runner; the lock vanishes with the process
  STOP             exists only while a stop is requested
  runs/<NNNN>/     one folder per run (not tracked by git): status.json, metrics.json, build.log, relax.log, exit.<attempt>, and the usual result files
campaigns/pattern1_matrix.txt   the 20-row x 16-column stitch matrix (rows separated by "/")
```

## Commands

```
venv/bin/python scripts/campaign.py plan   <name>          write manifest.json (refuses if one exists; it never overwrites)
venv/bin/python scripts/campaign.py run    <name> --dry-run   print the build and relax commands of every queued run; starts nothing
venv/bin/python scripts/campaign.py run    <name> --detach    start the runner in its own session with SIGHUP ignored (same effect as nohup setsid ... &), log in logs/campaign_<name>.log
venv/bin/python scripts/campaign.py status <name>          counts and a table of the runs (also rewrites results.csv)
venv/bin/python scripts/campaign.py stop   <name>          create STOP; the runner checks it between runs and then exits
```

Equivalent by hand: `nohup setsid venv/bin/python scripts/campaign.py run <name> > logs/campaign_<name>.log 2>&1 &`.

`stop` never kills anything. The run in progress finishes; the runner then renames STOP to `STOP.honored.<time>` and exits. A STOP file found when a runner starts is an old request: it is renamed `STOP.stale.<time>` and the runner continues.

## spec.json

| Key | Content |
|---|---|
| `name` | Must equal the folder name. |
| `fabric` | `matrix` (text, rows separated by `/`) or `matrix_file` (path relative to `campaigns/`), and `ne`, `cpi`, `wpi`, `tiles`. |
| `relax` | Fixed `relax_job.py` options, written without the leading dashes and with `_` for `-` (for example `"seed_stitch": 0.2`, `"curl_aniso": true`). `E` and `mu` are required. Give exactly one of `kc_ratio` (contact stiffness as a multiple of EA/R, the rule of `webui/main.py`) or `k_contact`. Names are checked against the options found in `relax_job.py`. |
| `vary` | List of `{name, min, max, scale: linear or log, integer: true or false}`. A name is a build option (`tiles`, `ne`, `cpi`, `wpi`) or a relax option. |
| `design` | `{"type": "latin_hypercube", "seed": s, "n_runs": n}` (one point in each of n equal strata per parameter; log scale is stratified in the logarithm) or `{"type": "explicit", "runs": [{...}, ...]}`. Values are rounded to 6 significant digits so that the manifest and the command line agree exactly. |
| `timeout_s` | Wall-clock limit per run, summed over its attempts. |
| `max_rss_gb` | Optional memory limit of a run (resident memory of the job and its children). |
| `max_interrupts` | Optional (default 3): a run interrupted more often than this is marked failed. |
| `thresholds` | `fold_min_turn_rad` and `wavelength_min_power_fraction`, used only to classify the fold metrics (below). |
| other keys | Ignored by the program; use them for notes (`unsourced`, `estimate`, ...). Mark every value without a source as UNSOURCED. |

`run` refuses to start if `spec.json` changed after `plan` (hash stored in the manifest); `--ignore-spec-change` overrides.

### Starting loop-length grade (`--loop-grade`, opt-in)

By default every run starts from a uniform rest length (the as-built chord times one ratio). The grade changes that start: rest length grows toward the boundary of every *section* (a 4-connected region of one loop type; the fabric edge counts as a boundary) and along *TM axes* (wales or courses that are mostly one type, stronger the purer they are). The total rest length of every section is conserved exactly. It works with and without `shell`. Rule, prototype and decisions: `tests/NOTES_loop_grade.md`, code: `scripts/loop_grade.py`.

Put the three keys in `relax` (they are checked against the options of `relax_job.py` like every other key), or give `loop_grade` in `vary`:

```
"relax": { ..., "loop_grade": 0.02, "loop_grade_target": "both", "loop_grade_min_strength": 0.2 }
"vary":  [ {"name": "loop_grade", "min": 0.0, "max": 0.05, "scale": "linear", "integer": false} ]
```

| Key | Meaning |
|---|---|
| `loop_grade` | `a`: how much longer the loops at the target are than the farthest loops of their section, as a fraction of the section's mean loop length (0.02 = 2 %). `0` or absent = off: the start is bit-identical to a run without the option. Range -0.4 to 1. |
| `loop_grade_target` | `boundary`, `tm` or `both` (default `both`). |
| `loop_grade_min_strength` | Smallest TM-axis strength `s = 2 (f - 0.5)` (f = fraction of the majority type of a wale or course) that counts as an axis. Default 0.2. |

The `--dry-run` printout shows the three flags with their values. `meta.json` of the run records them under `simulations[-1].params` and a `loop_grade` block (factor range, per-section totals before and after, the axis bands) only when the grade is on.

**Everything here is UNSOURCED**: `a`, the minimum strength 0.2, the strength mapping `s = 2 (f - 0.5)`, the linear fall-off with distance, the mode weights (both 1), the per-section additive normalisation. No measured loop-length field exists yet; K will set them from loop lengths measured by region.

**Two things to know.** (1) Tile joins merge sections: Pattern 1 has 3 sections at `tiles` 1, 6 at 2 and 9 at 3, so the sections grow with `tiles` and the same `a` gives a different field at a different `tiles`. (2) Creep (`creep_tau`, `hours`) moves every rest length toward its current length each round and does **not** conserve the total, so it can partly erase the starting grade.

## What happens after a power cut or a dropped connection

- The runner and every job are started in their own sessions, so a dropped SSH connection stops nothing.
- Each run has `status.json` (queued, running, done or failed), written atomically (temporary file, fsync, rename). The process of an attempt is recorded with its pid, its start time and the boot id, so a pid reused after a reboot is never mistaken for the job.
- When a runner starts it looks at every run marked `running`:
  - the job has finished while no runner was alive (an `exit.<n>` file exists): the result is recorded normally;
  - the job is still alive (only the runner died): the new runner adopts it and waits for it;
  - no live process and no exit file (power cut): the run is marked **interrupted** in its history, `interrupted` in results.csv is increased, and the run is **requeued**. If `checkpoint.npz` exists, `relax_job.py --resume` continues from it (at most one outer round is lost; the solver writes the checkpoint after every outer round); otherwise the run starts again from scratch. A build that was interrupted is repeated.
- Only one runner per campaign can run (flock on `runner.lock`). Start the runner again after a power cut by hand (or from a boot script); nothing restarts it by itself.
- `wall_s` is the sum over the attempts that ended normally; the length of an interrupted attempt is unknown and is not included (see the `interrupted` column).
- A run that exceeds `timeout_s` or `max_rss_gb` is stopped by the runner: SIGTERM to the process group of that run only (its wrapper was started with `setsid`), SIGKILL only if it ignores SIGTERM for 60 s. The run is marked failed with the reason. Nothing else is touched.

## One job at a time, and politeness

Before each run the runner checks that no other `relax_job.py` or `tile_sim.py` process is alive (a web-UI job, or a job of another campaign) and otherwise waits, checking again every 60 s (`--poll-s`). `job_stage.json` files under `webui/results` that belong to no live process (left by a power cut) are ignored. The check is made only when a run is about to start. The web UI checks only the jobs it started itself, so it does not know about a campaign run: do not press Simulate in the UI while a campaign is running.

## results.csv columns

Run number, status, number of attempts, number of interruptions, then one column per varied parameter, then:

| Column | Meaning |
|---|---|
| `exit_code` | Exit code of `relax_job.py` (0 settled, 2 ended without settling; both leave a relaxed state; anything else is a failure). |
| `relax_state`, `settled_by` | The solver's final state and, if settled early, why. |
| `build_s`, `wall_s` | Seconds for the build and for the relaxation attempts (see above). |
| `vertices` | Number of solver vertices (rows of `relaxed_axis.csv`). |
| `rounds_used`, `outer_max` | Outer rounds done and allowed. |
| `E_start`, `E_final`, `E_stretch`, `E_bend`, `E_contact` | Energy at the start and after the last round, with the parts of the last round. Energies are comparable only between runs with the same yarn settings. |
| `rel_dE_last` | (E of the last round - E of the round before) / abs(E of the last round). Negative means the energy was still falling; a value near 0 means settled. This is how to judge convergence when the iteration cap was reached. |
| `min_dist_over_2R_start`, `_end` | Minimum distance between yarn axes divided by the contact distance 2R; below about 0.97 the yarns interpenetrate. |
| `rest_len_change_total`, `rest_len_change_max_abs_wale` | Rest-length change relative to the start: total (from the solver's end summary) and the largest absolute change in any wale. These are the available check on yarn length conservation; the total yarn length itself is not recomputed. |
| `shrink_x`, `shrink_y`, `z_range_over_R_end` | Size change of the relaxed fabric against the built one, and its thickness in yarn radii. |
| `fold_kx_per_mm`, `fold_ky_per_mm` | Curvatures of the least-squares quadratic fit to all vertices (`fold_metric.fit`): kx along the courses (x), ky along the wales (y); positive bends toward +z. |
| `fold_turn_x_rad`, `fold_turn_y_rad` | abs(k) x sheet size: the total turning angle across the sheet. |
| `fold_direction` | `course` if only the turn along x reaches `fold_min_turn_rad`, `wale` if only along y, `both`, or `none`. |
| `fold_amplitude_mm` | Peak-to-peak of the stitch-cell mean heights after removing a best-fit plane (curl and pleats together). |
| `fold_local_amplitude_mm` | The same after removing the quadratic fit (local pleats only). |
| `wavelength_course_mm`, `wavelength_wale_mm` | Dominant wavelength of the local relief: sheet length divided by the strongest spatial frequency (k = 1 upward) of a 2D Fourier transform of the stitch-cell heights, along x and along y. Empty if that frequency carries less than `wavelength_min_power_fraction` of the power. |
| `wl_course_power_fraction`, `wl_wale_power_fraction` | That power fraction. |
| `reason` | Why a run failed (timeout, RAM limit, build error, last log line). |

Cells that cannot be computed are left empty; no value is guessed. Caveats: the wavelength of a nearly flat sheet is meaningless (it fits noise of a few millimetres), so read it together with `fold_local_amplitude_mm`; the wavelength resolution is two stitches; the thresholds in `thresholds` are classification choices, not physical constants.

## Reproducibility

Each run's `status.json` records the host name, the Python, jax, jaxlib, scipy and numpy versions, the git commit of the project and whether `scripts/` had uncommitted changes, and for every attempt the exact command line.

## Paths

`CAMPAIGN_ROOT` (or `LADDER_ROOT`) sets ROOT, default `~/auxetic_project`. `CAMPAIGN_LOG_DIR` (or `LADDER_LOG_DIR`) sets LOG_DIR, default `ROOT/logs`, where the runner log goes. `CAMPAIGN_DIR` and `CAMPAIGN_SCRIPTS` move the campaigns folder and the solver scripts.
