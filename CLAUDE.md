

\# Project: Design and Development of Three-Dimensional Auxetic Knitted Fabrics



Final-year textile technology project (PSG Polytechnic), due Jan/Feb 2027.

Goal: an open-source, loop/yarn-level finite element simulation of knitted fabric that reports the Poisson's ratio (NPR). No Abaqus or other commercial solvers.

The owner (K) is a textile student. Define new technical concepts explicitly, use a formal register, and cite sources for physical claims.



\## Where things run

\- This machine (Windows ThinkPad) holds the git clone and runs Claude Code. Do NOT run simulations, JAX jobs or the web UI here.

\- The server (old laptop, Linux Mint) runs all jobs. Reach it only as: ssh server "<command>". User is sarvesh, project folder is \~/auxetic\_project.

\- The web UI is FastAPI on the server, viewed through an SSH tunnel.

\- The server has abrupt power outages. Start long jobs with nohup or tmux so they survive a dropped connection. The solver checkpoints and can resume.



\## Workflow

1\. Edit code here. 2. Commit and push to GitHub only after K approves. 3. Update the server with: ssh server "cd \~/auxetic\_project \&\& git pull". 4. Run jobs on the server and read the results back (use scp for result files).

\- Restart the web service only when main.py changed AND no job is running. Check job\_stage.json or the /state endpoint first.

\- The web service unit is auxetic\_webui.service. The restart command is: ssh -t server "sudo systemctl restart auxetic\_webui.service". Only K runs it; Claude never runs any systemctl command, it gives K this command.



\## Hard rules

\- Never run git commit, git push, or any delete (rm, Remove-Item, rm -rf) without asking K first.

\- Never use rm or any other delete, even on Claude's own scratch files; ask K first.

\- Never read, print or commit .env, tokens or keys. .env is not in git and must stay that way.

\- Do not edit or delete anything under results/ or run\_knit/ on the server.

\- Do not invent constants. Mark any magnitude without a source as UNSOURCED in code comments and in the reply.

\- Report honestly. State what was measured, what was only tested on a small case, and what is not yet validated.

\- Give K one command at a time when walking through steps on his machine.



\## Code map

\- scripts/relax\_job.py: job driver, writes job\_stage.json (prepare, tile, shell, ramp, fold, relax).

\- scripts/relax\_solver.py, relax\_state.py: yarn-level elastic rod model (JAX autodiff + SciPy L-BFGS-B).

\- scripts/build\_knit.py: geometry and mesh export (KMSH .bin always, OBJ only when small).

\- scripts/shell\_sim.py: shell-hold simulation. yarn\_check.py, fold\_metric.py, fold\_scan.py, tile\_sim.py: checks and scans.

\- webui/main.py (FastAPI) and webui/index.html (viewer and UI).

## Current work (handoff from claude.ai chat, 2026-11-19)

- Loop-length migration: yarn slip between loops is modelled as capstan-style rest-length transfer along the yarn. In a small test the migration was only about 1e-7 per wale, so its effect is not yet demonstrated.
- Status: the yarn-only simulation does not yet rival the shell-hold simulation. Folding appears only when the start is seeded. The other AI's "roll" result is a single 12x12 case whose control run had not converged. Treat it as unverified.
- Planned (note: --tm-boost is superseded by --loop-grade-target tm, see tests/NOTES_loop_grade.md): --loop-grade a and --tm-boost b, giving a starting rest-length field (loops smallest at section centre, larger towards borders and corners; larger on the "transfer medium axis" wales). No-shell runs only. The direction of the effect is attributed to Basiri 2010 and Amanatides 2022 via the other AI's CHANGES.txt and is NOT independently verified. The magnitudes of a and b are UNSOURCED.
- Target case: --ne 12 --cpi 22 --wpi 16 --tiles 3 with K's 20-row x 16-column matrix of 1/2 blocks (K will supply the matrix).
- Still needed from physical measurement: yarn length per loop at section centre versus border on a real swatch, measured fold height, and the yarn EA/EI/stretch curve to calibrate.

## Planned, on hold: phone app

Not built yet. Do not start it until K says so. Decisions so far:

- The PIN is 6 digits (this replaces the earlier 4-digit idea).
- After 5 wrong tries the PIN stays locked until K clears it with a command on his PC terminal. There is no timed unlock. Claude gives K that command when the app is built.

## Decisions and findings, 2026-11-20

- The shell is the accepted fold reference. A shell comparison is optional per run. The final yarn-level run is scored for similarity out of 100.
- The first shell success was made with commit 8346645 (13 Nov) with E 3700, mu 0.49, contact 40, perturb 0.20, outer 40, L-BFGS 400, curl 0.5, tension 17 cN, creep 3 h over 6 h, shell on, hold 0.05 x EA/R, tiles 3, Ne 12, CPI 22, WPI 16. It ended unsettled but folded very close to the real fabric.
- Reproduction runs are in ~/old_runs on the server (A then B). Do not touch them.
- New plan: shell-only experiments first (no yarn mesh), validate against real fabrics, an ML loop on random then proposed patterns to find fold rules, then the yarn mesh with migration and creep, then parameter sensitivity.
- Loop-grade (sectioned, graded TM-axis rule) is designed and parked, see tests/. All its constants are UNSOURCED.
