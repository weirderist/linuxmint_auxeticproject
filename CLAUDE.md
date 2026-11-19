

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



\## Hard rules

\- Never run git commit, git push, or any delete (rm, Remove-Item, rm -rf) without asking K first.

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
- Planned: --loop-grade a and --tm-boost b, giving a starting rest-length field (loops smallest at section centre, larger towards borders and corners; larger on the "transfer medium axis" wales). No-shell runs only. The direction of the effect is attributed to Basiri 2010 and Amanatides 2022 via the other AI's CHANGES.txt and is NOT independently verified. The magnitudes of a and b are UNSOURCED.
- Target case: --ne 12 --cpi 22 --wpi 16 --tiles 3 with K's 20-row x 16-column matrix of 1/2 blocks (K will supply the matrix).
- Still needed from physical measurement: yarn length per loop at section centre versus border on a real swatch, measured fold height, and the yarn EA/EI/stretch curve to calibrate.
