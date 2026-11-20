# Project: Design and Development of Three-Dimensional Auxetic Knitted Fabrics

Final-year textile technology project (PSG Polytechnic), due Jan/Feb 2027.
Goal: an open-source, loop/yarn-level finite element simulation of knitted fabric that reports the Poisson's ratio (NPR). No Abaqus or other commercial solvers.
The owner (K) is a textile student. Define new technical concepts explicitly, use a formal register, and cite sources for physical claims.

## Where things run
- This machine (Windows ThinkPad) holds the git clone and runs Claude Code. Do NOT run simulations, JAX jobs or the web UI here.
- The server (old laptop, Linux Mint) runs all jobs. Reach it only as: ssh server "<command>". User is sarvesh, project folder is ~/auxetic_project.
- The web UI is FastAPI on the server, viewed through an SSH tunnel.
- The server has abrupt power outages. Start long jobs with nohup or tmux so they survive a dropped connection. The solver checkpoints and can resume.

## Workflow
1. Edit code here. 2. Commit and push to GitHub only after K approves. 3. Update the server with: ssh server "cd ~/auxetic_project && git pull". 4. Run jobs on the server and read the results back (use scp for result files).
- Restart the web service only when main.py changed AND no job is running. Check job_stage.json or the /state endpoint first.
- The web service unit is auxetic_webui.service. The restart command is: ssh -t server "sudo systemctl restart auxetic_webui.service". Only K runs it; Claude never runs any systemctl command, it gives K this command.

## Hard rules
- Never run git commit, git push, or any delete (rm, Remove-Item, rm -rf) without asking K first.
- Never use rm or any other delete, even on Claude's own scratch files; ask first.
- Never read, print or commit .env, tokens or keys. .env is not in git and must stay that way.
- Do not edit or delete anything under results/ or run_knit/ on the server.
- Never touch ~/old_runs on the server (the reproduction runs).
- Do not invent constants. Mark any magnitude without a source as UNSOURCED in code comments and in the reply.
- Report honestly. State what was measured, what was only tested on a small case, and what is not yet validated.
- Give K one command at a time when walking through steps on his machine.

## Start of every session (standing rule)
At the start of every session, before anything else, read the TO DO: section. Run only read-only checks needed to see whether each task's dependency is cleared (for example, is a job or campaign running on the server, via the existing read-only status commands). Then tell the user in a short list: tasks that are due now, and tasks still waiting with the reason. Ask which to do. Never start a server job, pull, restart or commit on your own because a task is due; the user decides. When a task is finished, mark it done with the date and move it to a Done list at the bottom of the section; when a new task or a dependency appears, add it. Never remove an entry any other way.

## TO DO:
Format: `- [ ] N. job | WAITING FOR: dependency | OWNER: Claude or K`.

- [ ] 1. Analyse the old-code reproduction runs ~/old_runs/A and ~/old_runs/B (convergence, fold shape against the Blender reference, compare with the 13 Nov settings) | WAITING FOR: server busy until both runs finish (about 23:00 IST on 2026-11-20; check ~/old_runs/chain.log) | OWNER: Claude
- [ ] 2. Run the identical settings in the CURRENT code at the same size, and diff 8346645..HEAD for the cause of any regression (about 5 h server time) | WAITING FOR: task 1 and user approval | OWNER: Claude
- [ ] 3. Pull the approved commits on the server and run the optional numpy/scipy check in webui/venv | WAITING FOR: user approval and no job running on the server | OWNER: Claude
- [ ] 4. Restart the web service (sudo) and add KillMode=process to auxetic_webui.service plus sudo systemctl daemon-reload | WAITING FOR: task 3 and no job running | OWNER: K
- [ ] 5. Open /?run=fidelity/0003, /strain?run=fidelity/0003 and /run?run=fidelity/0003 through the tunnel and check the visuals (none has been seen in a browser yet); then Claude fixes what is wrong | WAITING FOR: task 4 | OWNER: K, then Claude
- [ ] 6. Smallest confirmation run: 8x8 single jersey, tiles 1, Ne 12, CPI 22, WPI 16, current code (writes relaxed_l0.csv), with --loop-grade 0 and with the grade (e.g. 0.02), same other settings | WAITING FOR: loop-grade code (committed a177053..2097e8e) pulled on the server, server idle, user approval | OWNER: Claude
- [ ] 7. Shell-only stage: read-only investigation of shell_sim.py (inputs, time per run, fold descriptors), then a proposal to extend scripts/campaign.py with a shell-only job type, a pattern source (fixed / random / proposer) and a descriptor column | WAITING FOR: nothing for the investigation (do it when asked); design approval before any code | OWNER: Claude
- [ ] 8. Validate the shell against the real fabrics (Miura-ori, Pattern 1, Batch B samples 1-11) | WAITING FOR: task 7 | OWNER: Claude
- [ ] 9. ML loop (random patterns, then proposed patterns, readable rules) on the shell | WAITING FOR: tasks 7 and 8 | OWNER: Claude
- [ ] 10. Fit the loop-grade / migration parameters so the yarn-level fold matches the shell, scored out of 100 | WAITING FOR: tasks 6 and 9 | OWNER: Claude
- [ ] 11. Parameter sensitivity screen (about 36 inputs) | WAITING FOR: task 10, server time | OWNER: Claude
- [ ] 12. Power-cut hardening: checkpoint.npz fsync + checkpoint.prev.npz fallback + validation, then one-owner decide_resumes() with desired.json (resume only if the boot id changed and no STOP file; cap of 3 resumes per run, then needs_approval) | WAITING FOR: no campaign running; user approval | OWNER: Claude
- [ ] 13. Stretch-test engine (Test mode), scoring with the K-ratings queue, proposer/advisor, "Add measurement" form | WAITING FOR: shell validated (task 8) | OWNER: Claude
- [ ] 14. Add campaigns/*/runner.lock to .gitignore (the server shows it as untracked) | WAITING FOR: user approval (trivial) | OWNER: Claude
- [ ] 15. Phone app (Android PWA over Tailscale HTTPS; 6-digit PIN; after 5 wrong tries it stays locked until K clears it with a command on his PC, which Claude gives when built) | WAITING FOR: ON HOLD until K asks | OWNER: Claude
- [ ] 16. fidelity_large | STATUS: do NOT run (not worth it) | OWNER: -
- [ ] 17. Physical measurements (reminders): whole-course unravelled loop length and yarn length per loop at section centre versus border, relaxed CPI/WPI by region, GSM and thickness, fold height and spacing, jersey curl photos with parameters, yarn EA/EI/stretch curve (tensile) if possible, fabric tensile with lateral-strain video (2-3 weeks), and a note on how the real swatch lay when photographed and tested | WAITING FOR: needs user action | OWNER: K
- [ ] 18. Revoke the exposed Telegram token and remove it from .env (never print it) | WAITING FOR: needs user action | OWNER: K
- [ ] 19. The Google Doc "Yarn-Level Knit Simulation: Development Record" is stale | WAITING FOR: user request | OWNER: Claude
- [ ] 20. Per-loop relative rest-length colouring in the viewer (each loop against its own start, so as-built differences between loop types are not shown as migration) | WAITING FOR: user request | OWNER: Claude
- [ ] 21. Test --loop-grade on the 13 Nov settings (tiles 3, Pattern 1) | WAITING FOR: task 1 | OWNER: Claude
- [ ] 22. Decide how the grade scales with tile count: the fall-off now runs over the whole section, so the same a differs at tiles 1, 2 and 3 (option: a fixed length scale in loops) | WAITING FOR: task 6 | OWNER: K and Claude
- [ ] 23. Decide the single-jersey contour shape (square versus elliptical, corners longer) by comparing with the shell | WAITING FOR: task 8 | OWNER: K and Claude
- [ ] 24. Check the two-loop-wide ridge at shared boundaries and the dilution sensitivity against the shell | WAITING FOR: task 8 | OWNER: K and Claude

### Done
(none yet)

## Code map
- scripts/relax_job.py: job driver, writes job_stage.json (prepare, tile, shell, ramp, fold, relax); writes relaxed_l0.csv (rest lengths) at the end.
- scripts/relax_solver.py, relax_state.py: yarn-level elastic rod model (JAX autodiff + SciPy L-BFGS-B). scripts/loop_grade.py: optional starting loop-length field (numpy only).
- scripts/build_knit.py: geometry and mesh export (KMSH .bin always, OBJ only when small).
- scripts/shell_sim.py: shell-hold simulation. yarn_check.py, fold_metric.py, fold_scan.py, tile_sim.py: checks and scans. scripts/campaign.py: campaign runner (see campaigns/README.md).
- webui/main.py (FastAPI) and webui/index.html (viewer and UI); webui/campaign_api.py + campaigns.html (campaign control page); webui/runview.py + strain.html + run.html (run viewer, strain, graphs).
- tests/: TestClient and parse tests that run from the repo root without a server or solver (tests/README.md); tests/NOTES_loop_grade.md: the loop-grade rule and decisions.

## Current work (facts about the present state, 2026-11-20)
- The shell is the accepted fold reference. A shell comparison is optional per run. The final yarn-level run is scored for similarity out of 100.
- The first shell success was made with commit 8346645 (13 Nov) with E 3700, mu 0.49, contact 40, perturb 0.20, outer 40, L-BFGS 400, curl 0.5, tension 17 cN, creep 3 h over 6 h, shell on, hold 0.05 x EA/R, tiles 3, Ne 12, CPI 22, WPI 16. It ended unsettled but folded very close to the real fabric. Reproduction runs A then B are in ~/old_runs on the server.
- Plan order: shell-only experiments first (no yarn mesh), validate against real fabrics, an ML loop on random then proposed patterns to find fold rules, then the yarn mesh with migration and creep, then parameter sensitivity (tasks 7-11).
- Loop-length migration (relax_state.migrate): yarn slip is modelled as capstan-style rest-length transfer between the two segments at a contact node, capped at 20 % of a segment per outer round. From a uniform start it is zero by the slip rule (0 transfers in the probes; about 1e-19 per wale in the fidelity runs, which were unsettled and had no relaxed_l0.csv). After a relaxation it triggered at some nodes of one unsettled saved state (651 of 5858 nodes, largest 6.6e-4 mm per pass, 2.7 % of the cap). How large it becomes in a real run is not yet measured.
- Starting loop-length field: --loop-grade a, --loop-grade-target boundary|tm|both (supersedes the planned --tm-boost), --loop-grade-min-strength; sectioned rule, works with and without the shell, off by default; implemented and tested without the solver, never run in the solver. The direction of the effect is attributed to Basiri 2010 and Amanatides 2022 via the other AI's CHANGES.txt and is NOT independently verified. The magnitudes are UNSOURCED (list in tests/NOTES_loop_grade.md). Creep does not conserve the total rest length and can partly erase the grade.
- Status: the yarn-only simulation does not yet rival the shell-hold simulation. Folding appears only when the start is seeded. The other AI's "roll" result is a single 12x12 case whose control run had not converged. Treat it as unverified.
- Target case: --ne 12 --cpi 22 --wpi 16 --tiles 3 with K's 20-row x 16-column matrix of 1/2 blocks (campaigns/pattern1_matrix.txt).
