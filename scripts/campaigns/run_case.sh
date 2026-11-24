#!/bin/bash
# One timing case: build (timed) then relax (timed, per-line timestamps) with a RAM/stuck watchdog.
# Usage: run_case.sh <stamp> <label> <tiles>
# Plain jersey (all 1), no shell, --ne 12 --cpi 22 --wpi 16, --outer 20 --maxiter 150, no stall stop (--stall-tol 0 is the default).
# E, mu, k-contact are UNSOURCED (copied from the smoke test); this is a timing test, not a physical result.
# Paths (same variables as ladder.sh): LADDER_ROOT moves the project root (venv, scripts, smoke_test/ outputs);
# LADDER_LOG_DIR moves the per-run logs. Defaults are $HOME/auxetic_project and $ROOT/logs.
ROOT="${LADDER_ROOT:-$HOME/auxetic_project}"
LOG_DIR="${LADDER_LOG_DIR:-$ROOT/logs}"
STAMP="$1"; LABEL="$2"; TILES="$3"
P="$ROOT/venv/bin/python"
S="$ROOT/smoke_test/${STAMP}_${LABEL}"
L="$LOG_DIR/ladder_${STAMP}_${LABEL}"
RAM_LIMIT_KB=$((5*1024*1024)); STUCK_S=1200
cd "$ROOT" || exit 1
mkdir -p "$S" "$LOG_DIR"
echo "label $LABEL tiles $TILES start $(date +%s.%N)" > "$L.status"

# ---- build ----
/usr/bin/time -v -o "$L.build.time" "$P" -u scripts/yarn_model/build_knit.py --matrix "1 1/1 1" --tiles "$TILES" \
  --ne 12 --cpi 22 --wpi 16 --out "$S" --name "ladder-$LABEL" --no-tube > "$L.build.log" 2>&1
BRC=$?
echo "build_exit $BRC" >> "$L.status"
if [ $BRC -ne 0 ]; then echo "BUILD_FAILED" >> "$L.status"; exit 10; fi
[ "$LABEL" = "warmup" ] && { echo "DONE" >> "$L.status"; exit 0; }

# ---- relax ----
(
  /usr/bin/time -v -o "$L.relax.time" "$P" -u scripts/yarn_model/relax_job.py --folder "$S" --E 3700 --mu 0.49 \
    --k-contact 50017.10385104558 --perturb 0.02 --outer 20 --maxiter 150 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$L.relax.log"
) &
SUB=$!
sleep 3
while kill -0 "$SUB" 2>/dev/null; do
  PID=$(pgrep -f "^$P -u scripts/yarn_model/relax_job.py --folder $S " | head -1)
  if [ -n "$PID" ]; then
    RSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null)
    if [ -n "$RSS" ] && [ "$RSS" -gt "$RAM_LIMIT_KB" ]; then
      echo "KILLED_RAM pid $PID rss_kb $RSS" >> "$L.status"; kill -TERM "$PID"
    fi
    AGE=$(( $(date +%s) - $(stat -c %Y "$L.relax.log") ))
    if [ "$AGE" -gt "$STUCK_S" ]; then
      echo "KILLED_STUCK pid $PID no_output_s $AGE" >> "$L.status"; kill -TERM "$PID"
    fi
  fi
  sleep 5
done
wait "$SUB"
echo "end $(date +%s.%N)" >> "$L.status"
echo "DONE" >> "$L.status"
