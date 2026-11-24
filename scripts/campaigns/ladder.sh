#!/bin/bash
# Timing ladder driver (runs on the server). Usage: nohup bash scripts/campaigns/ladder.sh <stamp> &
# Calls run_case.sh from the same folder as this script. Paths can be overridden from the environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LADDER_ROOT:-$HOME/auxetic_project}"
LOG_DIR="${LADDER_LOG_DIR:-$ROOT/logs}"
CASE="$SCRIPT_DIR/run_case.sh"

STAMP="$1"
[ -n "$STAMP" ] || { echo "usage: bash scripts/campaigns/ladder.sh <stamp>" >&2; exit 2; }
cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"
bash "$CASE" "$STAMP" warmup 2 || { echo "STOP: warm-up build failed" > "$LOG_DIR/ladder_${STAMP}_ABORT"; exit 10; }
bash "$CASE" "$STAMP" t4x4 2
bash "$CASE" "$STAMP" t8x8 4
bash "$CASE" "$STAMP" t12x12 6
# two 4x4 copies at once
bash "$CASE" "$STAMP" pairA 2 & A=$!
bash "$CASE" "$STAMP" pairB 2 & B=$!
wait $A $B
echo "ALL_DONE $(date +%s)" > "$LOG_DIR/ladder_${STAMP}_ALLDONE"
