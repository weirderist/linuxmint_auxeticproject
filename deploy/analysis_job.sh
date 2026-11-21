#!/usr/bin/env bash
# deploy/analysis_job.sh - the scheduled analysis job of a pure-shell campaign (scripts/shell_analysis_job.py). Run ON THE SERVER by K, for example:
#   ssh server "bash ~/auxetic_project/deploy/analysis_job.sh install"     add the @reboot line to K's crontab (once; a second install changes nothing)
#   ssh server "bash ~/auxetic_project/deploy/analysis_job.sh start"       start the job now, detached (does nothing if it already runs)
#   ssh server "bash ~/auxetic_project/deploy/analysis_job.sh status"      is it running, the crontab line, the last log lines
#   ssh server "bash ~/auxetic_project/deploy/analysis_job.sh stop"        stop the running job (an analysis in progress is recorded as interrupted and repeated at the next start)
#   ssh server "bash ~/auxetic_project/deploy/analysis_job.sh uninstall"   remove the @reboot line (stop it as well to disable it completely)
# The job reads the campaign and writes only <campaign>/analysis/scheduled/ and its log. It runs at nice 19 and ionice class 3 (idle). It does not touch the runner, which uses its frozen copy.
set -u
CAMP="${2:-shell1}"
PROJ="$HOME/auxetic_project"; PY="$PROJ/venv/bin/python"; JOB="$PROJ/scripts/shell_analysis_job.py"; LOG="$HOME/shell_logs/analysis_${CAMP}.log"
TAG="shell_analysis_job.py ${CAMP} "
LINE="@reboot sleep 210 && /usr/bin/setsid /usr/bin/ionice -c 3 /usr/bin/nice -n 19 $PY $JOB ${CAMP} --poll-s 600 >> $LOG 2>&1 &"
running() { pgrep -f -- "$JOB ${CAMP} --poll-s" >/dev/null; }
case "${1:-status}" in
  install)
    if crontab -l 2>/dev/null | grep -F -q -- "$TAG"; then echo "already installed:"; crontab -l | grep -F -- "$TAG"
    else (crontab -l 2>/dev/null; echo "$LINE") | crontab - && echo "installed: $LINE"; fi ;;
  uninstall)
    if crontab -l 2>/dev/null | grep -F -q -- "$TAG"; then crontab -l | grep -v -F -- "$TAG" | crontab - && echo "removed the @reboot line (other crontab lines kept)"; else echo "no @reboot line for $CAMP"; fi ;;
  start)
    mkdir -p "$HOME/shell_logs"
    if running; then echo "already running:"; pgrep -af -- "$JOB ${CAMP} --poll-s"
    else setsid ionice -c 3 nice -n 19 "$PY" "$JOB" "$CAMP" --poll-s 600 >> "$LOG" 2>&1 < /dev/null & sleep 2; pgrep -af -- "$JOB ${CAMP} --poll-s" || { echo "did not start; see $LOG"; exit 1; }; fi ;;
  stop)
    if running; then pkill -TERM -f -- "$JOB ${CAMP} --poll-s"; sleep 2; running && echo "still running" || echo "stopped"; else echo "not running"; fi ;;
  status)
    if running; then echo "running:"; pgrep -af -- "$JOB ${CAMP} --poll-s"; else echo "not running"; fi
    echo "crontab:"; crontab -l 2>/dev/null | grep -F -- "$TAG" || echo "  (no @reboot line)"
    echo "last log lines ($LOG):"; tail -n 8 "$LOG" 2>/dev/null || echo "  (no log yet)" ;;
  *) echo "usage: bash deploy/analysis_job.sh install|start|status|stop|uninstall [campaign]"; exit 2 ;;
esac
