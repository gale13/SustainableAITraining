#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,60}$ ]]; then
  echo 'Usage: bash scripts/start-nohup.sh RUN_ID' >&2
  exit 2
fi
run_id=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir="$root/results/$run_id"
log="$run_dir/nohup.log"
mkdir -p "$run_dir"

# The background controller inherits fd 9, retaining this lock across bash -> Python exec.
exec 9>>"$run_dir/.nohup.lock"
if ! flock -n 9; then
  echo "Cannot acquire the launch lock for $run_id. Check $log before retrying." >&2
  exit 1
fi
# Also respect a run or export started through another project entry point.
if ! flock -n "$run_dir/.lock" true; then
  echo "The run directory is in use: $run_dir" >&2
  exit 1
fi

node=$(hostname)
printf '\n[%s] starting %s on %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$run_id" "$node" >>"$log"
nohup bash "$root/scripts/start-native.sh" "$run_id" </dev/null >>"$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$run_dir/nohup.pid.tmp"
mv -f "$run_dir/nohup.pid.tmp" "$run_dir/nohup.pid"
printf 'Node: %s   PID: %s\nLog: %s\nStatus: %s/status.json\n' "$node" "$pid" "$log" "$run_dir"

# Surface immediate failures (including a missing environment) in the submitting terminal.
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  if wait "$pid"; then
    echo "Controller exited successfully; check status.json."
  else
    code=$?
    tail -n 12 "$log" >&2
    exit "$code"
  fi
fi
