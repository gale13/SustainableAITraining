#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,60}$ ]]; then
  echo 'Usage: bash scripts/start-tmux.sh RUN_ID' >&2
  exit 2
fi
run_id=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base=${PROFILE_CONDA_ROOT:-"$HOME/miniconda3"}
tmux_bin="$base/envs/wt103-profile/bin/tmux"
if [[ ! -x "$tmux_bin" ]]; then
  echo "Install the project environment first: bash scripts/setup-native.sh" >&2
  exit 1
fi
mkdir -p "$root/results"
# A separate socket inherits this job's GPU allocation when starting a new session.
if ! "$tmux_bin" -L "$run_id" has-session -t '=train' 2>/dev/null; then
  "$tmux_bin" -L "$run_id" new-session -d -s train -c "$root" \
    bash -c 'set -o pipefail; bash scripts/start-native.sh "$1" 2>&1 | tee -a "results/$1.tmux.log"' bash "$run_id"
fi
printf 'Attach: %q -L %q attach -t train\n' "$tmux_bin" "$run_id"
printf 'Log: %s/results/%s.tmux.log\n' "$root" "$run_id"
