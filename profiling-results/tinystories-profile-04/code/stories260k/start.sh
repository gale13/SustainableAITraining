#!/usr/bin/env bash
set -euo pipefail
root=/path/to/experiments/tinystories-profile-04
cd "$root/code"
export PYTHONPATH="$root/code:/path/to/experiments/stories260k-smoke.uMUVU05u/libs"
export PYTHONNOUSERSITE=1
nohup /home/USER/miniconda3/envs/wt103-profile/bin/python -u -m stories260k.launch "$root" >>"$root/dispatcher.log" 2>&1 </dev/null &
echo $! >"$root/dispatcher.pid"
echo "Dispatcher started"
