#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,60}$ ]]; then
  echo 'Usage: bash scripts/start-native.sh RUN_ID' >&2
  exit 2
fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
base=${PROFILE_CONDA_ROOT:-"$HOME/miniconda3"}
prefix="$base/envs/wt103-profile"
if [[ ! -x "$prefix/bin/python" || ! -f "$prefix/conda-meta/history" ]]; then
  echo "Create the Conda environment first: bash scripts/setup-native.sh" >&2
  exit 1
fi
# Activate inside the session so launching also works from a fresh Termius login.
source "$base/etc/profile.d/conda.sh"
conda activate "$prefix"
export PROFILE_NATIVE_ROOT="$root" PROFILE_RUN_ID="$1"
export HF_HOME="$root/cache/huggingface" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
mkdir -p data results "$HF_HOME"
exec "$CONDA_PREFIX/bin/python" -m profiling.pipeline
