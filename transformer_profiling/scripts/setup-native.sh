#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
base=${PROFILE_CONDA_ROOT:-"$HOME/miniconda3"}
prefix="$base/envs/wt103-profile"
conda="$base/bin/conda"
channel=${PROFILE_CONDA_CHANNEL:-conda-forge}
if [[ ! -x "$conda" ]]; then
  echo "Install Miniconda first: bash scripts/install-miniconda.sh" >&2
  exit 1
fi
if [[ ! -f "$prefix/conda-meta/history" ]]; then
  "$conda" create -y --override-channels -c "$channel" -p "$prefix" python=3.11 pip tmux
elif [[ ! -x "$prefix/bin/tmux" || ! -x "$prefix/bin/pip" ]] ||
     ! "$prefix/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
  "$conda" install -y --override-channels -c "$channel" -p "$prefix" python=3.11 pip tmux
fi
py="$prefix/bin/python"
export PYTHONNOUSERSITE=1
if [[ -n ${PROFILE_WHEELHOUSE:-} ]]; then
  "$py" -m pip install --no-index --find-links "$PROFILE_WHEELHOUSE" 'torch==2.7.1+cu128' -r requirements.txt
else
  "$py" -m pip install 'torch==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
  "$py" -m pip install -r requirements.txt
fi
"$py" -m pip check
"$py" -c 'import torch; print("torch", torch.__version__, "CUDA build", torch.version.cuda); assert torch.version.cuda == "12.8"'
"$prefix/bin/tmux" -V
