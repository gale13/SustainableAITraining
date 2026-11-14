#!/usr/bin/env bash
set -euo pipefail
# Run from the project root. Paths with spaces are supported.
PROFILE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_IMAGE="${PROFILE_IMAGE:-sustainable-transformer:cu128-v2}"
mkdir -p "$PROFILE_ROOT/data" "$PROFILE_ROOT/cache" "$PROFILE_ROOT/results"
exec docker run --rm --init --gpus all \
  --user "$(id -u):$(id -g)" --shm-size 2g \
  --cap-drop ALL --security-opt no-new-privileges \
  -e HOME=/tmp -e HF_HOME=/cache/huggingface \
  -v "$PROFILE_ROOT/data:/data" \
  -v "$PROFILE_ROOT/cache:/cache" \
  -v "$PROFILE_ROOT/results:/results" \
  -v "$PROFILE_ROOT/configs:/app/configs:ro" \
  "$PROFILE_IMAGE" "$@"
