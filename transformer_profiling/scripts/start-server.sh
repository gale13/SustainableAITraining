#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,60}$ ]]; then
  echo 'Usage: bash scripts/start-server.sh RUN_ID (1-61 letters, numbers, dots, underscores or hyphens)' >&2
  exit 2
fi
run_id=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${PROFILE_IMAGE:-sustainable-transformer:cu128-v2}
image_id=$(docker image inspect --format '{{.Id}}' "$image")
mkdir -p "$root"/{data,cache,results}
docker run -d --init --restart unless-stopped --name "wt103-$run_id" \
  --gpus all --shm-size 2g --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  -e HOME=/tmp -e "PROFILE_RUN_ID=$run_id" -e "PROFILE_IMAGE_ID=$image_id" \
  -v "$root/data:/data" -v "$root/cache:/cache" -v "$root/results:/results" \
  -v "$root/configs:/app/configs:ro" \
  "$image_id" python -m profiling.pipeline
