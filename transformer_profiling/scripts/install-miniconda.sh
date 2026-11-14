#!/usr/bin/env bash
set -euo pipefail
prefix=${PROFILE_CONDA_ROOT:-"$HOME/miniconda3"}
if [[ -x "$prefix/bin/conda" ]]; then
  "$prefix/bin/conda" --version
  echo "Using existing Conda installation: $prefix"
  exit 0
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo "Run this installer on the Linux x86-64 server." >&2
  exit 1
fi
name=Miniconda3-py311_25.9.1-3-Linux-x86_64.sh
sha=e331037f492848e68764b0c430912b2417b977d654706009ac8eb475e03ae0a2
if [[ -n ${PROFILE_MINICONDA_INSTALLER:-} ]]; then
  installer="$PROFILE_MINICONDA_INSTALLER"
else
  temp=$(mktemp -d)
  trap 'rm -f -- "$temp/$name"; rmdir -- "$temp"' EXIT
  installer="$temp/$name"
  url="https://repo.anaconda.com/miniconda/$name"
  if command -v curl >/dev/null; then
    curl --fail --location --retry 3 --output "$installer" "$url"
  else
    wget -O "$installer" "$url"
  fi
fi
printf '%s  %s\n' "$sha" "$installer" | sha256sum --check --strict
bash "$installer" -b -p "$prefix"
"$prefix/bin/conda" --version
