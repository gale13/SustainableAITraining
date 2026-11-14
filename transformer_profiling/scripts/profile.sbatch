#!/usr/bin/env bash
#SBATCH --job-name=wt103-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the project directory}"
exec bash scripts/start-native.sh "${1:?Pass a run ID after the script name}"
