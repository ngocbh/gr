#!/usr/bin/env bash
#SBATCH --job-name=safa-qualify
#SBATCH --partition=h200
#SBATCH --qos=h200_mrs_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

: "${GR_CODE_SNAPSHOT:?missing immutable source snapshot path}"
repo_root="$(realpath -e "$GR_CODE_SNAPSHOT")"
exec /bin/bash "$repo_root/scripts/qualify_safa.sh"
