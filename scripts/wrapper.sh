#!/usr/bin/env bash
# Slurm wrapper: `sbatch scripts/wrapper.sh <command...>`.
# Used to submit scripts/train.sh (or any command) to the cluster.
#
# Examples:
#   sbatch scripts/wrapper.sh bash scripts/train.sh research configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin
#   sbatch --gres=gpu:1 scripts/wrapper.sh bash scripts/train.sh research configs/ml-1m/...
#
# Override SBATCH defaults from the CLI, e.g.:
#   sbatch --gres=gpu:8 --partition=highprio scripts/wrapper.sh ...

#SBATCH --job-name=hstu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=REQUEUE,FAIL,TIME_LIMIT
#SBATCH --output=logs/slurm/o_%A.out
#SBATCH --error=logs/slurm/e_%A.err
#SBATCH --partition=h200
#SBATCH --requeue
#SBATCH --gres=gpu:h200:4

set -euo pipefail

# Always run from the directory sbatch was invoked in (the repo root by convention).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"

mkdir -p logs/slurm

echo "=== slurm job ==="
date
hostname
echo "cwd: $(pwd)"
echo "job_id: ${SLURM_JOB_ID:-N/A}  array_task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "nodelist: ${SLURM_JOB_NODELIST:-N/A}  gpus: ${SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-N/A}}"

# nvidia-smi may not exist on login nodes — don't fail the job if it's missing.
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

if [[ $# -eq 0 ]]; then
  echo "wrapper.sh: no command supplied" >&2
  exit 2
fi

echo "=== command ==="
printf '%q ' "$@"
echo

# Run the user-supplied command directly — no eval, no requoting.
"$@"
status=$?

echo "=== done (exit=$status) ==="
date
exit $status
