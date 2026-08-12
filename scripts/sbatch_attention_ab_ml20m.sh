#!/usr/bin/env bash
# One-snapshot HSTU-vs-per-head-softmax controls on ML-20M.
# Submit with: bash scripts/submit_attention_experiments.sh ab

#SBATCH --job-name=hstu-attn-ab
#SBATCH --partition=h200
#SBATCH --qos=h200_mrs_shared
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=/checkpoints/ngocbh/longhstu/logs/slurm/attn_ab_%A_%a.out
#SBATCH --error=/checkpoints/ngocbh/longhstu/logs/slurm/attn_ab_%A_%a.err

set -euo pipefail

repo_root="${GR_CODE_SNAPSHOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$repo_root"

configs=(
  "configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin"
  "configs/ml-20m/hstu-softmax-per-head-20m.gin"
  "configs/ml-20m/hstu-softmax-per-head-unscaled-20m.gin"
  "configs/ml-20m/hstu-softmax-canonical-per-head-20m.gin"
)
names=(
  "ml20m-hstu-rerun-seed42"
  "ml20m-softmax-per-head-seed42"
  "ml20m-softmax-per-head-unscaled-seed42"
  "ml20m-softmax-canonical-seed42"
)

task_id="${SLURM_ARRAY_TASK_ID}"
config="${configs[$task_id]}"
run_name="${names[$task_id]}"
array_job_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
restart_count="${SLURM_RESTART_COUNT:-0}"
run_id="${run_name}-j${array_job_id}-t${task_id}-r${restart_count}"

export GR_DATA_ROOT=/checkpoints/ngocbh/longhstu/datasets
export GR_CKPTS_ROOT=/checkpoints/ngocbh/longhstu/checkpoints
export GR_EXPS_ROOT=/checkpoints/ngocbh/longhstu/exps
export GR_WANDB_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${WANDB_PROJECT:=gr}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be exported when submitting this job}"
export WANDB_PROJECT WANDB_API_KEY

mkdir -p \
  "$GR_CKPTS_ROOT" \
  "$GR_EXPS_ROOT" \
  /checkpoints/ngocbh/longhstu/logs/slurm

master_port=$((20000 + (SLURM_JOB_ID + task_id) % 20000))

echo "job=$SLURM_JOB_ID task=$task_id host=$(hostname) config=$config"
echo "source_snapshot=$repo_root"
if [[ -d .git ]]; then
  git status --short
else
  cat GIT_COMMIT GIT_STATUS
fi
sha256sum \
  generative_recommenders/research/modeling/sequential/hstu.py \
  generative_recommenders/research/modeling/sequential/encoder_utils.py \
  "$config"

bash scripts/train.sh research "$config" \
  --master_port="$master_port" \
  "--gin_bindings=train_fn.random_seed=42" \
  "--gin_bindings=train_fn.exp_suffix='$run_id'" \
  "--gin_bindings=train_fn.wandb_run_name='$run_id'" \
  "--gin_bindings=train_fn.wandb_tags=['ml-20m','attention-ab','seed-42']" \
  "--gin_bindings=train_fn.save_last_only=True"
