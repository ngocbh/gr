#!/usr/bin/env bash
# ML-1M pairwise Taylor and additive-moment mechanism controls.
# Submit with: bash scripts/submit_attention_experiments.sh moments

#SBATCH --job-name=hstu-moments
#SBATCH --partition=h200
#SBATCH --qos=h200_mrs_shared
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/checkpoints/ngocbh/longhstu/logs/slurm/hstu_moments_%A_%a.out
#SBATCH --error=/checkpoints/ngocbh/longhstu/logs/slurm/hstu_moments_%A_%a.err

set -euo pipefail

repo_root="${GR_CODE_SNAPSHOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$repo_root"

configs=(
  "configs/ml-1m/hstu-norab-large-final.gin"
  "configs/ml-1m/hstu-taylor1-large-final.gin"
  "configs/ml-1m/hstu-taylor2-large-final.gin"
  "configs/ml-1m/hstu-taylor1-norab-large-final.gin"
  "configs/ml-1m/hstu-additive-dot-large-final.gin"
)
names=(
  "ml1m-hstu-norab-seed42"
  "ml1m-hstu-taylor1-seed42"
  "ml1m-hstu-taylor2-seed42"
  "ml1m-hstu-taylor1-norab-seed42"
  "ml1m-hstu-additive-dot-seed42"
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

master_port=$((24000 + (SLURM_JOB_ID + task_id) % 16000))

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
  "--gin_bindings=train_fn.wandb_tags=['ml-1m','moment-controls','seed-42']" \
  "--gin_bindings=train_fn.save_last_only=True"
