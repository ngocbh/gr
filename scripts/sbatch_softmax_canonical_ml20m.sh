#!/usr/bin/env bash
# Canonical qk/sqrt(d) + relative-bias control for the ML-20M attention A/B.
# Submit with: bash scripts/submit_attention_experiments.sh softmax-canonical

#SBATCH --job-name=softmax-canonical
#SBATCH --partition=h200
#SBATCH --qos=h200_mrs_shared
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/checkpoints/ngocbh/longhstu/logs/slurm/softmax_canonical_%j.out
#SBATCH --error=/checkpoints/ngocbh/longhstu/logs/slurm/softmax_canonical_%j.err

set -euo pipefail

repo_root="${GR_CODE_SNAPSHOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$repo_root"

config="configs/ml-20m/hstu-softmax-canonical-per-head-20m.gin"
run_name="ml20m-softmax-canonical-seed42"
restart_count="${SLURM_RESTART_COUNT:-0}"
run_id="${run_name}-j${SLURM_JOB_ID}-r${restart_count}"

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

master_port=$((20000 + SLURM_JOB_ID % 20000))

echo "job=$SLURM_JOB_ID host=$(hostname) config=$config"
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
  "--gin_bindings=train_fn.wandb_tags=['ml-20m','attention-ab','canonical-bias','seed-42']" \
  "--gin_bindings=train_fn.save_last_only=True"
