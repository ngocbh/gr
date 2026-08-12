#!/usr/bin/env bash
# Parameter-matched Signed Additive Feature Attention screen on ML-1M.
# Submit with: bash scripts/submit_attention_experiments.sh signed-additive

#SBATCH --job-name=safa-ml1m
#SBATCH --partition=h200
#SBATCH --qos=h200_mrs_shared
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/checkpoints/ngocbh/longhstu/logs/slurm/safa_ml1m_%A_%a.out
#SBATCH --error=/checkpoints/ngocbh/longhstu/logs/slurm/safa_ml1m_%A_%a.err

set -euo pipefail

: "${GR_CODE_SNAPSHOT:?submit through scripts/submit_attention_experiments.sh signed-additive}"
: "${GR_SNAPSHOT_MANIFEST_SHA256:?missing expected snapshot manifest SHA256}"
repo_root="$GR_CODE_SNAPSHOT"
if [[ ! -d "$repo_root" || -L "$repo_root" ]]; then
  echo "snapshot root must be a non-symlink directory" >&2
  exit 1
fi
cd -- "$repo_root"

if [[ ! "$GR_SNAPSHOT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid expected snapshot manifest SHA256" >&2
  exit 1
fi
if [[ ! -f SOURCE_SHA256SUMS || -L SOURCE_SHA256SUMS ]]; then
  echo "missing snapshot manifest: $repo_root/SOURCE_SHA256SUMS" >&2
  exit 1
fi
if [[ ! -f SOURCE_TREE_INVENTORY || -L SOURCE_TREE_INVENTORY ]]; then
  echo "missing snapshot tree inventory: $repo_root/SOURCE_TREE_INVENTORY" >&2
  exit 1
fi
read -r snapshot_manifest_sha256 _ < <(sha256sum SOURCE_SHA256SUMS)
if [[ "$snapshot_manifest_sha256" != "$GR_SNAPSHOT_MANIFEST_SHA256" ]]; then
  echo "snapshot manifest SHA256 mismatch" >&2
  exit 1
fi
unsupported_node="$(
  LC_ALL=C find . -mindepth 1 ! -type f ! -type d -print -quit
)"
if [[ -n "$unsupported_node" ]]; then
  echo "snapshot contains a symlink or special node" >&2
  exit 1
fi
if ! sha256sum --check --strict --quiet SOURCE_SHA256SUMS; then
  echo "snapshot file checksum validation failed" >&2
  exit 1
fi
generated_inventory="$(mktemp /tmp/gr-snapshot-inventory.XXXXXX)"
if ! /bin/bash scripts/snapshot_tree_inventory.sh . >"$generated_inventory"; then
  rm -f -- "$generated_inventory"
  echo "could not regenerate snapshot tree inventory" >&2
  exit 1
fi
if ! cmp -s -- SOURCE_TREE_INVENTORY "$generated_inventory"; then
  rm -f -- "$generated_inventory"
  echo "snapshot tree inventory mismatch" >&2
  exit 1
fi
rm -f -- "$generated_inventory"
echo "snapshot_manifest_sha256=$snapshot_manifest_sha256"

configs=(
  "configs/ml-1m/hstu-signed-additive-identity-large-final.gin"
  "configs/ml-1m/hstu-signed-additive-tanh-large-final.gin"
  "configs/ml-1m/hstu-signed-additive-abs-tanh-large-final.gin"
  "configs/ml-1m/hstu-signed-additive-abs-coefficient-oracle-large-final.gin"
)
names=(
  "ml1m-safa-identity"
  "ml1m-safa-tanh"
  "ml1m-safa-abs-tanh"
  "ml1m-safa-abscoef-oracle"
)
tags=(
  "identity"
  "signed-tanh"
  "abs-tanh-features"
  "absolute-coefficient-oracle"
)

task_id="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
if [[ ! "$task_id" =~ ^[0-9]+$ || "$task_id" -gt 11 ]]; then
  echo "invalid array task id: $task_id" >&2
  exit 2
fi
config_idx=$((task_id / 3))
seed=$((42 + task_id % 3))
config="${configs[$config_idx]}"
run_name="${names[$config_idx]}-seed${seed}"
mode_tag="${tags[$config_idx]}"
array_job_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
restart_count="${SLURM_RESTART_COUNT:-0}"
# A requeue restarts deterministically and writes to a distinct run.
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

echo "job=$SLURM_JOB_ID task=$task_id host=$(hostname) config=$config seed=$seed"
echo "screen=signed-additive-feature-attention parameter_match=exact"
echo "restart_policy=clean-deterministic-retry restart=$restart_count resume=false run_id=$run_id"
echo "source_snapshot=$repo_root"
cat GIT_COMMIT GIT_STATUS
sha256sum \
  generative_recommenders/research/modeling/sequential/hstu.py \
  generative_recommenders/research/modeling/sequential/encoder_utils.py \
  "$config"

bash scripts/train.sh research "$config" \
  --master_port="$master_port" \
  "--gin_bindings=train_fn.random_seed=$seed" \
  "--gin_bindings=train_fn.exp_suffix='$run_id'" \
  "--gin_bindings=train_fn.wandb_run_name='$run_id'" \
  "--gin_bindings=train_fn.wandb_tags=['ml-1m','signed-additive','$mode_tag','seed-$seed','parameter-matched']" \
  "--gin_bindings=train_fn.save_last_only=True"
