#!/usr/bin/env bash
# Paired seed array: task 2k is HSTU and task 2k+1 is SAFA for the same seed.
#SBATCH --job-name=safa-ab
#SBATCH --partition=h200
#SBATCH --qos=h200_mrs_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-5

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GR_EXPECTED_SOURCE_MANIFEST:?missing pinned snapshot manifest}"
: "${GR_DATASET:?GR_DATASET must be ml-1m or ml-20m}"
: "${GR_DATA_ROOT:?GR_DATA_ROOT must be set}"
: "${GR_EXPS_ROOT:?GR_EXPS_ROOT must be set}"
: "${GR_CKPTS_ROOT:?GR_CKPTS_ROOT must be set}"
: "${SLURM_ARRAY_TASK_ID:?this wrapper must run as a SLURM array}"

actual_qos="${SLURM_JOB_QOS:-}"
if [[ -z "$actual_qos" && -n "${SLURM_JOB_ID:-}" ]]; then
  actual_qos="$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* QOS=\([^ ]*\).*/\1/p')"
fi
if [[ "$actual_qos" == "h200_dev" ]]; then
  echo "refusing experiment on h200_dev" >&2
  exit 1
fi
if [[ "$actual_qos" != "h200_mrs_shared" ]]; then
  echo "experiment requires QoS h200_mrs_shared" >&2
  exit 1
fi
if [[ "$GR_DATASET" != "ml-1m" && "$GR_DATASET" != "ml-20m" ]]; then
  echo "unsupported dataset: $GR_DATASET" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 5 )); then
  echo "array task must be in [0, 5]" >&2
  exit 2
fi

verifier_python="$(command -v python3)"
provenance_exports="$($verifier_python "$repo_root/scripts/snapshot.py" verify \
  "$repo_root" --expected-manifest "$GR_EXPECTED_SOURCE_MANIFEST" --shell)"
eval "$provenance_exports"

: "${GR_QUALIFICATION_ROOT:=$GR_EXPS_ROOT/qualifications}"
qualification_marker="$GR_QUALIFICATION_ROOT/$GR_SOURCE_MANIFEST.passed"
if [[ ! -f "$qualification_marker" || -L "$qualification_marker" ]]; then
  echo "missing qualification marker for this snapshot: $qualification_marker" >&2
  exit 1
fi
if ! grep -Fxq "source_manifest=$GR_SOURCE_MANIFEST" "$qualification_marker"; then
  echo "qualification marker does not match this snapshot" >&2
  exit 1
fi
for expected_line in \
  "status=passed" \
  "qualification_scope=preflight_only" \
  "source_commit=$GR_SOURCE_COMMIT" \
  "source_tree=$GR_SOURCE_TREE"; do
  if ! grep -Fxq "$expected_line" "$qualification_marker"; then
    echo "qualification marker is incomplete: $expected_line" >&2
    exit 1
  fi
done
expected_experiment_config_sha256="$(sed -n \
  "s/^experiment_config_${GR_DATASET}=//p" "$qualification_marker")"
if [[ ! "$expected_experiment_config_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "qualification marker has no valid experiment config identity" >&2
  exit 1
fi
export GR_EXPECTED_EXPERIMENT_CONFIG_SHA256="$expected_experiment_config_sha256"
unset GR_CONFIG_IDENTITY_ONLY GR_CONFIG_IDENTITY_OUTPUT

seeds=(42 43 44)
seed="${seeds[$((SLURM_ARRAY_TASK_ID / 2))]}"
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
  mode="hstu"
  config="configs/$GR_DATASET/hstu-matched-sampled-softmax-n128-large-final.gin"
else
  mode="safa"
  config="configs/$GR_DATASET/safa-sampled-softmax-n128-large-final.gin"
fi

run_name="safa-ab-$GR_DATASET-$mode-seed$seed-${SLURM_ARRAY_JOB_ID:-job}"
group_name="safa-ab-$GR_DATASET-${GR_SOURCE_MANIFEST:0:12}"

GR_WANDB_ENABLED=1
export GR_WANDB_ENABLED GR_SEED="$seed" GR_ATTENTION_MODE="$mode"
export GR_EXPERIMENT_NAME="$run_name"
export WANDB_NAME="$run_name"
export WANDB_RUN_GROUP="$group_name"
export WANDB_TAGS="safa-ab,$GR_DATASET,$mode,seed-$seed,parameter-matched"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-60}"
# Batch jobs must not inherit a session-scoped login-node proxy endpoint.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
  X2P_PROXY X2P_PROXY_URL 2>/dev/null || true

exec /bin/bash "$repo_root/scripts/train.sh" "$config" \
  "--gin_bindings=train_fn.random_seed=$seed"
