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

: "${GR_CODE_SNAPSHOT:?missing immutable source snapshot path}"
repo_root="$(realpath -e "$GR_CODE_SNAPSHOT")"
: "${GR_EXPECTED_SOURCE_MANIFEST:?missing pinned snapshot manifest}"
: "${GR_DATASET:?GR_DATASET must be ml-1m or ml-20m}"
: "${GR_DATA_ROOT:?GR_DATA_ROOT must be set}"
: "${GR_EXPS_ROOT:?GR_EXPS_ROOT must be set}"
: "${GR_CKPTS_ROOT:?GR_CKPTS_ROOT must be set}"
: "${SLURM_ARRAY_TASK_ID:?this wrapper must run as a SLURM array}"
: "${SLURM_ARRAY_JOB_ID:?missing SLURM array job ID}"
: "${SLURM_JOB_ID:?missing SLURM job ID}"

for job_id in "$SLURM_ARRAY_JOB_ID" "$SLURM_JOB_ID"; do
  if [[ ! "$job_id" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLURM job IDs must be positive decimal integers: $job_id" >&2
    exit 2
  fi
done
if [[ ! "$SLURM_ARRAY_TASK_ID" =~ ^[0-5]$ ]]; then
  echo "array task must be an integer in [0, 5]" >&2
  exit 2
fi

reported_qos="${SLURM_JOB_QOS:-}"
reported_partition="${SLURM_JOB_PARTITION:-}"
reported_restart_count="${SLURM_RESTART_COUNT:-}"
scheduler_record="$(scontrol show job \
  "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" -o)"
actual_job_id="$(sed -n 's/^JobId=\([^ ]*\).*/\1/p' <<<"$scheduler_record")"
actual_array_job_id="$(sed -n \
  's/.* ArrayJobId=\([^ ]*\).*/\1/p' <<<"$scheduler_record")"
actual_array_task_id="$(sed -n \
  's/.* ArrayTaskId=\([^ ]*\).*/\1/p' <<<"$scheduler_record")"
actual_qos="$(sed -n 's/.* QOS=\([^ ]*\).*/\1/p' <<<"$scheduler_record")"
actual_partition="$(sed -n \
  's/.* Partition=\([^ ]*\).*/\1/p' <<<"$scheduler_record")"
actual_restart_count="$(sed -n \
  's/.* Restarts=\([^ ]*\).*/\1/p' <<<"$scheduler_record")"
if [[ -z "$actual_job_id" || -z "$actual_array_job_id" || \
      -z "$actual_array_task_id" || -z "$actual_qos" || \
      -z "$actual_partition" || -z "$actual_restart_count" ]]; then
  echo "could not resolve actual SLURM provenance" >&2
  exit 1
fi
if [[ "$SLURM_JOB_ID" != "$actual_job_id" || \
      "$SLURM_ARRAY_JOB_ID" != "$actual_array_job_id" || \
      "$SLURM_ARRAY_TASK_ID" != "$actual_array_task_id" ]]; then
  echo "SLURM environment IDs disagree with scheduler state" >&2
  exit 1
fi
if [[ -n "$reported_qos" && "$reported_qos" != "$actual_qos" ]]; then
  echo "SLURM_JOB_QOS disagrees with scheduler state" >&2
  exit 1
fi
if [[ -n "$reported_partition" && "$reported_partition" != "$actual_partition" ]]; then
  echo "SLURM_JOB_PARTITION disagrees with scheduler state" >&2
  exit 1
fi
if [[ ! "$actual_restart_count" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "scheduler returned an invalid restart count" >&2
  exit 1
fi
if [[ -n "$reported_restart_count" ]]; then
  if [[ ! "$reported_restart_count" =~ ^(0|[1-9][0-9]*)$ || \
        "$reported_restart_count" != "$actual_restart_count" ]]; then
    echo "SLURM_RESTART_COUNT disagrees with scheduler state" >&2
    exit 1
  fi
fi
if [[ "$actual_qos" == "h200_dev" ]]; then
  echo "refusing experiment on h200_dev" >&2
  exit 1
fi
if [[ "$actual_qos" != "h200_mrs_shared" ]]; then
  echo "experiment requires QoS h200_mrs_shared" >&2
  exit 1
fi
if [[ "$actual_partition" != "h200" ]]; then
  echo "experiment requires partition h200" >&2
  exit 1
fi
export SLURM_JOB_QOS="$actual_qos"
export SLURM_JOB_PARTITION="$actual_partition"
export SLURM_RESTART_COUNT="$actual_restart_count"
if [[ "$GR_DATASET" != "ml-1m" && "$GR_DATASET" != "ml-20m" ]]; then
  echo "unsupported dataset: $GR_DATASET" >&2
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
  "source_tree=$GR_SOURCE_TREE" \
  "qualification_job_qos=h200_mrs_shared" \
  "qualification_job_partition=h200"; do
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

run_name="safa-ab-$GR_DATASET-$mode-seed$seed-$SLURM_ARRAY_JOB_ID-r$actual_restart_count"
group_name="safa-ab-$GR_DATASET-${GR_SOURCE_MANIFEST:0:12}"

GR_WANDB_ENABLED=1
GR_REQUIRE_WANDB=1
GR_REQUIRE_SLURM_PROVENANCE=1
export GR_WANDB_ENABLED GR_REQUIRE_WANDB GR_REQUIRE_SLURM_PROVENANCE
export GR_SEED="$seed" GR_ATTENTION_MODE="$mode"
export GR_EXPERIMENT_NAME="$run_name"
export WANDB_NAME="$run_name"
export WANDB_RUN_GROUP="$group_name"
export WANDB_TAGS="safa-ab,$GR_DATASET,$mode,seed-$seed,parameter-matched"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-60}"
# A required run must create an online record; do not inherit session overrides.
unset WANDB_MODE WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID
# Batch jobs must not inherit a session-scoped login-node proxy endpoint.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy \
  X2P_PROXY X2P_PROXY_URL 2>/dev/null || true

exec /bin/bash "$repo_root/scripts/train.sh" "$config" \
  "--gin_bindings=train_fn.random_seed=$seed"
