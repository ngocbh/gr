#!/usr/bin/env bash
# Run the exact-equivalence suite and short paired training smokes on one H200.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GR_EXPECTED_SOURCE_MANIFEST:?missing pinned snapshot manifest}"
: "${GR_DATA_ROOT:?GR_DATA_ROOT must point to preprocessed experiment data}"
: "${GR_EXPS_ROOT:?GR_EXPS_ROOT must be set}"
: "${GR_CKPTS_ROOT:?GR_CKPTS_ROOT must be set}"

: "${SLURM_JOB_ID:?qualification must run inside SLURM}"
reported_qos="${SLURM_JOB_QOS:-}"
reported_partition="${SLURM_JOB_PARTITION:-}"
reported_restart_count="${SLURM_RESTART_COUNT:-}"
scheduler_record="$(scontrol show job "$SLURM_JOB_ID" -o)"
actual_qos="$(sed -n 's/.* QOS=\([^ ]*\).*/\1/p' <<<"$scheduler_record")"
actual_partition="$(sed -n \
  's/.* Partition=\([^ ]*\).*/\1/p' <<<"$scheduler_record")"
actual_restart_count="$(sed -n \
  's/.* Restarts=\([^ ]*\).*/\1/p' <<<"$scheduler_record")"
if [[ -z "$actual_qos" || -z "$actual_partition" ]]; then
  echo "could not resolve qualification QoS and partition from SLURM" >&2
  exit 1
fi
if [[ ! "$actual_restart_count" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "could not resolve qualification restart count from SLURM" >&2
  exit 1
fi
if [[ -n "$reported_restart_count" && \
      "$reported_restart_count" != "$actual_restart_count" ]]; then
  echo "qualification SLURM_RESTART_COUNT disagrees with scheduler state" >&2
  exit 1
fi
if [[ -n "$reported_qos" && "$reported_qos" != "$actual_qos" ]]; then
  echo "qualification SLURM_JOB_QOS disagrees with scheduler state" >&2
  exit 1
fi
if [[ -n "$reported_partition" && \
      "$reported_partition" != "$actual_partition" ]]; then
  echo "qualification SLURM_JOB_PARTITION disagrees with scheduler state" >&2
  exit 1
fi
if [[ "$actual_qos" != "h200_dev" ]]; then
  echo "qualification requires QoS h200_dev" >&2
  exit 1
fi
if [[ "$actual_partition" != "h200" ]]; then
  echo "qualification requires partition h200" >&2
  exit 1
fi
export SLURM_JOB_QOS="$actual_qos"
export SLURM_JOB_PARTITION="$actual_partition"
export SLURM_RESTART_COUNT="$actual_restart_count"

verifier_python="$(command -v python3)"
provenance_exports="$($verifier_python "$repo_root/scripts/snapshot.py" verify \
  "$repo_root" --expected-manifest "$GR_EXPECTED_SOURCE_MANIFEST" --shell)"
eval "$provenance_exports"
unset GR_EXPECTED_EXPERIMENT_CONFIG_SHA256 \
  GR_CONFIG_IDENTITY_ONLY GR_CONFIG_IDENTITY_OUTPUT
# Keep the canonical identity independent of interactive W&B mode overrides.
unset WANDB_MODE WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID

: "${GR_CONDA_ENV:=gr}"
if [[ -n "${GR_PYTHON:-}" ]]; then
  python_bin="$GR_PYTHON"
else
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$GR_CONDA_ENV"
  python_bin="$CONDA_PREFIX/bin/python"
fi
export GR_PYTHON="$python_bin"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$gpu_count" != "1" ]]; then
  echo "qualification requires exactly one visible GPU, found $gpu_count" >&2
  exit 1
fi

cd "$repo_root"
amazon_books_data="$GR_DATA_ROOT/amzn_books/sasrec_format.csv"
amazon_books_sha256="b58804a08f835f0d85cb2d50628166670ee96c5808d622434ca57d2a48cdf491"
if [[ ! -f "$amazon_books_data" || -L "$amazon_books_data" ]]; then
  echo "Amazon Books data must be a regular file: $amazon_books_data" >&2
  exit 1
fi
actual_amazon_books_sha256="$(sha256sum "$amazon_books_data" | cut -d' ' -f1)"
if [[ "$actual_amazon_books_sha256" != "$amazon_books_sha256" ]]; then
  echo "Amazon Books data checksum does not match the frozen experiment data" >&2
  exit 1
fi
"$python_bin" -m unittest -v \
  generative_recommenders.research.modeling.sequential.safa_test
"$python_bin" -m unittest discover -v -s scripts/tests -p "test_*.py"
"$python_bin" -m unittest discover -v -s tests -p "test_*.py"
"$python_bin" scripts/audit_safa_ab.py --dataset all

datasets=(amzn-books ml-1m ml-20m)
config_for() {
  local dataset="$1"
  local mode="$2"
  local num_negatives=128
  if [[ "$dataset" == "amzn-books" ]]; then
    num_negatives=512
  fi
  if [[ "$mode" == "hstu" ]]; then
    printf 'configs/%s/hstu-matched-sampled-softmax-n%s-large-final.gin\n' \
      "$dataset" "$num_negatives"
  else
    printf 'configs/%s/safa-sampled-softmax-n%s-large-final.gin\n' \
      "$dataset" "$num_negatives"
  fi
}

qualification_exps="$GR_EXPS_ROOT/qualification/$GR_SOURCE_MANIFEST"
qualification_ckpts="$GR_CKPTS_ROOT/qualification/$GR_SOURCE_MANIFEST"
mkdir -p "$qualification_exps" "$qualification_ckpts"

declare -A expected_config_identities
identity_root="$qualification_exps/config-identities"
mkdir -p "$identity_root"
for dataset in "${datasets[@]}"; do
  expected_identity=""
  for mode in hstu safa; do
    config="$(config_for "$dataset" "$mode")"
    identity_file="$identity_root/$dataset-$mode.json"
    GR_CONFIG_IDENTITY_ONLY=1 \
    GR_CONFIG_IDENTITY_OUTPUT="$identity_file" \
    GR_WANDB_ENABLED=1 \
    GR_SEED=42 \
      /bin/bash "$repo_root/scripts/train.sh" "$config" \
      --gin_bindings=train_fn.random_seed=42
    identity="$($python_bin -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["experiment_config_sha256"])' \
      "$identity_file")"
    if [[ ! "$identity" =~ ^[0-9a-f]{64}$ ]]; then
      echo "invalid experiment config identity for $dataset/$mode" >&2
      exit 1
    fi
    if [[ -n "$expected_identity" && "$identity" != "$expected_identity" ]]; then
      echo "normalized experiment configs differ for $dataset" >&2
      exit 1
    fi
    expected_identity="$identity"
  done
  expected_config_identities["$dataset"]="$expected_identity"
done

for dataset in "${datasets[@]}"; do
  for mode in hstu safa; do
    config="$(config_for "$dataset" "$mode")"
    run_name="qualification-$dataset-$mode-seed42-$SLURM_JOB_ID-r$actual_restart_count"
    GR_WANDB_ENABLED=0 \
    GR_SEED=42 \
    GR_EXPS_ROOT="$qualification_exps" \
    GR_CKPTS_ROOT="$qualification_ckpts" \
      /bin/bash "$repo_root/scripts/train.sh" "$config" \
      --gin_bindings=train_fn.random_seed=42 \
      --gin_bindings=train_fn.num_epochs=1 \
      --gin_bindings=train_fn.max_train_batches_per_epoch=2 \
      --gin_bindings=train_fn.max_eval_batches_per_epoch=2 \
      --gin_bindings=train_fn.save_final_checkpoint=False \
      --gin_bindings=train_fn.save_ckpt_every_n=1000000 \
      "--gin_bindings=train_fn.experiment_name='$run_name'"
  done
done

: "${GR_QUALIFICATION_ROOT:=$GR_EXPS_ROOT/qualifications}"
mkdir -p "$GR_QUALIFICATION_ROOT"
marker="$GR_QUALIFICATION_ROOT/$GR_SOURCE_MANIFEST.passed"
temporary_marker="$marker.tmp.${SLURM_JOB_ID:-$$}"
umask 077
{
  echo "status=passed"
  echo "qualification_scope=preflight_only"
  echo "source_commit=$GR_SOURCE_COMMIT"
  echo "source_tree=$GR_SOURCE_TREE"
  echo "source_manifest=$GR_SOURCE_MANIFEST"
  echo "qualification_job_id=$SLURM_JOB_ID"
  echo "qualification_job_qos=$actual_qos"
  echo "qualification_job_partition=$actual_partition"
  echo "qualification_restart_count=$actual_restart_count"
  echo "dataset_amzn-books_sha256=$amazon_books_sha256"
  echo "experiment_config_amzn-books=${expected_config_identities[amzn-books]}"
  echo "experiment_config_ml-1m=${expected_config_identities[ml-1m]}"
  echo "experiment_config_ml-20m=${expected_config_identities[ml-20m]}"
} >"$temporary_marker"
mv -f "$temporary_marker" "$marker"
echo "qualification_marker=$marker"
