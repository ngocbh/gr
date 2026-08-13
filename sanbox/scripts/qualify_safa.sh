#!/usr/bin/env bash
# Run the exact-equivalence suite and short paired training smokes on one H200.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GR_EXPECTED_SOURCE_MANIFEST:?missing pinned snapshot manifest}"
: "${GR_DATA_ROOT:?GR_DATA_ROOT must point to preprocessed MovieLens data}"
: "${GR_EXPS_ROOT:?GR_EXPS_ROOT must be set}"
: "${GR_CKPTS_ROOT:?GR_CKPTS_ROOT must be set}"

actual_qos="${SLURM_JOB_QOS:-}"
if [[ -z "$actual_qos" && -n "${SLURM_JOB_ID:-}" ]]; then
  actual_qos="$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* QOS=\([^ ]*\).*/\1/p')"
fi
if [[ "$actual_qos" == "h200_dev" ]]; then
  echo "refusing qualification on h200_dev" >&2
  exit 1
fi
if [[ "$actual_qos" != "h200_mrs_shared" ]]; then
  echo "qualification requires QoS h200_mrs_shared" >&2
  exit 1
fi

verifier_python="$(command -v python3)"
provenance_exports="$($verifier_python "$repo_root/scripts/snapshot.py" verify \
  "$repo_root" --expected-manifest "$GR_EXPECTED_SOURCE_MANIFEST" --shell)"
eval "$provenance_exports"
unset GR_EXPECTED_EXPERIMENT_CONFIG_SHA256 \
  GR_CONFIG_IDENTITY_ONLY GR_CONFIG_IDENTITY_OUTPUT

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
"$python_bin" -m unittest -v \
  generative_recommenders.research.modeling.sequential.safa_test
"$python_bin" -m unittest discover -v -s scripts/tests -p "test_*.py"
"$python_bin" -m unittest discover -v -s tests -p "test_*.py"
"$python_bin" scripts/audit_safa_ab.py --dataset all

qualification_exps="$GR_EXPS_ROOT/qualification/$GR_SOURCE_MANIFEST"
qualification_ckpts="$GR_CKPTS_ROOT/qualification/$GR_SOURCE_MANIFEST"
mkdir -p "$qualification_exps" "$qualification_ckpts"

declare -A expected_config_identities
identity_root="$qualification_exps/config-identities"
mkdir -p "$identity_root"
for dataset in ml-1m ml-20m; do
  expected_identity=""
  for mode in hstu safa; do
    if [[ "$mode" == "hstu" ]]; then
      config="configs/$dataset/hstu-matched-sampled-softmax-n128-large-final.gin"
    else
      config="configs/$dataset/safa-sampled-softmax-n128-large-final.gin"
    fi
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

for dataset in ml-1m ml-20m; do
  for mode in hstu safa; do
    if [[ "$mode" == "hstu" ]]; then
      config="configs/$dataset/hstu-matched-sampled-softmax-n128-large-final.gin"
    else
      config="configs/$dataset/safa-sampled-softmax-n128-large-final.gin"
    fi
    run_name="qualification-$dataset-$mode-seed42-${SLURM_JOB_ID:-local}"
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
  echo "experiment_config_ml-1m=${expected_config_identities[ml-1m]}"
  echo "experiment_config_ml-20m=${expected_config_identities[ml-20m]}"
} >"$temporary_marker"
mv -f "$temporary_marker" "$marker"
echo "qualification_marker=$marker"
