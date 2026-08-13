#!/usr/bin/env bash
# Root-relative launcher for the academic HSTU/SAFA training path.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG.gin [--gin_bindings=...]" >&2
  exit 2
fi

config="$1"
shift
if [[ "$config" != /* ]]; then
  config="$repo_root/$config"
fi
if [[ ! -f "$config" ]]; then
  echo "config not found: $config" >&2
  exit 2
fi

: "${GR_DATA_ROOT:=$repo_root/tmp}"
: "${GR_EXPS_ROOT:=$repo_root/exps}"
: "${GR_CKPTS_ROOT:=$repo_root/ckpts}"
: "${GR_WANDB_ENABLED:=0}"
: "${GR_CONDA_ENV:=gr}"
GR_DATA_ROOT="$(realpath -m "$GR_DATA_ROOT")"
GR_EXPS_ROOT="$(realpath -m "$GR_EXPS_ROOT")"
GR_CKPTS_ROOT="$(realpath -m "$GR_CKPTS_ROOT")"
export GR_DATA_ROOT GR_EXPS_ROOT GR_CKPTS_ROOT
export GR_SOURCE_ROOT="$repo_root"

if [[ -f "$repo_root/SOURCE_MANIFEST_SHA256" ]]; then
  expected_manifest="${GR_EXPECTED_SOURCE_MANIFEST:-$(<"$repo_root/SOURCE_MANIFEST_SHA256")}"
  verifier_python="$(command -v python3)"
  provenance_exports="$($verifier_python "$repo_root/scripts/snapshot.py" verify \
    "$repo_root" --expected-manifest "$expected_manifest" --shell)"
  eval "$provenance_exports"
fi

if [[ -n "${GR_PYTHON:-}" ]]; then
  python_bin="$GR_PYTHON"
elif [[ -n "$GR_CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$GR_CONDA_ENV"
  python_bin="$CONDA_PREFIX/bin/python"
else
  python_bin="$(command -v python3)"
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter is not executable: $python_bin" >&2
  exit 1
fi

mkdir -p "$GR_DATA_ROOT" "$GR_EXPS_ROOT" "$GR_CKPTS_ROOT"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

seed="${GR_SEED:-42}"
arguments=("$@")
for ((argument_index = 0; argument_index < ${#arguments[@]}; argument_index++)); do
  argument="${arguments[$argument_index]}"
  case "$argument" in
    --gin_bindings=train_fn.random_seed=*)
      seed="${argument#--gin_bindings=train_fn.random_seed=}"
      ;;
    --gin_bindings)
      if ((argument_index + 1 < ${#arguments[@]})); then
        next_argument="${arguments[$((argument_index + 1))]}"
        if [[ "$next_argument" == train_fn.random_seed=* ]]; then
          seed="${next_argument#train_fn.random_seed=}"
        fi
      fi
      ;;
  esac
done
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "random seed must be a non-negative integer: $seed" >&2
  exit 2
fi
export PYTHONHASHSEED="$seed"

gin_string() {
  "$python_bin" -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

wandb_bindings=()
if [[ "$GR_WANDB_ENABLED" == "1" ]]; then
  wandb_bindings+=("--gin_bindings=train_fn.wandb_enabled=True")
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    wandb_bindings+=("--gin_bindings=train_fn.wandb_project=$(gin_string "$WANDB_PROJECT")")
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    wandb_bindings+=("--gin_bindings=train_fn.wandb_entity=$(gin_string "$WANDB_ENTITY")")
  fi
  if [[ -n "${WANDB_MODE:-}" ]]; then
    wandb_bindings+=("--gin_bindings=train_fn.wandb_mode=$(gin_string "$WANDB_MODE")")
  fi
elif [[ "$GR_WANDB_ENABLED" != "0" ]]; then
  echo "GR_WANDB_ENABLED must be 0 or 1" >&2
  exit 2
fi

run_cwd="$repo_root"
cleanup_run_cwd=0
if [[ "$(realpath -m "$GR_DATA_ROOT")" != "$(realpath -m "$repo_root/tmp")" ]]; then
  run_parent="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
  run_cwd="$(mktemp -d "$run_parent/gr-research-run.XXXXXX")"
  ln -s "$GR_DATA_ROOT" "$run_cwd/tmp"
  cleanup_run_cwd=1
fi
cleanup() {
  if [[ "$cleanup_run_cwd" == "1" ]]; then
    rm -rf "$run_cwd"
  fi
}
trap cleanup EXIT

cd "$run_cwd"
"$python_bin" "$repo_root/main.py" \
  --gin_config_file="$config" \
  "${wandb_bindings[@]}" \
  "$@"
