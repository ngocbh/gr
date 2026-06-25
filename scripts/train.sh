#!/usr/bin/env bash
# Train entry-point for HSTU (research + dlrm_v3).
#
# Usage:
#   bash scripts/train.sh [framework] [config-or-dataset] [extra args...]
#
#   framework: "research" (default) or "dlrm_v3"
#
#   research config:  path to a .gin under configs/, e.g.
#       configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin
#
#   dlrm_v3 dataset:  key from SUPPORTED_CONFIGS in train_ranker.py
#       (debug, kuairand-1k, movielens-1m, movielens-20m, ...)
#
# Examples:
#   bash scripts/train.sh research configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin
#   bash scripts/train.sh dlrm_v3 movielens-1m
#
# Reads .env for GR_*/WANDB_* defaults. Per-run overrides:
#   GR_WANDB_ENABLED=0 bash scripts/train.sh research configs/ml-1m/...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- load .env (export every variable) ---
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# --- defaults if .env didn't supply them ---
: "${GR_CONDA_ENV:=gr}"
: "${GR_DATA_ROOT:=$REPO_ROOT/tmp}"
: "${GR_CKPTS_ROOT:=$REPO_ROOT/ckpts}"
: "${GR_EXPS_ROOT:=$REPO_ROOT/exps}"
: "${GR_WANDB_ENABLED:=0}"
: "${WANDB_MODE:=online}"
: "${DEBUG:=0}"

# DEBUG=1 forces wandb off everywhere. GR_WANDB_ENABLED is the single switch read
# by both the research branch (gin bindings + curl check below) and the dlrm_v3
# branch (train_ranker.py reads GR_WANDB_ENABLED from the env), so clearing it
# here disables wandb for both frameworks.
if [[ "$DEBUG" == "1" ]]; then
  echo "[train.sh] DEBUG=1 -> wandb disabled" >&2
  GR_WANDB_ENABLED=0
fi

export GR_DATA_ROOT GR_CKPTS_ROOT GR_EXPS_ROOT
export WANDB_PROJECT WANDB_ENTITY WANDB_MODE WANDB_API_KEY WANDB_USERNAME

# --- drop the inherited per-session proxy; compute nodes have direct egress ---
# An sbatch job freezes the submitting session's dynamic x2p proxy port
# (https_proxy=http://10.0.2.2:<port>). Once that session ends the port is dead
# and any outbound call (wandb) hangs forever, stalling rank 0 and the whole job.
# The compute nodes reach the internet directly, so unset the proxy and go direct
# — this is both reliable and keeps wandb online.
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY all_proxy X2P_PROXY_URL X2P_PROXY 2>/dev/null || true

# Only fall back to offline if even direct egress to wandb fails.
if [[ "$GR_WANDB_ENABLED" == "1" && "${WANDB_MODE}" == "online" ]]; then
  if ! curl -sS -m 8 -o /dev/null https://api.wandb.ai 2>/dev/null; then
    echo "[train.sh] WARNING: api.wandb.ai unreachable even without proxy." >&2
    echo "[train.sh] Falling back to WANDB_MODE=offline; sync later with: wandb sync <run_dir>/wandb/offline-run-*" >&2
    WANDB_MODE=offline
    export WANDB_MODE
  fi
fi
# Never let wandb.init() block the job indefinitely.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-60}"

mkdir -p "$GR_DATA_ROOT" "$GR_CKPTS_ROOT" "$GR_EXPS_ROOT"

# --- activate conda env ---
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "$GR_CONDA_ENV" ]]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/ngocbh/miniconda3)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$GR_CONDA_ENV"
fi

# Workaround for small /tmp on this host (pip / torch dynamo extracts).
export TMPDIR="${TMPDIR:-/home/ngocbh/tmp/pip}"
mkdir -p "$TMPDIR"

framework="${1:-research}"
shift || true

case "$framework" in
  research)
    config_file="${1:-configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin}"
    shift || true

    if [[ ! -f "$config_file" ]]; then
      echo "[train.sh] config not found: $config_file" >&2
      exit 1
    fi

    gin_overrides=()
    if [[ "$GR_WANDB_ENABLED" == "1" ]]; then
      gin_overrides+=("--gin_bindings=train_fn.wandb_enabled=True")
      if [[ -n "${WANDB_PROJECT:-}" ]]; then
        gin_overrides+=("--gin_bindings=train_fn.wandb_project='${WANDB_PROJECT}'")
      fi
      if [[ -n "${WANDB_ENTITY:-}" ]]; then
        gin_overrides+=("--gin_bindings=train_fn.wandb_entity='${WANDB_ENTITY}'")
      fi
      if [[ -n "${WANDB_MODE:-}" ]]; then
        gin_overrides+=("--gin_bindings=train_fn.wandb_mode='${WANDB_MODE}'")
      fi
    fi

    echo "[train.sh] framework=research config=$config_file"
    echo "[train.sh] GR_EXPS_ROOT=$GR_EXPS_ROOT GR_CKPTS_ROOT=$GR_CKPTS_ROOT"
    echo "[train.sh] wandb_enabled=$GR_WANDB_ENABLED project=$WANDB_PROJECT mode=$WANDB_MODE"

    exec python3 main.py \
      --gin_config_file="$config_file" \
      "${gin_overrides[@]}" \
      "$@"
    ;;

  dlrm_v3)
    dataset="${1:-movielens-1m}"
    shift || true
    mode="${MODE:-train-eval}"

    echo "[train.sh] framework=dlrm_v3 dataset=$dataset mode=$mode WORLD_SIZE=${WORLD_SIZE:-1}"
    exec python3 -m generative_recommenders.dlrm_v3.train.train_ranker \
      --dataset "$dataset" \
      --mode "$mode" \
      "$@"
    ;;

  *)
    echo "[train.sh] unknown framework: $framework (expected: research|dlrm_v3)" >&2
    exit 1
    ;;
esac
