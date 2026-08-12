#!/usr/bin/env bash
# Interactive / foreground trainer for DEBUGGING the research HSTU.
#
# Runs training in a single process (no mp.spawn) via debug_train.py, so:
#   * output is live (unbuffered) and logging.info is visible,
#   * `breakpoint()` / pdb work, exceptions give a real traceback,
#   * per-layer probe metrics are on by default.
#
# If you're not already on a GPU, it grabs an interactive H200 for you (srun).
#
# Usage:
#   bash scripts/train_debug.sh                 # ml-20m probe config, wandb off
#   QUICK=1 bash scripts/train_debug.sh         # 1-epoch fast smoke run
#   WANDB=1 bash scripts/train_debug.sh         # also stream to wandb
#   EPOCHS=3 PROBE_INTERVAL=10 bash scripts/train_debug.sh
#   CONFIG=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin QUICK=1 \
#       bash scripts/train_debug.sh             # ml-1m loads fast; probe forced on
#   PROBE=0 bash scripts/train_debug.sh         # disable the probe
#   # pass any extra gin bindings straight through:
#   bash scripts/train_debug.sh --gin_bindings=create_data_loader.num_workers=0
#
# To use pdb: add `breakpoint()` in the code, then run this (num_workers=0 if you
# want to step into the dataset).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- grab an interactive GPU if there isn't one here (once) ---
if ! nvidia-smi >/dev/null 2>&1; then
  if [[ "${_GR_DEBUG_ALLOCATED:-0}" != "1" ]]; then
    echo "[train_debug] no GPU on this host; requesting an interactive H200 via srun..." >&2
    export _GR_DEBUG_ALLOCATED=1
    exec srun --partition=h200 --qos=h200_mrs_shared --gres=gpu:h200:1 \
      --cpus-per-task=16 --mem=64G --time=04:00:00 --export=ALL --pty \
      bash "$REPO_ROOT/scripts/train_debug.sh" "$@"
  fi
  echo "[train_debug] WARNING: still no GPU after allocation attempt." >&2
fi

# --- wandb creds etc. ---
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# --- debug defaults (all overridable via env) ---
: "${CONFIG:=configs/ml-20m/hstu-probe-dynamics.gin}"
: "${GR_DATA_ROOT:=/checkpoints/ngocbh/longhstu/datasets}"
# separate *_debug roots so debug runs never clobber the sbatch run's outputs
: "${GR_CKPTS_ROOT:=/checkpoints/ngocbh/longhstu/checkpoints_debug}"
: "${GR_EXPS_ROOT:=/checkpoints/ngocbh/longhstu/exps_debug}"
: "${WANDB:=0}"
: "${PROBE:=1}"
: "${GR_CONDA_ENV:=gr}"
export GR_DATA_ROOT GR_CKPTS_ROOT GR_EXPS_ROOT

# drop stale x2p proxy (would hang any wandb egress); harmless when wandb is off
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY all_proxy X2P_PROXY_URL X2P_PROXY 2>/dev/null || true

# --- conda ---
if [[ "${CONDA_DEFAULT_ENV:-}" != "$GR_CONDA_ENV" ]]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/ngocbh/miniconda3)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$GR_CONDA_ENV"
fi
export TMPDIR="${TMPDIR:-/home/ngocbh/tmp/pip}"; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$GR_CKPTS_ROOT" "$GR_EXPS_ROOT"

if [[ ! -f "$CONFIG" ]]; then echo "[train_debug] config not found: $CONFIG" >&2; exit 1; fi

# --- assemble gin bindings ---
bindings=()
# probe on/off (forced here so it works with ANY base config, e.g. ml-1m)
if [[ "$PROBE" == "1" ]]; then
  bindings+=("--gin_bindings=train_fn.probe_enabled=True")
else
  bindings+=("--gin_bindings=train_fn.probe_enabled=False")
fi
bindings+=("--gin_bindings=train_fn.save_last_only=True")
# wandb
if [[ "$WANDB" == "1" ]]; then
  bindings+=("--gin_bindings=train_fn.wandb_enabled=True")
  [[ -n "${WANDB_PROJECT:-}" ]] && bindings+=("--gin_bindings=train_fn.wandb_project='${WANDB_PROJECT}'")
  bindings+=("--gin_bindings=train_fn.wandb_run_name='debug-${USER:-run}'")
else
  bindings+=("--gin_bindings=train_fn.wandb_enabled=False")
fi
# QUICK smoke run: short + frequent eval + finer probe
if [[ "${QUICK:-0}" == "1" ]]; then
  : "${EPOCHS:=1}"; : "${PROBE_INTERVAL:=5}"
  bindings+=("--gin_bindings=train_fn.eval_interval=20")
  bindings+=("--gin_bindings=train_fn.partial_eval_num_iters=4")
fi
[[ -n "${EPOCHS:-}" ]]         && bindings+=("--gin_bindings=train_fn.num_epochs=${EPOCHS}")
[[ -n "${PROBE_INTERVAL:-}" ]] && bindings+=("--gin_bindings=train_fn.probe_interval=${PROBE_INTERVAL}")

MASTER_PORT=$(( 20000 + ($$ % 20000) ))

echo "[train_debug] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "[train_debug] config=$CONFIG  wandb=$WANDB  probe=$PROBE  quick=${QUICK:-0}  epochs=${EPOCHS:-<config>}"
echo "[train_debug] data=$GR_DATA_ROOT  ckpts=$GR_CKPTS_ROOT  exps=$GR_EXPS_ROOT  master_port=$MASTER_PORT"
echo "[train_debug] bindings: ${bindings[*]}"

# -u = unbuffered stdout so you see logs/prints live; single process (pdb-friendly).
exec python -u -m generative_recommenders.research.trainer.debug_train \
  --gin_config_file="$CONFIG" \
  --master_port="$MASTER_PORT" \
  "${bindings[@]}" \
  "$@"
