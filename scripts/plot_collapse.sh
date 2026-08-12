#!/usr/bin/env bash
# Plot representation collapse (Hyper-Connections Fig. 3 style) from a trained
# HSTU checkpoint: cos(h^i, h^{i+1}) vs layer index, median + 5-95 pct band.
#
# Grabs an interactive H200 if you're not on one, then runs a forward over a few
# eval batches to snapshot the per-token cosine distribution at each layer.
#
# Usage:
#   bash scripts/plot_collapse.sh                      # vanilla ml-20m run (auto)
#   OUTPUT=/path/fig.png NUM_BATCHES=16 bash scripts/plot_collapse.sh
#   CKPT=/path/to/_last.pt CONFIG=configs/ml-20m/hstu-probe-dynamics.gin \
#       LABEL="Residual (HSTU)" bash scripts/plot_collapse.sh
#   # overlay two runs (pass raw flags through; --run is "label,config,ckpt"):
#   bash scripts/plot_collapse.sh \
#     --run "Residual,configs/ml-20m/hstu-probe-dynamics.gin,/.../vanilla_last.pt" \
#     --run "mHC,configs/ml-1m/hstu-mhc-sampled-softmax-n128-large-final.gin,/.../mhc_last.pt"

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- grab an interactive GPU if there isn't one here (once) ---
if ! nvidia-smi >/dev/null 2>&1; then
  if [[ "${_GR_ALLOCATED:-0}" != "1" ]]; then
    echo "[plot_collapse] no GPU here; requesting an interactive H200 via srun..." >&2
    export _GR_ALLOCATED=1
    exec srun --partition=h200 --qos=h200_mrs_shared --gres=gpu:h200:1 \
      --cpus-per-task=8 --mem=64G --time=00:30:00 --export=ALL --pty \
      bash "$REPO_ROOT/scripts/plot_collapse.sh" "$@"
  fi
fi

# --- env / conda ---
: "${GR_DATA_ROOT:=/checkpoints/ngocbh/longhstu/datasets}"
: "${GR_CONDA_ENV:=gr}"
: "${CONFIG:=configs/ml-20m/hstu-probe-dynamics.gin}"
: "${CKPT:=$(ls -t /checkpoints/ngocbh/longhstu/checkpoints/ml-20m-l200/*_last.pt 2>/dev/null | head -1)}"
: "${LABEL:=Residual (vanilla HSTU)}"
: "${OUTPUT:=/checkpoints/ngocbh/longhstu/plots/representation_collapse.png}"
: "${NUM_BATCHES:=8}"
export GR_DATA_ROOT

if [[ "${CONDA_DEFAULT_ENV:-}" != "$GR_CONDA_ENV" ]]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/ngocbh/miniconda3)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$GR_CONDA_ENV"
fi
export TMPDIR="${TMPDIR:-/home/ngocbh/tmp/pip}"; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# This is a headless analysis job: neutralize any stray breakpoint() left in the
# model/trainer (e.g. from interactive debugging) so it can't hang on no TTY.
export PYTHONBREAKPOINT=0
mkdir -p "$(dirname "$OUTPUT")"

echo "[plot_collapse] host=$(hostname) config=$CONFIG"
echo "[plot_collapse] ckpt=$CKPT"
echo "[plot_collapse] output=$OUTPUT num_batches=$NUM_BATCHES"

# If the caller passed explicit --run/flags, forward them and skip the defaults.
if [[ "$*" == *"--run"* || "$*" == *"--gin_config_file"* ]]; then
  exec python -u -m generative_recommenders.research.scripts.plot_representation_collapse \
    --num_batches="$NUM_BATCHES" --output="$OUTPUT" "$@"
else
  exec python -u -m generative_recommenders.research.scripts.plot_representation_collapse \
    --gin_config_file="$CONFIG" \
    --checkpoint="$CKPT" \
    --label="$LABEL" \
    --num_batches="$NUM_BATCHES" \
    --output="$OUTPUT" \
    "$@"
fi
