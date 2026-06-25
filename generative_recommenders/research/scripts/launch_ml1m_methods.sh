#!/usr/bin/env bash
# Research-HSTU ml-1m verification: vanilla vs NeuTRENO vs AttnRes.
#   3 runs (one per method), full 101-epoch config, 1 GPU each, wandb on.
#
# Run name encodes the method: ml1m_<method>  (vanilla | neutreno | attnres)
#
# Usage: bash generative_recommenders/research/scripts/launch_ml1m_methods.sh
#
# Submits via the repo-root SLURM wrapper (scripts/wrapper.sh) + train.sh research path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CFG_DIR="configs/ml-1m"
declare -A CONFIGS=(
  [vanilla]="$CFG_DIR/hstu-sampled-softmax-n128-large-final.gin"
  [neutreno]="$CFG_DIR/hstu-neutreno-sampled-softmax-n128-large-final.gin"
  [attnres]="$CFG_DIR/hstu-attnres-sampled-softmax-n128-large-final.gin"
)

mkdir -p logs
: > logs/ml1m_methods_jobids.txt

submit() {
  local method="$1" config="$2"
  local name="ml1m_${method}"
  if [[ ! -f "$config" ]]; then
    echo "[launch] ERROR: config not found: $config" >&2
    exit 1
  fi
  # Randomize master_port so co-located jobs on one node don't collide on the
  # default 12355 (EADDRINUSE). Range stays in the dynamic/ephemeral band.
  local port=$(( 20000 + RANDOM % 20000 ))
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh research "$config" \
    --master_port="$port" \
    --gin_bindings="train_fn.wandb_run_name='${name}'")
  echo "$jid  $name  config=$config"
  echo "$jid $name $config" >> logs/ml1m_methods_jobids.txt
}

for method in vanilla neutreno attnres; do
  submit "$method" "${CONFIGS[$method]}"
done

echo "=== 3 ml-1m runs submitted. job ids in logs/ml1m_methods_jobids.txt ==="
