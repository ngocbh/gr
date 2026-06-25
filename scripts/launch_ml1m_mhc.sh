#!/usr/bin/env bash
# Launch 3 research HSTU-mHC runs on ml-1m (full training), seeds 1/2/3.
#   - wandb on (reads .env), 1 GPU each, distinct master_port + exp_suffix per seed
#   - exp_suffix=s<seed> keeps each run's logs/ckpts in its own directory
#
# Usage: bash scripts/launch_ml1m_mhc.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/ml-1m/hstu-mhc-sampled-softmax-n128-large-final.gin"
SEEDS=(1 2 3)

mkdir -p logs
: > logs/ml1m_mhc_jobids.txt

submit() {
  local seed="$1"
  local name="ml1m_mhc_s${seed}"
  local port=$((29500 + seed))
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh research "$CONFIG" \
    --master_port="$port" \
    --gin_bindings=train_fn.random_seed="$seed" \
    --gin_bindings=train_fn.exp_suffix="\"s${seed}\"")
  echo "$jid  $name  seed=$seed  port=$port"
  echo "$jid $name $seed" >> logs/ml1m_mhc_jobids.txt
}

echo "=== research ml-1m mHC: seeds ${SEEDS[*]} ==="
for s in "${SEEDS[@]}"; do
  submit "$s"
done
echo "=== submitted. job ids in logs/ml1m_mhc_jobids.txt ==="
