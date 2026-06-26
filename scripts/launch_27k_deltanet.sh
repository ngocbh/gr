#!/usr/bin/env bash
# Launch STU_DELTANET (windowed HSTU short-attn + gated-delta long memory) at
# max_seq_len=16384 on kuairand-27k: window in {128, 256, 512}, seeds 1..10,
# 1 epoch, 1 GPU each.
#   window=128 -> kr27k_deltanet_w128_s<seed>
#   window=256 -> kr27k_deltanet_w256_s<seed>
#   window=512 -> kr27k_deltanet_w512_s<seed>
#
# Direct A/B against scripts/launch_27k_stu_pytorch_window.sh (same windows/seeds,
# STU_PYTORCH short-only): isolates the effect of adding the long-term memory read.
#
# Requires KuaiRand-27K/data/processed_seqs.csv (see scripts/preprocess_27k.sh).
#
# Usage: bash scripts/launch_27k_deltanet.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
: "${GR_DATA_ROOT:=$REPO_ROOT/tmp}"

OUT="$GR_DATA_ROOT/KuaiRand-27K/data/processed_seqs.csv"
if [[ ! -f "$OUT" ]]; then
  echo "[launch] ERROR: $OUT not found. Run scripts/preprocess_27k.sh first." >&2
  exit 1
fi

SEEDS=(1 2 3 4 5 6 7 8 9 10)
WINDOWS=(128 256 512)

mkdir -p logs
: > logs/kr27k_deltanet_jobids.txt

# submit <window> <seed>
submit() {
  local window="$1" seed="$2"
  local name="kr27k_deltanet_w${window}_s${seed}"
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --stu-module STU_DELTANET --seed "$seed" --num-epochs 1 \
    --max-seq-len 16384 --max-attn-len "$window" \
    --run-name "$name")
  echo "$jid  $name  window=$window seed=$seed"
  echo "$jid $name $window $seed" >> logs/kr27k_deltanet_jobids.txt
}

for W in "${WINDOWS[@]}"; do
  echo "=== STU_DELTANET window=${W} (seeds ${SEEDS[*]}) ==="
  for s in "${SEEDS[@]}"; do
    submit "$W" "$s"
  done
done

echo "=== all $(( ${#SEEDS[@]} * ${#WINDOWS[@]} )) submitted. job ids in logs/kr27k_deltanet_jobids.txt ==="
