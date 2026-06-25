#!/usr/bin/env bash
# Launch 3 extra kuairand-27k variants, seeds 1..10, 1 epoch, train-eval, 1 GPU each,
# max_seq_len=16384 (matches the existing 10-seed comparison table):
#   AttnRes block_size=2          -> kr27k_attnres_bs2_s<seed>
#   AttnRes block_size=4          -> kr27k_attnres_bs4_s<seed>
#   NeuTRENO lambda=0.2 after-norm-> kr27k_neutreno_an_l02_s<seed>
#
# Requires KuaiRand-27K/data/processed_seqs.csv (see scripts/preprocess_27k.sh).
#
# Usage: bash scripts/launch_27k_variants2.sh

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

mkdir -p logs
: > logs/kr27k_variants2_jobids.txt

# submit <name> <seed> <extra args...>
submit() {
  local name="$1" seed="$2"; shift 2
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --seed "$seed" --num-epochs 1 --max-seq-len 16384 --run-name "$name" "$@")
  echo "$jid  $name  seed=$seed  $*"
  echo "$jid $name $seed $*" >> logs/kr27k_variants2_jobids.txt
}

echo "=== AttnRes block_size=2 (seeds ${SEEDS[*]}) ==="
for s in "${SEEDS[@]}"; do
  submit "kr27k_attnres_bs2_s${s}" "$s" --stu-module AttnRes --attnres-block-size 2
done

echo "=== AttnRes block_size=4 (seeds ${SEEDS[*]}) ==="
for s in "${SEEDS[@]}"; do
  submit "kr27k_attnres_bs4_s${s}" "$s" --stu-module AttnRes --attnres-block-size 4
done

echo "=== NeuTRENO lambda=0.2 after-norm (seeds ${SEEDS[*]}) ==="
for s in "${SEEDS[@]}"; do
  submit "kr27k_neutreno_an_l02_s${s}" "$s" \
    --stu-module NeuTRENO --neutreno-lambda 0.2 --neutreno-after-norm
done

echo "=== all 30 submitted. job ids in logs/kr27k_variants2_jobids.txt ==="
