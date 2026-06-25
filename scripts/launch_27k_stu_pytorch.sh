#!/usr/bin/env bash
# Launch the vanilla pure-PyTorch HSTU (STU_PYTORCH) baseline on kuairand-27k:
# seeds 1..10, 1 epoch, train-eval, 1 GPU each, max_seq_len=16384
# (matches the existing 10-seed comparison protocol).
#   STU_PYTORCH -> kr27k_stu_pytorch_s<seed>
#
# Requires KuaiRand-27K/data/processed_seqs.csv (see scripts/preprocess_27k.sh).
#
# Usage: bash scripts/launch_27k_stu_pytorch.sh

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
: > logs/kr27k_stu_pytorch_jobids.txt

submit() {
  local name="$1" seed="$2"
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --stu-module STU_PYTORCH --seed "$seed" --num-epochs 1 --max-seq-len 16384 \
    --run-name "$name")
  echo "$jid  $name  seed=$seed"
  echo "$jid $name $seed" >> logs/kr27k_stu_pytorch_jobids.txt
}

echo "=== STU_PYTORCH (seeds ${SEEDS[*]}) ==="
for s in "${SEEDS[@]}"; do
  submit "kr27k_stu_pytorch_s${s}" "$s"
done

echo "=== all ${#SEEDS[@]} submitted. job ids in logs/kr27k_stu_pytorch_jobids.txt ==="
