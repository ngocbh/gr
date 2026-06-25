#!/usr/bin/env bash
# Launch STU_PYTORCH at shorter sequence lengths on kuairand-27k:
# seq_len in {512, 2048, 4096, 8192}, seeds 1..10, 1 epoch, train-eval, 1 GPU each.
#   seq_len=512  -> kr27k_stu_pytorch_l512_s<seed>
#   seq_len=2048 -> kr27k_stu_pytorch_l2048_s<seed>
#   seq_len=4096 -> kr27k_stu_pytorch_l4096_s<seed>
#   seq_len=8192 -> kr27k_stu_pytorch_l8192_s<seed>
# Companion to scripts/launch_27k_stu_pytorch.sh (which runs seq_len=16384).
#
# Requires KuaiRand-27K/data/processed_seqs.csv (see scripts/preprocess_27k.sh).
#
# Usage: bash scripts/launch_27k_stu_pytorch_seqlen.sh

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
SEQLENS=(512 2048 4096 8192)

mkdir -p logs
: > logs/kr27k_stu_pytorch_seqlen_jobids.txt

# submit <seqlen> <seed>
submit() {
  local seqlen="$1" seed="$2"
  local name="kr27k_stu_pytorch_l${seqlen}_s${seed}"
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --stu-module STU_PYTORCH --seed "$seed" --num-epochs 1 --max-seq-len "$seqlen" \
    --run-name "$name")
  echo "$jid  $name  seqlen=$seqlen seed=$seed"
  echo "$jid $name $seqlen $seed" >> logs/kr27k_stu_pytorch_seqlen_jobids.txt
}

for L in "${SEQLENS[@]}"; do
  echo "=== STU_PYTORCH seq_len=${L} (seeds ${SEEDS[*]}) ==="
  for s in "${SEEDS[@]}"; do
    submit "$L" "$s"
  done
done

echo "=== all $(( ${#SEEDS[@]} * ${#SEQLENS[@]} )) submitted. job ids in logs/kr27k_stu_pytorch_seqlen_jobids.txt ==="
