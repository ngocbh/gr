#!/usr/bin/env bash
# Launch the STUpDeltaNet (STU_PDELTANET) sweep + the "full attention at W"
# baselines on kuairand-27k.
#
# STUpDeltaNet (non-overlapping split: windowed attn covers recent W, gated-delta
# memory summarizes history older than W; candidate-conditioned blend gate):
#   max_seq_len=16384, window in {128,256,512}, seeds 1..10
#   -> kr27k_pdeltanet_w<W>_s<seed>   (logs/kr27k_pdeltanet_jobids.txt)
#
# Full-attention-at-W baseline (STU_PYTORCH, max_seq_len=W, no window = full attn
# over the last W interactions only, no long memory):
#   max_seq_len in {128,256}, seeds 1..10
#   -> kr27k_pytorch_fa_l<W>_s<seed>  (logs/kr27k_fullattn_short_jobids.txt)
#   NOTE: W=512 already exists (seqlen sweep kr27k_stu_pytorch_l512_s*, jobs
#   1469718-1469727); W=16384 is the full-attn reference (1469665-1469675).
#
# Usage: bash scripts/launch_27k_pdeltanet.sh

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
WINDOWS=(256 512 1024)
FA_SEQLENS=(256 1024)   # full-attn-at-W; W=512 already covered by seqlen sweep

mkdir -p logs
: > logs/kr27k_pdeltanet_jobids.txt
: > logs/kr27k_fullattn_short_jobids.txt

submit_pdelta() {
  local window="$1" seed="$2"
  local name="kr27k_pdeltanet_w${window}_s${seed}"
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --stu-module STU_PDELTANET --seed "$seed" --num-epochs 1 \
    --max-seq-len 16384 --max-attn-len "$window" \
    --run-name "$name")
  echo "$jid  $name  window=$window seed=$seed"
  echo "$jid $name $window $seed" >> logs/kr27k_pdeltanet_jobids.txt
}

submit_fa() {
  local seqlen="$1" seed="$2"
  local name="kr27k_pytorch_fa_l${seqlen}_s${seed}"
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --stu-module STU_PYTORCH --seed "$seed" --num-epochs 1 \
    --max-seq-len "$seqlen" \
    --run-name "$name")
  echo "$jid  $name  seqlen=$seqlen seed=$seed"
  echo "$jid $name $seqlen $seed" >> logs/kr27k_fullattn_short_jobids.txt
}

for W in "${WINDOWS[@]}"; do
  echo "=== STU_PDELTANET window=${W} (seeds ${SEEDS[*]}) ==="
  for s in "${SEEDS[@]}"; do submit_pdelta "$W" "$s"; done
done

for L in "${FA_SEQLENS[@]}"; do
  echo "=== STU_PYTORCH full-attn seq_len=${L} (seeds ${SEEDS[*]}) ==="
  for s in "${SEEDS[@]}"; do submit_fa "$L" "$s"; done
done

n=$(( ${#SEEDS[@]} * (${#WINDOWS[@]} + ${#FA_SEQLENS[@]}) ))
echo "=== all ${n} submitted. ids in logs/kr27k_pdeltanet_jobids.txt + logs/kr27k_fullattn_short_jobids.txt ==="
