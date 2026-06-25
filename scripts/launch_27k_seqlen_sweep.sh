#!/usr/bin/env bash
# kuairand-27k max_seq_len sweep: vary the HSTU model max sequence length.
#   lengths: 512 1024 2048 4096 8192 16384   (dataset truncates uih to len-10)
#   seeds:   1 2 3                            (paired across lengths)
#   1 epoch, train-eval, wandb on, 1 GPU each  -> 18 runs total
#
# Run name encodes length+seed: kr27k_len<L>_s<S>
#
# Usage: bash scripts/launch_27k_seqlen_sweep.sh

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

LENGTHS=(512 1024 2048 4096 8192 16384)
SEEDS=(1 2 3)

submit() {
  local name="$1" seed="$2" len="$3"
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --seed "$seed" --num-epochs 1 --max-seq-len "$len" --run-name "$name")
  echo "$jid  $name  seed=$seed  max_seq_len=$len"
  echo "$jid $name $seed $len" >> logs/kr27k_seqlen_sweep_jobids.txt
}

mkdir -p logs
: > logs/kr27k_seqlen_sweep_jobids.txt

for len in "${LENGTHS[@]}"; do
  echo "=== max_seq_len=$len (seeds ${SEEDS[*]}) ==="
  for s in "${SEEDS[@]}"; do
    submit "kr27k_len${len}_s${s}" "$s" "$len"
  done
done

echo "=== all $(( ${#LENGTHS[@]} * ${#SEEDS[@]} )) submitted. job ids in logs/kr27k_seqlen_sweep_jobids.txt ==="
