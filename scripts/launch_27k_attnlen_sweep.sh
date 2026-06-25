#!/usr/bin/env bash
# kuairand-27k sliding-window (max_attn_len) sweep.
#   max_seq_len:  16384 8192        (full-attention baselines reused from len sweep)
#   max_attn_len: 128 256 512 1024 2048 4096
#   seeds:        1 2 3
#   1 epoch, train-eval, wandb on, 1 GPU each -> 2*6*3 = 36 runs
#
# Run name: kr27k_l<seqlen>_w<window>_s<seed>
#
# Usage: bash scripts/launch_27k_attnlen_sweep.sh
#   (set SKIP="l8192_w128_s1 ..." to skip already-run cells, space-separated tags)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${GR_DATA_ROOT:=$REPO_ROOT/tmp}"
: "${SKIP:=}"

OUT="$GR_DATA_ROOT/KuaiRand-27K/data/processed_seqs.csv"
[[ -f "$OUT" ]] || { echo "[launch] ERROR: $OUT not found." >&2; exit 1; }

SEQLENS=(16384 8192)
WINDOWS=(128 256 512 1024 2048 4096)
SEEDS=(1 2 3)

mkdir -p logs
: > logs/kr27k_attnlen_sweep_jobids.txt

submit() {
  local seqlen="$1" win="$2" seed="$3"
  local name="kr27k_l${seqlen}_w${win}_s${seed}"
  for sk in $SKIP; do [[ "$sk" == "$name" ]] && { echo "skip $name"; return; }; done
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --seed "$seed" --num-epochs 1 --max-seq-len "$seqlen" --max-attn-len "$win" \
    --run-name "$name")
  echo "$jid  $name"
  echo "$jid $name $seqlen $win $seed" >> logs/kr27k_attnlen_sweep_jobids.txt
}

for sl in "${SEQLENS[@]}"; do
  for w in "${WINDOWS[@]}"; do
    for s in "${SEEDS[@]}"; do
      submit "$sl" "$w" "$s"
    done
  done
done

echo "=== submitted (skipped: ${SKIP:-none}) ==="
