#!/usr/bin/env bash
# dlrm_v3 kuairand-27k method verification: vanilla vs NeuTRENO vs AttnRes.
#   methods: STU(vanilla) NeuTRENO AttnRes
#   seeds:   1 2 3                       (paired across methods)
#   1 epoch, train-eval, wandb on, 1 GPU each, max_seq_len=16384  -> 9 runs total
#
# Run name encodes method+seed: kr27k_<method>_s<S>
#
# Usage: bash generative_recommenders/dlrm_v3/scripts/launch_27k_methods.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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

MAX_SEQ_LEN=16384
SEEDS=(1 2 3)
METHODS=(STU NeuTRENO AttnRes)

submit() {
  local method="$1" seed="$2"
  local name="kr27k_${method,,}_s${seed}"
  local extra=()
  case "$method" in
    NeuTRENO) extra=(--stu-module NeuTRENO --neutreno-lambda 0.4) ;;
    AttnRes)  extra=(--stu-module AttnRes --attnres-block-size 1) ;;
    STU)      extra=(--stu-module STU) ;;
  esac
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --seed "$seed" --num-epochs 1 --max-seq-len "$MAX_SEQ_LEN" \
    --run-name "$name" "${extra[@]}")
  echo "$jid  $name  method=$method  seed=$seed  max_seq_len=$MAX_SEQ_LEN"
  echo "$jid $name $method $seed $MAX_SEQ_LEN" >> logs/kr27k_methods_jobids.txt
}

mkdir -p logs
: > logs/kr27k_methods_jobids.txt

for method in "${METHODS[@]}"; do
  echo "=== method=$method (seeds ${SEEDS[*]}) ==="
  for s in "${SEEDS[@]}"; do
    submit "$method" "$s"
  done
done

echo "=== all $(( ${#METHODS[@]} * ${#SEEDS[@]} )) submitted. job ids in logs/kr27k_methods_jobids.txt ==="
