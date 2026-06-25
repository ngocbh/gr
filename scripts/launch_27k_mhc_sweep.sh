#!/usr/bin/env bash
# Launch the kuairand-27k mHC sweep: 10 different seeds, 1 epoch, train-eval, 1 GPU each.
#   stu_module=mHC (defaults: streams=4, iters=20, tau=0.05), seeds 1..10
#   run name: kr27k_mhc_s<seed>
#
# Requires KuaiRand-27K/data/processed_seqs.csv (see scripts/preprocess_27k.sh).
#
# Usage: bash scripts/launch_27k_mhc_sweep.sh

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
: > logs/kr27k_mhc_sweep_jobids.txt

submit() {
  local seed="$1"
  local name="kr27k_mhc_s${seed}"
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --stu-module mHC --seed "$seed" --num-epochs 1 --run-name "$name")
  echo "$jid  $name  seed=$seed"
  echo "$jid $name $seed" >> logs/kr27k_mhc_sweep_jobids.txt
}

echo "=== dlrm_v3 kuairand-27k mHC: seeds ${SEEDS[*]} ==="
for s in "${SEEDS[@]}"; do
  submit "$s"
done
echo "=== all ${#SEEDS[@]} submitted. job ids in logs/kr27k_mhc_sweep_jobids.txt ==="
