#!/usr/bin/env bash
# Launch the kuairand-27k reproducibility sweep: 1 epoch, train-eval, 1 GPU each.
#   5 same-seed runs  (seed=100)      -> kr27k_same_r1..r5
#   5 diff-seed runs  (seed=1..5)     -> kr27k_diff_s1..s5
#
# Mirrors the kuairand-1k methodology. Requires KuaiRand-27K/data/processed_seqs.csv
# to already exist (see scripts/preprocess_27k.sh).
#
# Usage: bash scripts/launch_27k_sweep.sh

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

submit() {
  local name="$1" seed="$2"
  local jid
  jid=$(sbatch --parsable --gres=gpu:h200:1 --job-name="$name" \
    scripts/wrapper.sh bash scripts/train.sh dlrm_v3 kuairand-27k \
    --seed "$seed" --num-epochs 1 --run-name "$name")
  echo "$jid  $name  seed=$seed"
  echo "$jid $name $seed" >> logs/kr27k_sweep_jobids.txt
}

mkdir -p logs
: > logs/kr27k_sweep_jobids.txt

echo "=== same-seed (seed=100) x5 ==="
for r in 1 2 3 4 5; do
  submit "kr27k_same_r${r}" 100
done

echo "=== diff-seed (seed=1..5) x5 ==="
for s in 1 2 3 4 5; do
  submit "kr27k_diff_s${s}" "$s"
done

echo "=== all 10 submitted. job ids in logs/kr27k_sweep_jobids.txt ==="
