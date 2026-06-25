#!/usr/bin/env bash
# One-shot: generate KuaiRand-27K/data/processed_seqs.csv from the raw logs.
#
# The raw 27K logs are already extracted under $GR_DATA_ROOT/KuaiRand-27K/data/,
# so we avoid the 9.8GB re-download by running the processor with data_path=""
# from inside $GR_DATA_ROOT. The tar is present in cwd, so download() is a no-op
# (file_exists finds it and skips urlretrieve).
#
# Pure-pandas CPU job; reads ~24GB of CSV with groupby+agg(list) -> needs lots
# of RAM. Submit via: sbatch --gres=gpu:h200:1 scripts/wrapper.sh bash scripts/preprocess_27k.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${GR_CONDA_ENV:=gr}"
: "${GR_DATA_ROOT:=$REPO_ROOT/tmp}"

# --- activate conda env ---
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "$GR_CONDA_ENV" ]]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/ngocbh/miniconda3)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$GR_CONDA_ENV"
fi

export TMPDIR="${TMPDIR:-/home/ngocbh/tmp/pip}"
mkdir -p "$TMPDIR"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUT="$GR_DATA_ROOT/KuaiRand-27K/data/processed_seqs.csv"
if [[ -f "$OUT" ]]; then
  echo "[preprocess_27k] already exists: $OUT — nothing to do"
  exit 0
fi

echo "[preprocess_27k] GR_DATA_ROOT=$GR_DATA_ROOT"
echo "[preprocess_27k] generating $OUT"

cd "$GR_DATA_ROOT"
python3 - <<'PY'
from generative_recommenders.dlrm_v3.preprocess_public_data import DLRMKuaiRandProcessor

# data_path="" + cwd=$GR_DATA_ROOT => log/output paths resolve to
# KuaiRand-27K/data/*. The tar in cwd makes download() a no-op.
proc = DLRMKuaiRandProcessor(
    download_url="",
    data_path="",
    file_name="KuaiRand-27K.tar.gz",
    prefix="KuaiRand-27K",
)
proc.preprocess()
print("DONE")
PY

echo "[preprocess_27k] done"
ls -la "$OUT"
