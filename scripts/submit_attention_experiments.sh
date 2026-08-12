#!/usr/bin/env bash
# Snapshot the current source and submit the attention experiments reproducibly.
# Usage: bash scripts/submit_attention_experiments.sh [core|ab|fohstu|fohstu-repl|fosoftmax|softmax-canonical|moments|hybrid|lift-ml20|tanh|signed-additive|signed-lift]

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:-core}"
if [[ "$mode" == "all" ]]; then
  echo "warning: mode 'all' is a deprecated alias for the core AB + FoHSTU set" >&2
  mode="core"
fi
if [[ "$mode" != "core" && "$mode" != "ab" && "$mode" != "fohstu" && "$mode" != "fohstu-repl" && "$mode" != "fosoftmax" && "$mode" != "softmax-canonical" && "$mode" != "moments" && "$mode" != "hybrid" && "$mode" != "lift-ml20" && "$mode" != "tanh" && "$mode" != "signed-additive" && "$mode" != "signed-lift" ]]; then
  echo "usage: $0 [core|ab|fohstu|fohstu-repl|fosoftmax|softmax-canonical|moments|hybrid|lift-ml20|tanh|signed-additive|signed-lift]" >&2
  exit 2
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in .env or the environment}"
: "${WANDB_PROJECT:=gr}"
export WANDB_API_KEY WANDB_PROJECT

slurm_log_dir=/checkpoints/ngocbh/longhstu/logs/slurm
mkdir -p "$slurm_log_dir"

snapshot_base="${GR_CODE_SNAPSHOT_ROOT:-/checkpoints/ngocbh/longhstu/code_snapshots}"
snapshot="$snapshot_base/attention_$(date -u +%Y%m%dT%H%M%SZ)_$$"
bash scripts/snapshot_experiment.sh "$snapshot"

snapshot_manifest="$snapshot/SOURCE_SHA256SUMS"
snapshot_inventory="$snapshot/SOURCE_TREE_INVENTORY"
if [[ ! -f "$snapshot_manifest" || -L "$snapshot_manifest" ]]; then
  echo "snapshot manifest not found: $snapshot_manifest" >&2
  exit 1
fi
if [[ ! -f "$snapshot_inventory" || -L "$snapshot_inventory" ]]; then
  echo "snapshot tree inventory not found: $snapshot_inventory" >&2
  exit 1
fi
unsupported_node="$(
  cd "$snapshot"
  LC_ALL=C find . -mindepth 1 ! -type f ! -type d -print -quit
)"
if [[ -n "$unsupported_node" ]]; then
  echo "snapshot contains a symlink or special node before submission" >&2
  exit 1
fi
if ! (
  cd "$snapshot"
  sha256sum --check --strict --quiet SOURCE_SHA256SUMS
); then
  echo "snapshot checksum validation failed before submission" >&2
  exit 1
fi
generated_inventory="$(mktemp /tmp/gr-snapshot-inventory.XXXXXX)"
if ! (
  cd "$snapshot"
  /bin/bash scripts/snapshot_tree_inventory.sh . >"$generated_inventory"
); then
  rm -f -- "$generated_inventory"
  echo "could not regenerate snapshot tree inventory before submission" >&2
  exit 1
fi
if ! cmp -s -- "$snapshot_inventory" "$generated_inventory"; then
  rm -f -- "$generated_inventory"
  echo "snapshot tree inventory mismatch before submission" >&2
  exit 1
fi
rm -f -- "$generated_inventory"
read -r snapshot_manifest_sha256 _ < <(sha256sum "$snapshot_manifest")
if [[ ! "$snapshot_manifest_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "could not compute a valid snapshot manifest SHA256" >&2
  exit 1
fi
export GR_SNAPSHOT_MANIFEST_SHA256="$snapshot_manifest_sha256"

submit() {
  local script="$1"
  local snapshot_script="$snapshot/$script"
  if [[ ! -f "$snapshot_script" ]]; then
    echo "snapshot wrapper not found: $snapshot_script" >&2
    return 1
  fi
  sbatch \
    --parsable \
    --export="ALL,GR_CODE_SNAPSHOT=$snapshot,GR_SNAPSHOT_MANIFEST_SHA256=$GR_SNAPSHOT_MANIFEST_SHA256" \
    "$snapshot_script"
}

echo "source_snapshot=$snapshot"
echo "snapshot_manifest_sha256=$GR_SNAPSHOT_MANIFEST_SHA256"
if [[ "$mode" == "core" || "$mode" == "ab" ]]; then
  echo "attention_ab_job=$(submit scripts/sbatch_attention_ab_ml20m.sh)"
fi
if [[ "$mode" == "core" || "$mode" == "fohstu" ]]; then
  echo "fohstu_ml1m_job=$(submit scripts/sbatch_fohstu_ml1m.sh)"
fi
if [[ "$mode" == "fohstu-repl" ]]; then
  echo "fohstu_ml1m_replication_job=$(submit scripts/sbatch_fohstu_ml1m_replication.sh)"
fi
if [[ "$mode" == "fosoftmax" ]]; then
  echo "fosoftmax_ml1m_job=$(submit scripts/sbatch_fosoftmax_ml1m.sh)"
fi
if [[ "$mode" == "softmax-canonical" ]]; then
  echo "softmax_canonical_ml20m_job=$(submit scripts/sbatch_softmax_canonical_ml20m.sh)"
fi
if [[ "$mode" == "moments" ]]; then
  echo "moment_controls_ml1m_job=$(submit scripts/sbatch_moment_controls_ml1m.sh)"
fi
if [[ "$mode" == "hybrid" ]]; then
  echo "hybrid_fohstu_ml1m_job=$(submit scripts/sbatch_hybrid_fohstu_ml1m.sh)"
fi
if [[ "$mode" == "lift-ml20" ]]; then
  echo "lift_ml20m_job=$(submit scripts/sbatch_lift_ml20m.sh)"
fi
if [[ "$mode" == "tanh" ]]; then
  echo "hstu_tanh_ml20m_job=$(submit scripts/sbatch_hstu_tanh_ml20m.sh)"
fi
if [[ "$mode" == "signed-additive" ]]; then
  echo "signed_additive_ml1m_job=$(submit scripts/sbatch_signed_additive_ml1m.sh)"
fi
if [[ "$mode" == "signed-lift" ]]; then
  echo "signed_lift_ml1m_job=$(submit scripts/sbatch_signed_lift_ml1m.sh)"
fi
