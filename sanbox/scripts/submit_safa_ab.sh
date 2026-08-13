#!/usr/bin/env bash
# Snapshot, qualify, then submit ML-1M and ML-20M paired HSTU/SAFA arrays.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GR_DATA_ROOT:?GR_DATA_ROOT must be set}"
: "${GR_EXPS_ROOT:?GR_EXPS_ROOT must be set}"
: "${GR_CKPTS_ROOT:?GR_CKPTS_ROOT must be set}"
: "${GR_CODE_SNAPSHOT_ROOT:=$GR_CKPTS_ROOT/source_snapshots}"
: "${GR_QUALIFICATION_ROOT:=$GR_EXPS_ROOT/qualifications}"
GR_DATA_ROOT="$(realpath -m "$GR_DATA_ROOT")"
GR_EXPS_ROOT="$(realpath -m "$GR_EXPS_ROOT")"
GR_CKPTS_ROOT="$(realpath -m "$GR_CKPTS_ROOT")"
GR_CODE_SNAPSHOT_ROOT="$(realpath -m "$GR_CODE_SNAPSHOT_ROOT")"
GR_QUALIFICATION_ROOT="$(realpath -m "$GR_QUALIFICATION_ROOT")"
for export_value in \
  "$GR_DATA_ROOT" \
  "$GR_EXPS_ROOT" \
  "$GR_CKPTS_ROOT" \
  "$GR_QUALIFICATION_ROOT"; do
  if [[ "$export_value" == *","* || "$export_value" == *$'\n'* ]]; then
    echo "SLURM export paths must not contain commas or newlines" >&2
    exit 2
  fi
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
snapshot="$GR_CODE_SNAPSHOT_ROOT/safa-$timestamp-$$"
mkdir -p "$GR_CODE_SNAPSHOT_ROOT" "$GR_QUALIFICATION_ROOT"
python3 "$repo_root/scripts/snapshot.py" create "$snapshot" --source "$repo_root"
manifest="$(<"$snapshot/SOURCE_MANIFEST_SHA256")"
commit="$(<"$snapshot/SOURCE_COMMIT")"
tree="$(<"$snapshot/SOURCE_TREE")"
qualification_marker="$GR_QUALIFICATION_ROOT/$manifest.passed"

common_export="ALL,GR_EXPECTED_SOURCE_MANIFEST=$manifest,GR_DATA_ROOT=$GR_DATA_ROOT,GR_EXPS_ROOT=$GR_EXPS_ROOT,GR_CKPTS_ROOT=$GR_CKPTS_ROOT,GR_QUALIFICATION_ROOT=$GR_QUALIFICATION_ROOT"
qualification_job="$(sbatch --parsable \
  --partition=h200 --qos=h200_mrs_shared \
  --export="$common_export" \
  "$snapshot/scripts/sbatch_qualify_safa.sh")"
qualification_job="${qualification_job%%;*}"

ml1m_job="$(sbatch --parsable \
  --partition=h200 --qos=h200_mrs_shared \
  --job-name=safa-ab-ml1m \
  --dependency="afterok:$qualification_job" \
  --export="$common_export,GR_DATASET=ml-1m" \
  "$snapshot/scripts/sbatch_safa_ab.sh")"
ml20m_job="$(sbatch --parsable \
  --partition=h200 --qos=h200_mrs_shared \
  --job-name=safa-ab-ml20m \
  --dependency="afterok:$qualification_job" \
  --export="$common_export,GR_DATASET=ml-20m" \
  "$snapshot/scripts/sbatch_safa_ab.sh")"

echo "source_snapshot=$snapshot"
echo "source_commit=$commit"
echo "source_tree=$tree"
echo "source_manifest=$manifest"
echo "qualification_job=${qualification_job%%;*}"
echo "ml1m_array_job=${ml1m_job%%;*}"
echo "ml20m_array_job=${ml20m_job%%;*}"
echo "postrun_ml1m=/bin/bash $snapshot/scripts/check_safa_results.sh RESULTS.json --expected-dataset ml-1m --expected-source-commit $commit --expected-source-tree $tree --expected-source-manifest $manifest --expected-experiment-config-sha256 \$(sed -n 's/^experiment_config_ml-1m=//p' '$qualification_marker')"
echo "postrun_ml20m=/bin/bash $snapshot/scripts/check_safa_results.sh RESULTS.json --expected-dataset ml-20m --expected-source-commit $commit --expected-source-tree $tree --expected-source-manifest $manifest --expected-experiment-config-sha256 \$(sed -n 's/^experiment_config_ml-20m=//p' '$qualification_marker')"
