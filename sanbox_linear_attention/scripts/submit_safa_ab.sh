#!/usr/bin/env bash
# Snapshot, qualify, then submit selected paired HSTU/SAFA arrays.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
selection="${1:-all}"
if (( $# > 1 )); then
  echo "usage: $0 [all|amzn-books|kuairand-1k|ml-1m|ml-20m]" >&2
  exit 2
fi
case "$selection" in
  all)
    datasets=(amzn-books kuairand-1k ml-1m ml-20m)
    ;;
  amzn-books|kuairand-1k|ml-1m|ml-20m)
    datasets=("$selection")
    ;;
  *)
    echo "usage: $0 [all|amzn-books|kuairand-1k|ml-1m|ml-20m]" >&2
    exit 2
    ;;
esac

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
log_root="$repo_root/logs"
for export_value in \
  "$GR_DATA_ROOT" \
  "$GR_EXPS_ROOT" \
  "$GR_CKPTS_ROOT" \
  "$GR_CODE_SNAPSHOT_ROOT" \
  "$GR_QUALIFICATION_ROOT"; do
  if [[ "$export_value" == *","* || "$export_value" == *$'\n'* ]]; then
    echo "SLURM export paths must not contain commas or newlines" >&2
    exit 2
  fi
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
snapshot="$GR_CODE_SNAPSHOT_ROOT/safa-$timestamp-$$"
mkdir -p "$GR_CODE_SNAPSHOT_ROOT" "$GR_QUALIFICATION_ROOT" "$log_root"
python3 "$repo_root/scripts/snapshot.py" create "$snapshot" --source "$repo_root"
manifest="$(<"$snapshot/SOURCE_MANIFEST_SHA256")"
commit="$(<"$snapshot/SOURCE_COMMIT")"
tree="$(<"$snapshot/SOURCE_TREE")"
qualification_marker="$GR_QUALIFICATION_ROOT/$manifest.passed"

common_export="ALL,GR_CODE_SNAPSHOT=$snapshot,GR_EXPECTED_SOURCE_MANIFEST=$manifest,GR_DATA_ROOT=$GR_DATA_ROOT,GR_EXPS_ROOT=$GR_EXPS_ROOT,GR_CKPTS_ROOT=$GR_CKPTS_ROOT,GR_QUALIFICATION_ROOT=$GR_QUALIFICATION_ROOT,GR_REQUIRE_WANDB=0,GR_REQUIRE_SLURM_PROVENANCE=0"
qualification_job="$(sbatch --parsable \
  --partition=h200 --qos=h200_dev \
  --output="$log_root/slurm-%j.out" \
  --error="$log_root/slurm-%j.out" \
  --export="$common_export" \
  "$snapshot/scripts/sbatch_qualify_safa.sh")"
qualification_job="${qualification_job%%;*}"
if [[ ! "$qualification_job" =~ ^[1-9][0-9]*$ ]]; then
  echo "sbatch returned an invalid qualification job ID: $qualification_job" >&2
  exit 1
fi

declare -A array_jobs
for dataset in "${datasets[@]}"; do
  case "$dataset" in
    amzn-books)
      qos="h200_mrs_2_high"
      time_limit="3-00:00:00"
      job_name="safa-ab-amzn-books"
      ;;
    kuairand-1k)
      qos="h200_mrs_2_high"
      time_limit="3-00:00:00"
      job_name="safa-ab-kuairand1k"
      ;;
    ml-1m)
      qos="h200_dev"
      time_limit="24:00:00"
      job_name="safa-ab-ml1m"
      ;;
    ml-20m)
      qos="h200_mrs_2_high"
      time_limit="24:00:00"
      job_name="safa-ab-ml20m"
      ;;
  esac
  array_job="$(sbatch --parsable \
    --partition=h200 --qos="$qos" --time="$time_limit" \
    --job-name="$job_name" \
    --output="$log_root/slurm-%A_%a.out" \
    --error="$log_root/slurm-%A_%a.out" \
    --dependency="afterok:$qualification_job" \
    --export="$common_export,GR_DATASET=$dataset" \
    "$snapshot/scripts/sbatch_safa_ab.sh")"
  array_job="${array_job%%;*}"
  if [[ ! "$array_job" =~ ^[1-9][0-9]*$ ]]; then
    echo "sbatch returned an invalid array job ID: $array_job" >&2
    exit 1
  fi
  array_jobs["$dataset"]="$array_job"
done

echo "source_snapshot=$snapshot"
echo "source_commit=$commit"
echo "source_tree=$tree"
echo "source_manifest=$manifest"
echo "qualification_job=${qualification_job%%;*}"
for dataset in "${datasets[@]}"; do
  array_job="${array_jobs[$dataset]}"
  label="${dataset//-/}"
  echo "${label}_array_job=$array_job"
  echo "postrun_${label}=/bin/bash $snapshot/scripts/check_safa_results.sh RESULTS.json --expected-dataset $dataset --expected-source-commit $commit --expected-source-tree $tree --expected-source-manifest $manifest --expected-experiment-config-sha256 \$(sed -n 's/^experiment_config_${dataset}=//p' '$qualification_marker') --expected-array-job-id $array_job"
done
