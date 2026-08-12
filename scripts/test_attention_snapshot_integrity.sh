#!/usr/bin/env bash
# Exercise attention snapshot submission and fail-closed job verification locally.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tmp_root="$(mktemp -d /tmp/gr-attention-snapshot-test.XXXXXX)"
cleanup() {
  chmod -R u+w "$tmp_root" 2>/dev/null || true
  rm -rf -- "$tmp_root"
}
trap cleanup EXIT

fail() {
  echo "snapshot integrity test failed: $*" >&2
  exit 1
}

# The real submission script runs unchanged, but this exported function prevents
# any contact with SLURM and records the wrapper and exported environment names.
sbatch() {
  [[ -d /checkpoints/ngocbh/longhstu/logs/slurm ]] || {
    echo "SLURM log directory did not exist before sbatch" >&2
    return 97
  }
  local wrapper="${!#}"
  printf 'FAKE_SBATCH wrapper=%s' "$wrapper"
  local argument
  for argument in "$@"; do
    printf ' argument=%q' "$argument"
  done
}
export -f sbatch

submit_output="$(
  WANDB_API_KEY=snapshot-integrity-test \
    WANDB_PROJECT=snapshot-integrity-test \
    GR_CODE_SNAPSHOT_ROOT="$tmp_root/snapshots" \
    /bin/bash scripts/submit_attention_experiments.sh lift-ml20
)"
unset -f sbatch

snapshot="$(printf '%s\n' "$submit_output" | sed -n 's/^source_snapshot=//p')"
[[ -n "$snapshot" && -d "$snapshot" ]] || fail "snapshot path missing"
manifest="$snapshot/SOURCE_SHA256SUMS"
[[ -f "$manifest" ]] || fail "snapshot manifest missing"
inventory="$snapshot/SOURCE_TREE_INVENTORY"
[[ -f "$inventory" ]] || fail "snapshot tree inventory missing"
read -r expected_manifest_sha256 _ < <(sha256sum "$manifest")

expected_wrapper="$snapshot/scripts/sbatch_lift_ml20m.sh"
printf '%s\n' "$submit_output" \
  | grep -Fq "FAKE_SBATCH wrapper=$expected_wrapper" \
  || fail "sbatch did not receive the snapshotted wrapper"
printf '%s\n' "$submit_output" \
  | grep -Fq "GR_SNAPSHOT_MANIFEST_SHA256=$expected_manifest_sha256" \
  || fail "sbatch did not receive the expected manifest digest"

expected_tanh_wrapper="$snapshot/scripts/sbatch_hstu_tanh_ml20m.sh"
[[ -f "$expected_tanh_wrapper" ]] || fail "snapshotted tanh wrapper missing"
grep -Fq '#SBATCH --qos=h200_mrs_shared' "$expected_tanh_wrapper" \
  || fail "tanh wrapper does not request shared QoS"
grep -Fq '#SBATCH --gres=gpu:h200:1' "$expected_tanh_wrapper" \
  || fail "tanh wrapper does not request exactly one H200"
if grep -Fq 'h200_dev' "$expected_tanh_wrapper"; then
  fail "tanh wrapper references forbidden h200_dev QoS"
fi
if grep -Fq '#SBATCH --array=' "$expected_tanh_wrapper"; then
  fail "single-task tanh wrapper unexpectedly defines an array"
fi
grep -Fq 'hstu_tanh_ml20m_job=$(submit scripts/sbatch_hstu_tanh_ml20m.sh)' \
  "$snapshot/scripts/submit_attention_experiments.sh" \
  || fail "tanh submission mode is not wired to its snapshotted wrapper"
grep -Fq '  configs/ml-20m/hstu-tanh-20m.gin' "$manifest" \
  || fail "tanh gin config is not checksummed"

if find "$snapshot" -perm /222 -print -quit | grep -q .; then
  fail "snapshot contains writable paths"
fi
(
  cd "$snapshot"
  sha256sum --check --strict --quiet SOURCE_SHA256SUMS
) || fail "clean snapshot checksum verification failed"
grep -Fq '  SOURCE_TREE_INVENTORY' "$manifest" \
  || fail "tree inventory is not checksummed"
regenerated_inventory="$tmp_root/regenerated-inventory"
(
  cd "$snapshot"
  /bin/bash scripts/snapshot_tree_inventory.sh . >"$regenerated_inventory"
)
cmp -s -- "$inventory" "$regenerated_inventory" \
  || fail "clean snapshot tree inventory mismatch"

source_tamper="$tmp_root/source-tamper"
manifest_tamper="$tmp_root/manifest-tamper"
gin_tamper="$tmp_root/gin-tamper"
env_tamper="$tmp_root/env-tamper"
symlink_tamper="$tmp_root/symlink-tamper"
empty_dir_tamper="$tmp_root/empty-dir-tamper"
deleted_file_tamper="$tmp_root/deleted-file-tamper"
special_node_tamper="$tmp_root/special-node-tamper"
for case_root in \
  "$source_tamper" \
  "$manifest_tamper" \
  "$gin_tamper" \
  "$env_tamper" \
  "$symlink_tamper" \
  "$empty_dir_tamper" \
  "$deleted_file_tamper" \
  "$special_node_tamper"; do
  cp -a "$snapshot" "$case_root"
done

# Intercept only the final training command. Calling the wrapper itself through
# /bin/bash guarantees this function cannot replace its integrity checks.
bash() {
  if [[ "${1:-}" == "scripts/train.sh" ]]; then
    printf 'FAKE_TRAIN_REACHED\n'
    return 0
  fi
  /bin/bash "$@"
}
export -f bash
[[ "$(/bin/bash -c 'type -t bash')" == "function" ]] \
  || fail "could not install the fake training interceptor"

run_wrapper() {
  local case_root="$1"
  env \
    GR_CODE_SNAPSHOT="$case_root" \
    GR_SNAPSHOT_MANIFEST_SHA256="$expected_manifest_sha256" \
    SLURM_ARRAY_TASK_ID=2 \
    SLURM_ARRAY_JOB_ID=900001 \
    SLURM_JOB_ID=900001 \
    SLURM_RESTART_COUNT=0 \
    WANDB_API_KEY=snapshot-integrity-test \
    WANDB_PROJECT=snapshot-integrity-test \
    /bin/bash "$case_root/scripts/sbatch_lift_ml20m.sh"
}

run_tanh_wrapper() {
  local case_root="$1"
  env \
    GR_CODE_SNAPSHOT="$case_root" \
    GR_SNAPSHOT_MANIFEST_SHA256="$expected_manifest_sha256" \
    SLURM_JOB_ID=900002 \
    SLURM_RESTART_COUNT=0 \
    WANDB_API_KEY=snapshot-integrity-test \
    WANDB_PROJECT=snapshot-integrity-test \
    /bin/bash "$case_root/scripts/sbatch_hstu_tanh_ml20m.sh"
}

expect_blocked() {
  local case_root="$1"
  local label="$2"
  local expected_error="$3"
  local output
  if output="$(run_wrapper "$case_root" 2>&1)"; then
    fail "$label reached a successful exit"
  fi
  if printf '%s\n' "$output" | grep -Fq 'FAKE_TRAIN_REACHED'; then
    fail "$label reached fake training"
  fi
  printf '%s\n' "$output" | grep -Fq "$expected_error" \
    || fail "$label did not fail with: $expected_error"
}

clean_output="$(run_wrapper "$snapshot" 2>&1)"
printf '%s\n' "$clean_output" | grep -Fq 'FAKE_TRAIN_REACHED' \
  || fail "clean snapshot did not reach fake training"
tanh_clean_output="$(run_tanh_wrapper "$snapshot" 2>&1)"
printf '%s\n' "$tanh_clean_output" | grep -Fq 'FAKE_TRAIN_REACHED' \
  || fail "clean tanh snapshot did not reach fake training"

source_file="$source_tamper/generative_recommenders/research/modeling/sequential/hstu.py"
chmod u+w "$source_file"
printf '\n# snapshot integrity test mutation\n' >>"$source_file"
chmod a-w "$source_file"
expect_blocked \
  "$source_tamper" \
  "source-tampered snapshot" \
  "snapshot file checksum validation failed"

tampered_manifest="$manifest_tamper/SOURCE_SHA256SUMS"
chmod u+w "$tampered_manifest"
printf '# snapshot integrity test mutation\n' >>"$tampered_manifest"
chmod a-w "$tampered_manifest"
expect_blocked \
  "$manifest_tamper" \
  "manifest-tampered snapshot" \
  "snapshot manifest SHA256 mismatch"

chmod u+w "$gin_tamper"
printf 'raise RuntimeError("injected gin.py")\n' >"$gin_tamper/gin.py"
chmod a-w "$gin_tamper/gin.py" "$gin_tamper"
expect_blocked \
  "$gin_tamper" \
  "root gin.py injection" \
  "snapshot tree inventory mismatch"

chmod u+w "$env_tamper"
printf 'GR_WANDB_ENABLED=0\n' >"$env_tamper/.env"
chmod a-w "$env_tamper/.env" "$env_tamper"
expect_blocked \
  "$env_tamper" \
  "root .env injection" \
  "snapshot tree inventory mismatch"

chmod u+w "$symlink_tamper"
ln -s main.py "$symlink_tamper/injected-symlink"
chmod a-w "$symlink_tamper"
expect_blocked \
  "$symlink_tamper" \
  "symlink injection" \
  "snapshot contains a symlink or special node"

chmod u+w "$empty_dir_tamper"
mkdir "$empty_dir_tamper/injected-empty-directory"
chmod a-w "$empty_dir_tamper/injected-empty-directory" "$empty_dir_tamper"
expect_blocked \
  "$empty_dir_tamper" \
  "empty-directory injection" \
  "snapshot tree inventory mismatch"

chmod u+w "$deleted_file_tamper"
rm -- "$deleted_file_tamper/main.py"
chmod a-w "$deleted_file_tamper"
expect_blocked \
  "$deleted_file_tamper" \
  "deleted source file" \
  "snapshot file checksum validation failed"

chmod u+w "$special_node_tamper"
mkfifo "$special_node_tamper/injected-fifo"
chmod a-w "$special_node_tamper/injected-fifo" "$special_node_tamper"
expect_blocked \
  "$special_node_tamper" \
  "special-node injection" \
  "snapshot contains a symlink or special node"

printf 'fake_sbatch_snapshot_wrapper=PASS\n'
printf 'clean_snapshot_fake_train=PASS\n'
printf 'clean_tanh_snapshot_fake_train=PASS\n'
printf 'source_tamper_blocked_before_train=PASS\n'
printf 'manifest_tamper_blocked_before_train=PASS\n'
printf 'root_gin_injection_blocked_before_train=PASS\n'
printf 'root_env_injection_blocked_before_train=PASS\n'
printf 'symlink_injection_blocked_before_train=PASS\n'
printf 'empty_directory_injection_blocked_before_train=PASS\n'
printf 'deleted_file_blocked_before_train=PASS\n'
printf 'special_node_injection_blocked_before_train=PASS\n'
