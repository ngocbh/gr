#!/usr/bin/env bash
# Create a small immutable source snapshot for a SLURM experiment.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DESTINATION" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$1"

if [[ -e "$destination" ]]; then
  echo "snapshot destination already exists: $destination" >&2
  exit 2
fi

mkdir -p "$destination"
cd "$repo_root"

rsync -a \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  main.py \
  requirements.txt \
  generative_recommenders \
  configs \
  scripts \
  "$destination/"

git rev-parse HEAD >"$destination/GIT_COMMIT"
git status --short >"$destination/GIT_STATUS"
git diff --binary >"$destination/WORKTREE.patch"

# The inventory excludes only itself and its checksum manifest to avoid
# self-reference; SOURCE_TREE_INVENTORY is then pinned by SOURCE_SHA256SUMS.
/bin/bash scripts/snapshot_tree_inventory.sh "$destination" \
  >"$destination/SOURCE_TREE_INVENTORY"

(
  cd "$destination"
  LC_ALL=C find . -mindepth 1 -type f \
    ! -path './SOURCE_SHA256SUMS' \
    -printf '%P\0' \
    | LC_ALL=C sort -z \
    | xargs -0 -r sha256sum --
) >"$destination/SOURCE_SHA256SUMS"

chmod -R a-w "$destination"
echo "$destination"
