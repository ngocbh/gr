#!/usr/bin/env bash
# Emit a deterministic NUL-delimited type/path inventory for a snapshot root.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SNAPSHOT_ROOT" >&2
  exit 2
fi

snapshot_root="$1"
if [[ ! -d "$snapshot_root" || -L "$snapshot_root" ]]; then
  echo "snapshot root must be a non-symlink directory" >&2
  exit 1
fi
cd -- "$snapshot_root"

unsupported_node="$(
  LC_ALL=C find . -mindepth 1 ! -type f ! -type d -print -quit
)"
if [[ -n "$unsupported_node" ]]; then
  echo "snapshot contains a symlink or special node" >&2
  exit 1
fi

LC_ALL=C find . -mindepth 1 \
  ! -path './SOURCE_TREE_INVENTORY' \
  ! -path './SOURCE_SHA256SUMS' \
  \( \
    -type d -printf 'd %P\0' \
    -o -type f -printf 'f %P\0' \
  \) \
  | LC_ALL=C sort -z
