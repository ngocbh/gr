#!/usr/bin/env bash
# Post-run scientific gate. This is distinct from the pre-run smoke qualification.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GR_PYTHON:-$(command -v python3)}"
exec "$python_bin" "$repo_root/scripts/qualify_safa_results.py" "$@"
