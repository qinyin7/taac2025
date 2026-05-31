#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[compat] scripts/run_hf_smoke_train.sh is deprecated; use scripts/run_taac_train.sh instead." >&2
exec "$ROOT_DIR/scripts/run_taac_train.sh" "$@"
