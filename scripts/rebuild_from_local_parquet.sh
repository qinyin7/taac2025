#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-$ROOT_DIR/data_full}"

: "${LOCAL_PARQUET_DIR:=D:/hf_cache/TAAC2025/TencentGR-1M}"
: "${LOCAL_NUM_USERS:=0}"
: "${LOCAL_EVAL_USERS:=4096}"
: "${LOCAL_EXTRA_ITEMS:=0}"
: "${LOCAL_BATCH_SIZE:=8192}"

if [[ ! -d "$LOCAL_PARQUET_DIR/seq" || ! -d "$LOCAL_PARQUET_DIR/user_feat" || ! -d "$LOCAL_PARQUET_DIR/item_feat" ]]; then
  echo "Missing local parquet dirs under: $LOCAL_PARQUET_DIR" >&2
  echo "Expected: seq/, user_feat/, item_feat/" >&2
  exit 1
fi

python -u scripts/make_local_parquet_data.py \
  --parquet-root "$LOCAL_PARQUET_DIR" \
  --output-dir "$OUT_DIR" \
  --num-users "$LOCAL_NUM_USERS" \
  --eval-users "$LOCAL_EVAL_USERS" \
  --extra-items "$LOCAL_EXTRA_ITEMS" \
  --batch-size "$LOCAL_BATCH_SIZE"

export USER_CACHE_PATH="$OUT_DIR/cache"
mkdir -p "$USER_CACHE_PATH"
python -u get_stat.py --data_path "$OUT_DIR"

echo "Done rebuilding local parquet data at: $OUT_DIR"
