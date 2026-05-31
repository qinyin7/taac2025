#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_DIR="${1:-$ROOT_DIR/data_smoke}"

: "${SMOKE_NUM_USERS:=32768}"
: "${SMOKE_EVAL_USERS:=2048}"
: "${SMOKE_EXTRA_ITEMS:=262144}"
: "${SMOKE_LOCAL_PARQUET_DIR:=D:/hf_cache/TAAC2025/TencentGR-1M}"
: "${SMOKE_STREAM_DATA:=1}"
: "${SMOKE_REBUILD_DATA:=0}"
: "${SMOKE_AUTO_RESUME:=1}"
: "${SMOKE_RESUME_PATH:=}"
: "${SMOKE_SAVE_INTERVAL:=100}"
: "${SMOKE_BATCH_SIZE:=96}"
: "${SMOKE_EVAL_BATCH_SIZE:=128}"
: "${SMOKE_MAXLEN:=96}"
: "${SMOKE_VAL_RATIO:=0.1}"
: "${SMOKE_NUM_EPOCHS:=10}"
: "${SMOKE_NUM_BLOCKS:=6}"
: "${SMOKE_HIDDEN_UNITS:=128}"
: "${SMOKE_NUM_HEADS:=8}"
: "${SMOKE_ITEM_EMB_BATCH_SIZE:=8192}"
: "${SMOKE_DEVICE:=cuda}"
: "${SMOKE_RUN_NAME:=h${SMOKE_HIDDEN_UNITS}_b${SMOKE_NUM_BLOCKS}_l${SMOKE_MAXLEN}_bs${SMOKE_BATCH_SIZE}}"

if [[ "$SMOKE_REBUILD_DATA" == "1" || ! -f "$SMOKE_DIR/seq.jsonl" || ! -f "$SMOKE_DIR/indexer.pkl" || ! -f "$SMOKE_DIR/item_feat_dict.json" ]]; then
  if [[ -n "$SMOKE_LOCAL_PARQUET_DIR" && -d "$SMOKE_LOCAL_PARQUET_DIR/seq" && -d "$SMOKE_LOCAL_PARQUET_DIR/user_feat" && -d "$SMOKE_LOCAL_PARQUET_DIR/item_feat" ]]; then
    python -u scripts/make_local_parquet_data.py \
      --parquet-root "$SMOKE_LOCAL_PARQUET_DIR" \
      --output-dir "$SMOKE_DIR" \
      --num-users "$SMOKE_NUM_USERS" \
      --eval-users "$SMOKE_EVAL_USERS" \
      --extra-items "$SMOKE_EXTRA_ITEMS"
  elif [[ "$SMOKE_STREAM_DATA" == "1" ]]; then
    python -u scripts/make_hf_stream_data.py \
      --output-dir "$SMOKE_DIR" \
      --num-users "$SMOKE_NUM_USERS" \
      --eval-users "$SMOKE_EVAL_USERS" \
      --extra-items "$SMOKE_EXTRA_ITEMS"
  else
    python -u scripts/make_hf_smoke_data.py \
      --output-dir "$SMOKE_DIR" \
      --num-users "$SMOKE_NUM_USERS" \
      --eval-users "$SMOKE_EVAL_USERS" \
      --extra-items "$SMOKE_EXTRA_ITEMS"
  fi
else
  echo "[1/5] Reusing existing converted data in $SMOKE_DIR"
  echo "      Set SMOKE_REBUILD_DATA=1 to rebuild it."
fi

export TRAIN_DATA_PATH="$SMOKE_DIR"
export TRAIN_CKPT_PATH="$SMOKE_DIR/checkpoints/$SMOKE_RUN_NAME"
export TRAIN_LOG_PATH="$SMOKE_DIR/logs/$SMOKE_RUN_NAME"
export TRAIN_TF_EVENTS_PATH="$SMOKE_DIR/tf_events/$SMOKE_RUN_NAME"
export USER_CACHE_PATH="$SMOKE_DIR/cache"

mkdir -p \
  "$TRAIN_CKPT_PATH" \
  "$TRAIN_LOG_PATH" \
  "$TRAIN_TF_EVENTS_PATH" \
  "$USER_CACHE_PATH"

if [[ "$SMOKE_REBUILD_DATA" == "1" || ! -f "$USER_CACHE_PATH/item_freq.csv" || ! -f "$USER_CACHE_PATH/item_last_ts.json" ]]; then
  python -u get_stat.py --data_path "$TRAIN_DATA_PATH"
else
  echo "[stats] Reusing existing cache in $USER_CACHE_PATH"
fi

RESUME_ARGS=()
if [[ -n "$SMOKE_RESUME_PATH" ]]; then
  RESUME_ARGS=(--resume "$SMOKE_RESUME_PATH")
elif [[ "$SMOKE_REBUILD_DATA" != "1" && "$SMOKE_AUTO_RESUME" == "1" && -f "$TRAIN_CKPT_PATH/latest/ckpt.pt" ]]; then
  RESUME_ARGS=(--resume "$TRAIN_CKPT_PATH/latest/ckpt.pt")
  echo "[resume] Using latest checkpoint: $TRAIN_CKPT_PATH/latest/ckpt.pt"
else
  echo "[resume] Starting a fresh training run"
fi

python -u main.py \
  --data_path "$TRAIN_DATA_PATH" \
  --ckpt_root "$TRAIN_CKPT_PATH" \
  --log_dir "$TRAIN_LOG_PATH" \
  --tf_dir "$TRAIN_TF_EVENTS_PATH" \
  --batch_size "$SMOKE_BATCH_SIZE" \
  --eval_batch_size "$SMOKE_EVAL_BATCH_SIZE" \
  --maxlen "$SMOKE_MAXLEN" \
  --num_workers 0 \
  --val_ratio "$SMOKE_VAL_RATIO" \
  --num_epochs "$SMOKE_NUM_EPOCHS" \
  --num_blocks "$SMOKE_NUM_BLOCKS" \
  --hidden_units "$SMOKE_HIDDEN_UNITS" \
  --num_heads "$SMOKE_NUM_HEADS" \
  --device "$SMOKE_DEVICE" \
  --item_emb_batch_size "$SMOKE_ITEM_EMB_BATCH_SIZE" \
  --topk 10 \
  --save_interval "$SMOKE_SAVE_INTERVAL" \
  --save_final \
  "${RESUME_ARGS[@]}"

FINAL_MODEL_DIR=""
if [[ "$SMOKE_NUM_USERS" -gt 0 ]]; then
  FINAL_MODEL_DIR="$TRAIN_CKPT_PATH/final_global_step$(( (SMOKE_NUM_USERS * 8 / 10) / SMOKE_BATCH_SIZE ))"
fi
if [[ -z "$FINAL_MODEL_DIR" || ! -d "$FINAL_MODEL_DIR" ]]; then
  FINAL_MODEL_DIR="$(find "$TRAIN_CKPT_PATH" -maxdepth 1 -type d -name 'final_global_step*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
fi

if [[ -n "$FINAL_MODEL_DIR" && -f "$FINAL_MODEL_DIR/model.pt" ]]; then
  python -u scripts/run_taac_infer.py \
    --model-dir "$FINAL_MODEL_DIR" \
    --eval-data-path "$SMOKE_DIR/eval" \
    --output-json "$FINAL_MODEL_DIR/result.json" \
    --topk 10 \
    --batch-size "$SMOKE_EVAL_BATCH_SIZE" \
    --maxlen "$SMOKE_MAXLEN" \
    --num-blocks "$SMOKE_NUM_BLOCKS" \
    --hidden-units "$SMOKE_HIDDEN_UNITS" \
    --num-heads "$SMOKE_NUM_HEADS" \
    --device "$SMOKE_DEVICE"

  python -u scripts/score_taac2025.py \
    --result-json "$FINAL_MODEL_DIR/result.json" \
    --truth-json "$SMOKE_DIR/eval/ground_truth.json" \
    --round preliminary \
    --k 10
fi
