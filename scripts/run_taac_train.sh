#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TAAC_DATA_DIR="${1:-$ROOT_DIR/data_full}"

# Backward-compatible fallback: TAAC_* takes priority, old SMOKE_* still works.
: "${TAAC_NUM_USERS:=${SMOKE_NUM_USERS:-0}}"
: "${TAAC_EVAL_USERS:=${SMOKE_EVAL_USERS:-4096}}"
: "${TAAC_EXTRA_ITEMS:=${SMOKE_EXTRA_ITEMS:-0}}"
: "${TAAC_LOCAL_PARQUET_DIR:=${SMOKE_LOCAL_PARQUET_DIR:-D:/hf_cache/TAAC2025/TencentGR-1M}}"
: "${TAAC_STREAM_DATA:=${SMOKE_STREAM_DATA:-0}}"
: "${TAAC_REBUILD_DATA:=${SMOKE_REBUILD_DATA:-0}}"
: "${TAAC_AUTO_RESUME:=${SMOKE_AUTO_RESUME:-1}}"
: "${TAAC_RESUME_PATH:=${SMOKE_RESUME_PATH:-}}"
: "${TAAC_SAVE_INTERVAL:=${SMOKE_SAVE_INTERVAL:-5000}}"
: "${TAAC_BATCH_SIZE:=${SMOKE_BATCH_SIZE:-232}}"
: "${TAAC_EVAL_BATCH_SIZE:=${SMOKE_EVAL_BATCH_SIZE:-512}}"
: "${TAAC_NUM_WORKERS:=${SMOKE_NUM_WORKERS:-8}}"
: "${TAAC_MAXLEN:=${SMOKE_MAXLEN:-101}}"
: "${TAAC_VAL_RATIO:=${SMOKE_VAL_RATIO:-0.1}}"
: "${TAAC_NUM_EPOCHS:=${SMOKE_NUM_EPOCHS:-4}}"
: "${TAAC_NUM_BLOCKS:=${SMOKE_NUM_BLOCKS:-16}}"
: "${TAAC_HIDDEN_UNITS:=${SMOKE_HIDDEN_UNITS:-128}}"
: "${TAAC_NUM_HEADS:=${SMOKE_NUM_HEADS:-8}}"
: "${TAAC_ITEM_EMB_BATCH_SIZE:=${SMOKE_ITEM_EMB_BATCH_SIZE:-8192}}"
: "${TAAC_EVAL_EACH_EPOCH:=${SMOKE_EVAL_EACH_EPOCH:-1}}"
: "${TAAC_PLOT_CURVES:=${SMOKE_PLOT_CURVES:-1}}"
: "${TAAC_DEVICE:=${SMOKE_DEVICE:-cuda}}"
: "${TAAC_RUN_NAME:=${SMOKE_RUN_NAME:-full_h${TAAC_HIDDEN_UNITS}_b${TAAC_NUM_BLOCKS}_l${TAAC_MAXLEN}_bs${TAAC_BATCH_SIZE}}}"

if [[ "$TAAC_REBUILD_DATA" == "1" || ! -f "$TAAC_DATA_DIR/seq.jsonl" || ! -f "$TAAC_DATA_DIR/indexer.pkl" || ! -f "$TAAC_DATA_DIR/item_feat_dict.json" ]]; then
  if [[ -n "$TAAC_LOCAL_PARQUET_DIR" && -d "$TAAC_LOCAL_PARQUET_DIR/seq" && -d "$TAAC_LOCAL_PARQUET_DIR/user_feat" && -d "$TAAC_LOCAL_PARQUET_DIR/item_feat" ]]; then
    python -u scripts/make_local_parquet_data.py \
      --parquet-root "$TAAC_LOCAL_PARQUET_DIR" \
      --output-dir "$TAAC_DATA_DIR" \
      --num-users "$TAAC_NUM_USERS" \
      --eval-users "$TAAC_EVAL_USERS" \
      --extra-items "$TAAC_EXTRA_ITEMS"
  elif [[ "$TAAC_STREAM_DATA" == "1" ]]; then
    python -u scripts/make_hf_stream_data.py \
      --output-dir "$TAAC_DATA_DIR" \
      --num-users "$TAAC_NUM_USERS" \
      --eval-users "$TAAC_EVAL_USERS" \
      --extra-items "$TAAC_EXTRA_ITEMS"
  else
    python -u scripts/make_hf_smoke_data.py \
      --output-dir "$TAAC_DATA_DIR" \
      --num-users "$TAAC_NUM_USERS" \
      --eval-users "$TAAC_EVAL_USERS" \
      --extra-items "$TAAC_EXTRA_ITEMS"
  fi
else
  echo "[1/5] Reusing existing converted data in $TAAC_DATA_DIR"
  echo "      Set TAAC_REBUILD_DATA=1 to rebuild it."
fi

export TRAIN_DATA_PATH="$TAAC_DATA_DIR"
export TRAIN_CKPT_PATH="$TAAC_DATA_DIR/checkpoints/$TAAC_RUN_NAME"
export TRAIN_LOG_PATH="$TAAC_DATA_DIR/logs/$TAAC_RUN_NAME"
export TRAIN_TF_EVENTS_PATH="$TAAC_DATA_DIR/tf_events/$TAAC_RUN_NAME"
export USER_CACHE_PATH="$TAAC_DATA_DIR/cache"

mkdir -p \
  "$TRAIN_CKPT_PATH" \
  "$TRAIN_LOG_PATH" \
  "$TRAIN_TF_EVENTS_PATH" \
  "$USER_CACHE_PATH"

if [[ "$TAAC_REBUILD_DATA" == "1" || ! -f "$USER_CACHE_PATH/item_freq.csv" || ! -f "$USER_CACHE_PATH/item_last_ts.json" ]]; then
  python -u get_stat.py --data_path "$TRAIN_DATA_PATH"
else
  echo "[stats] Reusing existing cache in $USER_CACHE_PATH"
fi

RESUME_ARGS=()
if [[ -n "$TAAC_RESUME_PATH" ]]; then
  RESUME_ARGS=(--resume "$TAAC_RESUME_PATH")
elif [[ "$TAAC_REBUILD_DATA" != "1" && "$TAAC_AUTO_RESUME" == "1" && -f "$TRAIN_CKPT_PATH/latest/ckpt.pt" ]]; then
  RESUME_ARGS=(--resume "$TRAIN_CKPT_PATH/latest/ckpt.pt")
  echo "[resume] Using latest checkpoint: $TRAIN_CKPT_PATH/latest/ckpt.pt"
else
  echo "[resume] Starting a fresh training run"
fi

EXTRA_TRAIN_ARGS=()
if [[ "$TAAC_EVAL_EACH_EPOCH" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--eval_each_epoch)
fi
if [[ "$TAAC_PLOT_CURVES" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--plot_curves)
fi

python -u main.py \
  --data_path "$TRAIN_DATA_PATH" \
  --ckpt_root "$TRAIN_CKPT_PATH" \
  --log_dir "$TRAIN_LOG_PATH" \
  --tf_dir "$TRAIN_TF_EVENTS_PATH" \
  --batch_size "$TAAC_BATCH_SIZE" \
  --eval_batch_size "$TAAC_EVAL_BATCH_SIZE" \
  --maxlen "$TAAC_MAXLEN" \
  --num_workers "$TAAC_NUM_WORKERS" \
  --val_ratio "$TAAC_VAL_RATIO" \
  --num_epochs "$TAAC_NUM_EPOCHS" \
  --num_blocks "$TAAC_NUM_BLOCKS" \
  --hidden_units "$TAAC_HIDDEN_UNITS" \
  --num_heads "$TAAC_NUM_HEADS" \
  --device "$TAAC_DEVICE" \
  --item_emb_batch_size "$TAAC_ITEM_EMB_BATCH_SIZE" \
  --topk 10 \
  --save_interval "$TAAC_SAVE_INTERVAL" \
  --save_final \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

FINAL_MODEL_DIR=""
if [[ "$TAAC_NUM_USERS" -gt 0 ]]; then
  FINAL_MODEL_DIR="$TRAIN_CKPT_PATH/final_global_step$(( (TAAC_NUM_USERS * 8 / 10) / TAAC_BATCH_SIZE ))"
fi
if [[ -z "$FINAL_MODEL_DIR" || ! -d "$FINAL_MODEL_DIR" ]]; then
  FINAL_MODEL_DIR="$(find "$TRAIN_CKPT_PATH" -maxdepth 1 -type d -name 'final_global_step*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
fi

if [[ -n "$FINAL_MODEL_DIR" && -f "$FINAL_MODEL_DIR/model.pt" ]]; then
  python -u scripts/run_taac_infer.py \
    --model-dir "$FINAL_MODEL_DIR" \
    --eval-data-path "$TAAC_DATA_DIR/eval" \
    --output-json "$FINAL_MODEL_DIR/result.json" \
    --topk 10 \
    --batch-size "$TAAC_EVAL_BATCH_SIZE" \
    --maxlen "$TAAC_MAXLEN" \
    --num-blocks "$TAAC_NUM_BLOCKS" \
    --hidden-units "$TAAC_HIDDEN_UNITS" \
    --num-heads "$TAAC_NUM_HEADS" \
    --device "$TAAC_DEVICE"

  python -u scripts/score_taac2025.py \
    --result-json "$FINAL_MODEL_DIR/result.json" \
    --truth-json "$TAAC_DATA_DIR/eval/ground_truth.json" \
    --round preliminary \
    --k 10
fi
