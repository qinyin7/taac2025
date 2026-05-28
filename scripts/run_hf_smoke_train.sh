#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_DIR="${1:-$ROOT_DIR/data_smoke}"

: "${SMOKE_NUM_USERS:=256}"
: "${SMOKE_EXTRA_ITEMS:=4096}"
: "${SMOKE_BATCH_SIZE:=8}"
: "${SMOKE_MAXLEN:=32}"
: "${SMOKE_DEVICE:=cuda}"

python -u scripts/make_hf_smoke_data.py \
  --output-dir "$SMOKE_DIR" \
  --num-users "$SMOKE_NUM_USERS" \
  --extra-items "$SMOKE_EXTRA_ITEMS"

export TRAIN_DATA_PATH="$SMOKE_DIR"
export TRAIN_CKPT_PATH="$SMOKE_DIR/checkpoints"
export TRAIN_LOG_PATH="$SMOKE_DIR/logs"
export TRAIN_TF_EVENTS_PATH="$SMOKE_DIR/tf_events"
export USER_CACHE_PATH="$SMOKE_DIR/cache"

mkdir -p \
  "$TRAIN_CKPT_PATH" \
  "$TRAIN_LOG_PATH" \
  "$TRAIN_TF_EVENTS_PATH" \
  "$USER_CACHE_PATH"

python -u get_stat.py --data_path "$TRAIN_DATA_PATH"

python -u main.py \
  --data_path "$TRAIN_DATA_PATH" \
  --ckpt_root "$TRAIN_CKPT_PATH" \
  --log_dir "$TRAIN_LOG_PATH" \
  --tf_dir "$TRAIN_TF_EVENTS_PATH" \
  --batch_size "$SMOKE_BATCH_SIZE" \
  --eval_batch_size 16 \
  --maxlen "$SMOKE_MAXLEN" \
  --num_workers 0 \
  --val_ratio 0.2 \
  --num_epochs 1 \
  --num_blocks 1 \
  --hidden_units 32 \
  --num_heads 4 \
  --device "$SMOKE_DEVICE" \
  --item_emb_batch_size 1024 \
  --topk 10 \
  --save_final

FINAL_MODEL_DIR="$TRAIN_CKPT_PATH/final_global_step$(( (SMOKE_NUM_USERS * 8 / 10) / SMOKE_BATCH_SIZE ))"
if [[ ! -d "$FINAL_MODEL_DIR" ]]; then
  FINAL_MODEL_DIR="$(find "$TRAIN_CKPT_PATH" -maxdepth 1 -type d -name 'final_global_step*' | sort | tail -n 1)"
fi

if [[ -n "$FINAL_MODEL_DIR" && -f "$FINAL_MODEL_DIR/model.pt" ]]; then
  python -u scripts/run_taac_infer.py \
    --model-dir "$FINAL_MODEL_DIR" \
    --eval-data-path "$SMOKE_DIR/eval" \
    --output-json "$FINAL_MODEL_DIR/result.json" \
    --topk 10

  python -u scripts/score_taac2025.py \
    --result-json "$FINAL_MODEL_DIR/result.json" \
    --truth-json "$SMOKE_DIR/eval/ground_truth.json" \
    --round preliminary \
    --k 10
fi
