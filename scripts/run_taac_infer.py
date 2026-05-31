#!/usr/bin/env python3
"""Run this repo's inference and write a baseline-compatible result.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infer import infer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and write TAAC2025 result.json")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.pt")
    parser.add_argument("--eval-data-path", required=True, help="Directory containing predict_seq/predict_set files")
    parser.add_argument("--output-json", default=None, help="Output result.json path")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--user-cache-path", default=None, help="Cache directory containing item_freq.csv")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--maxlen", type=int, default=101)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["MODEL_OUTPUT_PATH"] = args.model_dir
    os.environ["EVAL_DATA_PATH"] = args.eval_data_path
    if args.user_cache_path:
        os.environ["USER_CACHE_PATH"] = args.user_cache_path
    elif "USER_CACHE_PATH" not in os.environ:
        fallback_cache = Path(args.eval_data_path).resolve().parent / "cache"
        os.environ["USER_CACHE_PATH"] = str(fallback_cache)

    sys.argv = [
        "infer.py",
        "--topk",
        str(args.topk),
        "--batch_size",
        str(args.batch_size),
        "--maxlen",
        str(args.maxlen),
        "--num_blocks",
        str(args.num_blocks),
        "--hidden_units",
        str(args.hidden_units),
        "--num_heads",
        str(args.num_heads),
        "--device",
        args.device,
    ]
    top10s, users = infer()
    out_path = Path(args.output_json) if args.output_json else Path(args.model_dir) / "result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "time": int(time.time()),
        "top10s": top10s,
        "user": users,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
