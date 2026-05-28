#!/usr/bin/env python3
"""Run this repo's inference and write a baseline-compatible result.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from infer import infer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and write TAAC2025 result.json")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.pt")
    parser.add_argument("--eval-data-path", required=True, help="Directory containing predict_seq/predict_set files")
    parser.add_argument("--output-json", default=None, help="Output result.json path")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--user-cache-path", default=None, help="Cache directory containing item_freq.csv")
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

    sys.argv = ["infer.py", "--topk", str(args.topk)]
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
