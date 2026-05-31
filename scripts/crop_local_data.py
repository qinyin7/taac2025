#!/usr/bin/env python3
"""Crop a local converted TAAC dataset without touching Hugging Face or parquet.

The cropped dataset keeps the source indexer/item feature vocabulary so model
IDs remain compatible. It only rewrites seq.jsonl/seq_offsets.pkl and copies the
eval files, then get_stat.py can rebuild cache statistics for the cropped train
split.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop local converted data to a smaller training split")
    parser.add_argument("--src", required=True, help="Source converted data directory, e.g. data_full")
    parser.add_argument("--dst", required=True, help="Destination directory, e.g. data_32k_from_full")
    parser.add_argument("--num-users", type=int, required=True)
    parser.add_argument("--mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_offsets(path: Path) -> List[int]:
    with open(path, "rb") as f:
        return pickle.load(f)


def copy_required_metadata(src: Path, dst: Path) -> None:
    for name in ["indexer.pkl", "item_feat_dict.json"]:
        shutil.copy2(src / name, dst / name)

    src_eval = src / "eval"
    if src_eval.exists():
        dst_eval = dst / "eval"
        if dst_eval.exists():
            shutil.rmtree(dst_eval)
        shutil.copytree(src_eval, dst_eval)

    meta = {
        "source": str(src.resolve()),
        "note": "seq.jsonl was cropped locally; indexer/item_feat_dict/eval were copied from source.",
    }
    with open(dst / "crop_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    required = ["seq.jsonl", "seq_offsets.pkl", "indexer.pkl", "item_feat_dict.json"]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files under {src}: {missing}")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    offsets = load_offsets(src / "seq_offsets.pkl")
    total = len(offsets)
    n = min(args.num_users, total)
    if args.mode == "random":
        rng = random.Random(args.seed)
        selected = sorted(rng.sample(range(total), n))
    else:
        selected = list(range(n))

    out_offsets: List[int] = []
    src_seq = src / "seq.jsonl"
    dst_seq = dst / "seq.jsonl"
    with open(src_seq, "rb") as fin, open(dst_seq, "wb") as fout:
        for idx in tqdm(selected, total=len(selected), desc="Copy seq lines", unit="user"):
            fin.seek(offsets[idx])
            line = fin.readline()
            out_offsets.append(fout.tell())
            fout.write(line)

    with open(dst / "seq_offsets.pkl", "wb") as f:
        pickle.dump(out_offsets, f)

    copy_required_metadata(src, dst)

    print("Done")
    print(f"  src:        {src.resolve()}")
    print(f"  dst:        {dst.resolve()}")
    print(f"  users:      {len(out_offsets)}")
    print(f"  mode:       {args.mode}")


if __name__ == "__main__":
    main()
