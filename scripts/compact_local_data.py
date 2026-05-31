#!/usr/bin/env python3
"""Compact an already converted TAAC dataset.

This rewrites seq.jsonl so user ids, item ids, and sparse feature values are
dense in the destination split. It is useful after cropping from data_full:
copying the full indexer keeps millions of unused embedding rows alive, which
can easily cause CUDA OOM on 8 GB GPUs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


ITEM_SPARSE_FIDS = [
    "100",
    "117",
    "118",
    "101",
    "102",
    "119",
    "120",
    "114",
    "112",
    "121",
    "115",
    "122",
    "116",
]
USER_SPARSE_FIDS = ["103", "104", "105", "109"]
USER_ARRAY_FIDS = ["106", "107", "108", "110"]
ALL_FIDS = ITEM_SPARSE_FIDS + USER_SPARSE_FIDS + USER_ARRAY_FIDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact a converted local TAAC dataset")
    parser.add_argument("--src", required=True, help="Source data dir, e.g. data_32k_from_full")
    parser.add_argument("--dst", required=True, help="Destination data dir, e.g. data_32k_compact")
    parser.add_argument("--eval-users", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class DenseMaps:
    def __init__(self) -> None:
        self.users: Dict[int, int] = {}
        self.items: Dict[int, int] = {}
        self.feats: Dict[str, Dict[int, int]] = {fid: {} for fid in ALL_FIDS}

    @staticmethod
    def _add(mapping: Dict[int, int], old_id: int) -> int:
        if old_id <= 0:
            return 0
        if old_id not in mapping:
            mapping[old_id] = len(mapping) + 1
        return mapping[old_id]

    def user(self, old_id: int) -> int:
        return self._add(self.users, old_id)

    def item(self, old_id: int) -> int:
        return self._add(self.items, old_id)

    def feat_value(self, fid: str, old_value: Any) -> int:
        if old_value is None:
            return 0
        return self._add(self.feats[fid], int(old_value))

    def feat(self, raw_feat: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not raw_feat:
            return raw_feat
        out: Dict[str, Any] = {}
        for fid, value in raw_feat.items():
            if fid not in self.feats:
                out[fid] = value
                continue
            if isinstance(value, list):
                vals = [self.feat_value(fid, v) for v in value if v is not None]
                if vals:
                    out[fid] = vals
            else:
                out[fid] = self.feat_value(fid, value)
        return out

    def indexer(self) -> Dict[str, Any]:
        return {
            "u": {str(old): new for old, new in self.users.items()},
            "i": {str(old): new for old, new in self.items.items()},
            "f": {
                fid: {str(old): new for old, new in mapping.items()}
                for fid, mapping in self.feats.items()
            },
        }


def line_count(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def positive_indices(records: List[List[Any]]) -> List[int]:
    return [
        idx
        for idx, rec in enumerate(records)
        if idx > 0 and rec[1] is not None and rec[4] is not None and int(rec[4]) > 0
    ]


def write_seq(src_seq: Path, dst_seq: Path, maps: DenseMaps, total: int) -> Tuple[List[int], Dict[str, Dict[str, Any]], List[List[List[Any]]]]:
    offsets: List[int] = []
    item_feat_dict: Dict[str, Dict[str, Any]] = {}
    eval_candidates: List[List[List[Any]]] = []

    with open(src_seq, "rb") as fin, open(dst_seq, "wb") as fout:
        for line in tqdm(fin, total=total, desc="Compact seq", unit="user"):
            records = json.loads(line)
            new_records: List[List[Any]] = []
            for rec in records:
                uid, iid, user_feat, item_feat, action, ts = rec
                new_uid = maps.user(int(uid)) if uid is not None else None
                new_iid = maps.item(int(iid)) if iid is not None else None
                new_user_feat = maps.feat(user_feat)
                new_item_feat = maps.feat(item_feat)
                if new_iid is not None and new_item_feat is not None:
                    item_feat_dict[str(new_iid)] = new_item_feat
                new_records.append([new_uid, new_iid, new_user_feat, new_item_feat, action, ts])

            offsets.append(fout.tell())
            fout.write(json.dumps(new_records, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
            if positive_indices(new_records):
                eval_candidates.append(new_records)

    return offsets, item_feat_dict, eval_candidates


def write_eval(dst: Path, maps: DenseMaps, item_feat_dict: Dict[str, Dict[str, Any]], eval_candidates: Iterable[List[List[Any]]], eval_users: int) -> None:
    eval_dir = dst / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    indexer = maps.indexer()
    item_rev = {new: str(old) for old, new in indexer["i"].items()}
    eval_offsets: List[int] = []
    user_action_type: Dict[str, int] = {}
    ground_truth: Dict[str, List[Dict[str, Any]]] = {}
    used = 0

    with open(eval_dir / "predict_seq.jsonl", "wb") as f:
        for records in tqdm(eval_candidates, desc="Write eval seq", unit="user"):
            if used >= eval_users:
                break
            positives = positive_indices(records)
            if not positives:
                continue
            target_idx = positives[-1]
            if target_idx <= 1:
                continue

            history = records[:target_idx]
            target = records[target_idx]
            uid = int(records[0][0])
            target_iid = int(target[1])
            action_type = int(target[4])
            raw_user_key = str(uid)

            eval_offsets.append(f.tell())
            f.write(json.dumps(history, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
            user_action_type[raw_user_key] = action_type
            ground_truth[raw_user_key] = [{"creative_id": item_rev[target_iid], "action_type": action_type}]
            used += 1

    with open(eval_dir / "predict_seq_offsets.pkl", "wb") as f:
        pickle.dump(eval_offsets, f)
    with open(eval_dir / "indexer.pkl", "wb") as f:
        pickle.dump(indexer, f)
    with open(eval_dir / "item_feat_dict.json", "w", encoding="utf-8") as f:
        json.dump(item_feat_dict, f, ensure_ascii=False, separators=(",", ":"))
    with open(eval_dir / "user_action_type.json", "w", encoding="utf-8") as f:
        json.dump(user_action_type, f, ensure_ascii=False, separators=(",", ":"))
    with open(eval_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, separators=(",", ":"))

    with open(eval_dir / "predict_set.jsonl", "w", encoding="utf-8") as f:
        for old_iid, new_iid in tqdm(maps.items.items(), desc="Write predict set", unit="item"):
            feat = item_feat_dict.get(str(new_iid), {})
            rec = {"creative_id": str(old_iid), "features": feat}
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    src_seq = src / "seq.jsonl"
    if not src_seq.exists():
        raise FileNotFoundError(src_seq)
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    total = line_count(src_seq)
    maps = DenseMaps()
    offsets, item_feat_dict, eval_candidates = write_seq(src_seq, dst / "seq.jsonl", maps, total)

    with open(dst / "seq_offsets.pkl", "wb") as f:
        pickle.dump(offsets, f)
    with open(dst / "indexer.pkl", "wb") as f:
        pickle.dump(maps.indexer(), f)
    with open(dst / "item_feat_dict.json", "w", encoding="utf-8") as f:
        json.dump(item_feat_dict, f, ensure_ascii=False, separators=(",", ":"))

    write_eval(dst, maps, item_feat_dict, eval_candidates, args.eval_users)

    meta = {
        "source": str(src.resolve()),
        "users": len(maps.users),
        "items": len(maps.items),
        "eval_users": args.eval_users,
        "note": "Ids are compacted from the source split. creative_id in eval uses the source internal item id.",
    }
    with open(dst / "compact_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done")
    print(f"  src:   {src.resolve()}")
    print(f"  dst:   {dst.resolve()}")
    print(f"  users: {len(maps.users)}")
    print(f"  items: {len(maps.items)}")
    print(f"  eval:  {min(args.eval_users, len(eval_candidates))}")


if __name__ == "__main__":
    main()
