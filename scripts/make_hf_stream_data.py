#!/usr/bin/env python3
"""Build a larger local training set from TAAC2025/TencentGR-1M with bounded sequence memory.

This writes the same files as make_hf_smoke_data.py, but avoids storing all
sampled user sequences in RAM. It scans the HF seq split once to collect user
and item ids, builds compact feature vocabularies, then scans seq again to
write seq.jsonl and eval files.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


DATASET_ID = "TAAC2025/TencentGR-1M"

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
    parser = argparse.ArgumentParser(description="Create a larger local TAAC2025 training set")
    parser.add_argument("--output-dir", default="data_stream", help="Directory to write converted files")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--num-users", type=int, default=0, help="Users to sample; 0 means full seq train split")
    parser.add_argument("--eval-users", type=int, default=4096)
    parser.add_argument("--extra-items", type=int, default=0, help="Extra item rows to include beyond sequence items")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--keep-original-ts", action="store_true")
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def unwrap_feature_value(value: Any) -> Any:
    if isinstance(value, dict) and "feature_value" in value:
        return value.get("feature_value")
    return value


def int_or_default(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    return int(value)


def iter_seq_events(seq_obj: Any) -> Iterable[Tuple[int, int, int]]:
    if isinstance(seq_obj, dict):
        item_ids = seq_obj.get("item_id", [])
        actions = seq_obj.get("action_type", [])
        timestamps = seq_obj.get("timestamp", [])
        for item_id, action, ts in zip(item_ids, actions, timestamps):
            if not is_missing(item_id):
                yield int(item_id), int_or_default(action), int_or_default(ts)
        return

    for event in seq_obj or []:
        if not event:
            continue
        item_id = event.get("item_id")
        if is_missing(item_id):
            continue
        yield int(item_id), int_or_default(event.get("action_type")), int_or_default(event.get("timestamp"))


def row_to_feat(row: Dict[str, Any], fids: List[str]) -> Dict[str, Any]:
    feat: Dict[str, Any] = {}
    for fid in fids:
        value = unwrap_feature_value(row.get(fid))
        if is_missing(value):
            continue
        if isinstance(value, list):
            arr = [v for v in value if not is_missing(v)]
            if arr:
                feat[fid] = arr
        else:
            feat[fid] = value
    return feat


class CompactIndex:
    def __init__(self) -> None:
        self.user: Dict[int, int] = {}
        self.item: Dict[int, int] = {}
        self.feat: Dict[str, Dict[str, int]] = {fid: {} for fid in ALL_FIDS}

    @staticmethod
    def _key(value: Any) -> str:
        return str(value)

    def user_id(self, raw_user_id: int) -> int:
        if raw_user_id not in self.user:
            self.user[raw_user_id] = len(self.user) + 1
        return self.user[raw_user_id]

    def item_id(self, raw_item_id: int) -> int:
        if raw_item_id not in self.item:
            self.item[raw_item_id] = len(self.item) + 1
        return self.item[raw_item_id]

    def feat_value(self, fid: str, raw_value: Any) -> int:
        fmap = self.feat[fid]
        key = self._key(raw_value)
        if key not in fmap:
            fmap[key] = len(fmap) + 1
        return fmap[key]

    def encode_feat(self, raw_feat: Dict[str, Any], fids: List[str]) -> Dict[str, Any]:
        encoded: Dict[str, Any] = {}
        for fid in fids:
            if fid not in raw_feat:
                continue
            value = raw_feat[fid]
            if isinstance(value, list):
                vals = [self.feat_value(fid, v) for v in value if not is_missing(v)]
                if vals:
                    encoded[fid] = vals
            elif not is_missing(value):
                encoded[fid] = self.feat_value(fid, value)
        return encoded

    def to_indexer(self) -> Dict[str, Any]:
        return {
            "u": {str(raw): rid for raw, rid in self.user.items()},
            "i": {str(raw): rid for raw, rid in self.item.items()},
            "f": self.feat,
        }


def load_dataset_config(dataset_id: str, name: str, split: str, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install it with `python -m pip install datasets pyarrow`.") from exc
    return load_dataset(dataset_id, name=name, split=split, cache_dir=cache_dir)


def dataset_len(dataset: Any) -> int | None:
    try:
        return len(dataset)
    except TypeError:
        return None


def seq_split(num_users: int) -> str:
    return "train" if num_users <= 0 else f"train[:{num_users}]"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = seq_split(args.num_users)
    print(f"[1/6] Scanning {args.dataset_id}/seq {split}")
    ds_seq = load_dataset_config(args.dataset_id, "seq", split, args.hf_cache_dir)

    selected_users: set[int] = set()
    needed_items: set[int] = set()
    seq_rows = 0
    for row in tqdm(ds_seq, total=dataset_len(ds_seq), desc="Scan seq", unit="user"):
        events = list(iter_seq_events(row["seq"]))
        if not events:
            continue
        raw_user_id = int(row["user_id"])
        selected_users.add(raw_user_id)
        needed_items.update(item_id for item_id, _, _ in events)
        seq_rows += 1

    if not selected_users:
        raise RuntimeError("No non-empty sequences were sampled.")
    print(f"  users: {len(selected_users)}")
    print(f"  sequence items: {len(needed_items)}")

    print(f"[2/6] Loading user features for {len(selected_users)} users")
    ds_user = load_dataset_config(args.dataset_id, "user_feat", "train", args.hf_cache_dir)
    user_feat_raw: Dict[int, Dict[str, Any]] = {}
    with tqdm(ds_user, total=dataset_len(ds_user), desc="Scan user_feat", unit="row") as pbar:
        for row in pbar:
            raw_user_id = int(row["user_id"])
            if raw_user_id in selected_users:
                user_feat_raw[raw_user_id] = row_to_feat(row, USER_SPARSE_FIDS + USER_ARRAY_FIDS)
                pbar.set_postfix(found=f"{len(user_feat_raw)}/{len(selected_users)}")
                if len(user_feat_raw) == len(selected_users):
                    break

    print(f"[3/6] Loading item features for sequence items plus {args.extra_items} extras")
    ds_item = load_dataset_config(args.dataset_id, "item_feat", "train", args.hf_cache_dir)
    compact = CompactIndex()
    item_feat_dict: Dict[str, Dict[str, Any]] = {}
    extra_count = 0
    needed_found = 0
    with tqdm(ds_item, total=dataset_len(ds_item), desc="Scan item_feat", unit="row") as pbar:
        for row in pbar:
            raw_item_id = int(row["item_id"])
            take = raw_item_id in needed_items or extra_count < args.extra_items
            if not take:
                continue
            iid = compact.item_id(raw_item_id)
            if str(iid) in item_feat_dict:
                continue
            raw_feat = row_to_feat(row, ITEM_SPARSE_FIDS)
            item_feat_dict[str(iid)] = compact.encode_feat(raw_feat, ITEM_SPARSE_FIDS)
            if raw_item_id in needed_items:
                needed_found += 1
            else:
                extra_count += 1
            pbar.set_postfix(needed=f"{needed_found}/{len(needed_items)}", extras=f"{extra_count}/{args.extra_items}")
            if needed_items.issubset(compact.item.keys()) and extra_count >= args.extra_items:
                break

    missing_items = needed_items - compact.item.keys()
    if missing_items:
        print(f"[warn] Missing item_feat rows for {len(missing_items)} sequence items; defaults will be used.")

    print("[4/6] Writing train files")
    offsets: List[int] = []
    recent_floor = 1_748_454_400
    seq_path = out_dir / "seq.jsonl"
    ds_seq = load_dataset_config(args.dataset_id, "seq", split, args.hf_cache_dir)
    with open(seq_path, "wb") as f:
        for row in tqdm(ds_seq, total=dataset_len(ds_seq), desc="Write train seq", unit="user"):
            events = list(iter_seq_events(row["seq"]))
            if not events:
                continue
            raw_user_id = int(row["user_id"])
            uid = compact.user_id(raw_user_id)
            user_feat = compact.encode_feat(user_feat_raw.get(raw_user_id, {}), USER_SPARSE_FIDS + USER_ARRAY_FIDS)
            records: List[List[Any]] = [[uid, None, user_feat, None, None, 0]]
            for raw_item_id, action, ts in events:
                iid = compact.item_id(raw_item_id)
                item_feat = item_feat_dict.get(str(iid), {})
                out_ts = int(ts if args.keep_original_ts else max(ts, recent_floor))
                records.append([uid, iid, None, item_feat, int(action), out_ts])
            offsets.append(f.tell())
            f.write(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")

    with open(out_dir / "seq_offsets.pkl", "wb") as f:
        pickle.dump(offsets, f)
    with open(out_dir / "indexer.pkl", "wb") as f:
        pickle.dump(compact.to_indexer(), f)
    with open(out_dir / "item_feat_dict.json", "w", encoding="utf-8") as f:
        json.dump(item_feat_dict, f, ensure_ascii=False, separators=(",", ":"))

    print("[5/6] Building eval files")
    eval_rows = []
    ds_seq = load_dataset_config(args.dataset_id, "seq", split, args.hf_cache_dir)
    for row in tqdm(ds_seq, total=dataset_len(ds_seq), desc="Build eval rows", unit="user"):
        if len(eval_rows) >= args.eval_users:
            break
        events = list(iter_seq_events(row["seq"]))
        positives = [idx for idx, (_, action, _) in enumerate(events) if int(action) > 0]
        if not positives:
            continue
        target_idx = positives[-1]
        if target_idx == 0:
            continue
        raw_item_id, action, _ = events[target_idx]
        eval_rows.append(
            {
                "user_id": int(row["user_id"]),
                "history": events[:target_idx],
                "target_item_id": raw_item_id,
                "target_action_type": int(action),
            }
        )

    if eval_rows:
        eval_dir = out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "indexer.pkl", "wb") as f:
            pickle.dump(compact.to_indexer(), f)
        with open(eval_dir / "item_feat_dict.json", "w", encoding="utf-8") as f:
            json.dump(item_feat_dict, f, ensure_ascii=False, separators=(",", ":"))

        eval_offsets: List[int] = []
        user_action_type: Dict[str, int] = {}
        ground_truth: Dict[str, List[Dict[str, Any]]] = {}
        with open(eval_dir / "predict_seq.jsonl", "wb") as f:
            for row in tqdm(eval_rows, total=len(eval_rows), desc="Write eval seq", unit="user"):
                uid = compact.user_id(row["user_id"])
                raw_user_key = str(row["user_id"])
                user_feat = compact.encode_feat(user_feat_raw.get(row["user_id"], {}), USER_SPARSE_FIDS + USER_ARRAY_FIDS)
                records: List[List[Any]] = [[uid, None, user_feat, None, None, 0]]
                for raw_item_id, action, ts in row["history"]:
                    iid = compact.item_id(raw_item_id)
                    item_feat = item_feat_dict.get(str(iid), {})
                    out_ts = int(ts if args.keep_original_ts else max(ts, recent_floor))
                    records.append([uid, iid, None, item_feat, int(action), out_ts])
                eval_offsets.append(f.tell())
                f.write(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")

                action_type = int(row["target_action_type"])
                user_action_type[raw_user_key] = action_type
                ground_truth[raw_user_key] = [{"creative_id": str(row["target_item_id"]), "action_type": action_type}]

        with open(eval_dir / "predict_seq_offsets.pkl", "wb") as f:
            pickle.dump(eval_offsets, f)
        with open(eval_dir / "user_action_type.json", "w", encoding="utf-8") as f:
            json.dump(user_action_type, f, ensure_ascii=False, separators=(",", ":"))
        with open(eval_dir / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, ensure_ascii=False, separators=(",", ":"))

        candidate_raw_ids = []
        seen_cands = set()
        for row in eval_rows:
            raw_item_id = row["target_item_id"]
            if raw_item_id not in seen_cands:
                seen_cands.add(raw_item_id)
                candidate_raw_ids.append(raw_item_id)
        for raw_item_id in compact.item:
            if raw_item_id in seen_cands:
                continue
            seen_cands.add(raw_item_id)
            candidate_raw_ids.append(raw_item_id)

        with open(eval_dir / "predict_set.jsonl", "w", encoding="utf-8") as f:
            for raw_item_id in tqdm(candidate_raw_ids, total=len(candidate_raw_ids), desc="Write predict set", unit="item"):
                iid = compact.item_id(raw_item_id)
                feat = item_feat_dict.get(str(iid), {})
                rec = {"creative_id": str(raw_item_id), "features": feat}
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    print("[6/6] Done")
    print(f"  output_dir: {out_dir.resolve()}")
    print(f"  users:      {len(offsets)}")
    print(f"  items:      {len(compact.item)}")
    print(f"  seq lines:  {len(offsets)}")
    if eval_rows:
        print(f"  eval_dir:   {(out_dir / 'eval').resolve()}")
        print(f"  eval users: {len(eval_rows)}")


if __name__ == "__main__":
    main()
