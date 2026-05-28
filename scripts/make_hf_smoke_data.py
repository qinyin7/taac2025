#!/usr/bin/env python3
"""Build a tiny local training set from TAAC2025/TencentGR-1M.

The main training code in this repo expects:
  - seq.jsonl
  - seq_offsets.pkl
  - indexer.pkl
  - item_feat_dict.json

The public Hugging Face dataset stores the same information as parquet configs.
This script samples a small number of users, compacts item/user/feature ids, and
writes the minimal files needed for a quick connectivity test.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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
    parser = argparse.ArgumentParser(description="Create a tiny smoke-test dataset from TencentGR-1M")
    parser.add_argument("--output-dir", default="data_smoke", help="Directory to write converted files")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--num-users", type=int, default=256, help="Number of user sequences to sample")
    parser.add_argument(
        "--eval-users",
        type=int,
        default=64,
        help="Number of sampled users with a held-out positive event to export for local TAAC scoring",
    )
    parser.add_argument(
        "--extra-items",
        type=int,
        default=4096,
        help="Extra item feature rows to include so negative sampling has a wider pool",
    )
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face datasets cache directory")
    parser.add_argument(
        "--keep-original-ts",
        action="store_true",
        help="Keep original timestamps. By default timestamps are forced after 2025-05-29 for smoke negative sampling.",
    )
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def unwrap_feature_value(value: Any) -> Any:
    """Support raw parquet values and candidate-style {feature_value: ...} dicts."""
    if isinstance(value, dict) and "feature_value" in value:
        return value.get("feature_value")
    return value


def iter_seq_events(seq_obj: Any) -> Iterable[Tuple[int, int, int]]:
    """Yield (item_id, action_type, timestamp) from HF list-of-dict or dict-of-list layouts."""
    if isinstance(seq_obj, dict):
        item_ids = seq_obj.get("item_id", [])
        actions = seq_obj.get("action_type", [])
        timestamps = seq_obj.get("timestamp", [])
        for item_id, action, ts in zip(item_ids, actions, timestamps):
            if not is_missing(item_id):
                yield int(item_id), int(action), int(ts)
        return

    for event in seq_obj or []:
        if not event:
            continue
        item_id = event.get("item_id")
        if is_missing(item_id):
            continue
        yield int(item_id), int(event.get("action_type", 0)), int(event.get("timestamp", 0))


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
        raise SystemExit(
            "Missing dependency: datasets. Install it with `python -m pip install datasets pyarrow`."
        ) from exc
    return load_dataset(dataset_id, name=name, split=split, cache_dir=cache_dir)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_split = f"train[:{args.num_users}]"
    print(f"[1/5] Loading {args.dataset_id}/seq {seq_split}")
    ds_seq = load_dataset_config(args.dataset_id, "seq", seq_split, args.hf_cache_dir)

    selected_rows: List[Dict[str, Any]] = []
    selected_users: set[int] = set()
    needed_items: set[int] = set()
    for row in ds_seq:
        events = list(iter_seq_events(row["seq"]))
        if not events:
            continue
        raw_user_id = int(row["user_id"])
        selected_rows.append({"user_id": raw_user_id, "events": events})
        selected_users.add(raw_user_id)
        needed_items.update(item_id for item_id, _, _ in events)

    if not selected_rows:
        raise RuntimeError("No non-empty sequences were sampled.")

    print(f"[2/5] Loading user features for {len(selected_users)} users")
    ds_user = load_dataset_config(args.dataset_id, "user_feat", "train", args.hf_cache_dir)
    user_feat_raw: Dict[int, Dict[str, Any]] = {}
    for row in ds_user:
        raw_user_id = int(row["user_id"])
        if raw_user_id in selected_users:
            user_feat_raw[raw_user_id] = row_to_feat(row, USER_SPARSE_FIDS + USER_ARRAY_FIDS)
            if len(user_feat_raw) == len(selected_users):
                break

    print(f"[3/5] Loading item features for sequence items plus {args.extra_items} extras")
    ds_item = load_dataset_config(args.dataset_id, "item_feat", "train", args.hf_cache_dir)
    item_feat_raw: Dict[int, Dict[str, Any]] = {}
    extra_count = 0
    for row in ds_item:
        raw_item_id = int(row["item_id"])
        take = raw_item_id in needed_items or extra_count < args.extra_items
        if not take:
            continue
        if raw_item_id not in item_feat_raw:
            item_feat_raw[raw_item_id] = row_to_feat(row, ITEM_SPARSE_FIDS)
            if raw_item_id not in needed_items:
                extra_count += 1
        if needed_items.issubset(item_feat_raw.keys()) and extra_count >= args.extra_items:
            break

    missing_items = needed_items - item_feat_raw.keys()
    if missing_items:
        print(f"[warn] Missing item_feat rows for {len(missing_items)} sequence items; defaults will be used.")

    print("[4/5] Compacting ids and writing files")
    compact = CompactIndex()
    item_feat_dict: Dict[str, Dict[str, Any]] = {}

    # Register extra items first too, so they can appear in item_feat_dict and negative pools if copied later.
    for raw_item_id, raw_feat in item_feat_raw.items():
        iid = compact.item_id(raw_item_id)
        item_feat_dict[str(iid)] = compact.encode_feat(raw_feat, ITEM_SPARSE_FIDS)

    offsets: List[int] = []
    recent_floor = 1_748_454_400  # 2025-05-30 00:00:00 UTC; keeps smoke negatives "active".
    seq_path = out_dir / "seq.jsonl"
    with open(seq_path, "wb") as f:
        for row in selected_rows:
            uid = compact.user_id(row["user_id"])
            user_feat = compact.encode_feat(user_feat_raw.get(row["user_id"], {}), USER_SPARSE_FIDS + USER_ARRAY_FIDS)

            records: List[List[Any]] = [[uid, None, user_feat, None, None, 0]]
            for raw_item_id, action, ts in row["events"]:
                iid = compact.item_id(raw_item_id)
                raw_feat = item_feat_raw.get(raw_item_id, {})
                item_feat = compact.encode_feat(raw_feat, ITEM_SPARSE_FIDS)
                item_feat_dict.setdefault(str(iid), item_feat)
                out_ts = int(ts if args.keep_original_ts else max(ts, recent_floor))
                records.append([uid, iid, None, item_feat, int(action), out_ts])

            offsets.append(f.tell())
            line = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            f.write(line + b"\n")

    with open(out_dir / "seq_offsets.pkl", "wb") as f:
        pickle.dump(offsets, f)
    with open(out_dir / "indexer.pkl", "wb") as f:
        pickle.dump(compact.to_indexer(), f)
    with open(out_dir / "item_feat_dict.json", "w", encoding="utf-8") as f:
        json.dump(item_feat_dict, f, ensure_ascii=False, separators=(",", ":"))

    eval_rows = []
    for row in selected_rows:
        if len(eval_rows) >= args.eval_users:
            break
        positives = [idx for idx, (_, action, _) in enumerate(row["events"]) if int(action) > 0]
        if not positives:
            continue
        target_idx = positives[-1]
        if target_idx == 0:
            continue
        raw_item_id, action, _ = row["events"][target_idx]
        eval_rows.append(
            {
                "user_id": row["user_id"],
                "history": row["events"][:target_idx],
                "target_item_id": raw_item_id,
                "target_action_type": int(action),
            }
        )

    if eval_rows:
        eval_dir = out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # The inference code expects the same vocab and item feature dictionary under EVAL_DATA_PATH.
        with open(eval_dir / "indexer.pkl", "wb") as f:
            pickle.dump(compact.to_indexer(), f)
        with open(eval_dir / "item_feat_dict.json", "w", encoding="utf-8") as f:
            json.dump(item_feat_dict, f, ensure_ascii=False, separators=(",", ":"))

        eval_offsets: List[int] = []
        user_action_type: Dict[str, int] = {}
        ground_truth: Dict[str, List[Dict[str, Any]]] = {}
        predict_seq_path = eval_dir / "predict_seq.jsonl"
        with open(predict_seq_path, "wb") as f:
            for row in eval_rows:
                uid = compact.user_id(row["user_id"])
                raw_user_key = str(row["user_id"])
                user_feat = compact.encode_feat(
                    user_feat_raw.get(row["user_id"], {}), USER_SPARSE_FIDS + USER_ARRAY_FIDS
                )
                records: List[List[Any]] = [[uid, None, user_feat, None, None, 0]]
                for raw_item_id, action, ts in row["history"]:
                    iid = compact.item_id(raw_item_id)
                    raw_feat = item_feat_raw.get(raw_item_id, {})
                    item_feat = compact.encode_feat(raw_feat, ITEM_SPARSE_FIDS)
                    out_ts = int(ts if args.keep_original_ts else max(ts, recent_floor))
                    records.append([uid, iid, None, item_feat, int(action), out_ts])

                eval_offsets.append(f.tell())
                f.write(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")

                action_type = int(row["target_action_type"])
                user_action_type[raw_user_key] = action_type
                ground_truth[raw_user_key] = [
                    {
                        "creative_id": str(row["target_item_id"]),
                        "action_type": action_type,
                    }
                ]

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
        for raw_item_id in item_feat_raw:
            if raw_item_id in seen_cands:
                continue
            seen_cands.add(raw_item_id)
            candidate_raw_ids.append(raw_item_id)

        with open(eval_dir / "predict_set.jsonl", "w", encoding="utf-8") as f:
            for raw_item_id in candidate_raw_ids:
                iid = compact.item_id(raw_item_id)
                feat = item_feat_dict.get(str(iid), {})
                rec = {"creative_id": str(raw_item_id), "features": feat}
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    print("[5/5] Done")
    print(f"  output_dir: {out_dir.resolve()}")
    print(f"  users:      {len(selected_rows)}")
    print(f"  items:      {len(compact.item)}")
    print(f"  seq lines:  {len(offsets)}")
    if eval_rows:
        print(f"  eval_dir:   {(out_dir / 'eval').resolve()}")
        print(f"  eval users: {len(eval_rows)}")


if __name__ == "__main__":
    main()
