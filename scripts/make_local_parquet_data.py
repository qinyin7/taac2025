#!/usr/bin/env python3
"""Convert local TAAC2025 parquet files into this repo's seq.jsonl training format.

Expected local parquet layout:
  <parquet-root>/seq/*.parquet
  <parquet-root>/user_feat/*.parquet
  <parquet-root>/item_feat/*.parquet

This script never calls the Hugging Face Hub. It scans local parquet files with
pyarrow batches and writes the same output files as make_hf_smoke_data.py.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import pyarrow.dataset as ds
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
    parser = argparse.ArgumentParser(description="Create local training data from downloaded TAAC2025 parquet")
    parser.add_argument("--parquet-root", required=True, help="Directory containing seq/user_feat/item_feat parquet dirs")
    parser.add_argument("--output-dir", default="data_full")
    parser.add_argument("--num-users", type=int, default=0, help="Users to sample from seq; 0 means all local seq rows")
    parser.add_argument("--eval-users", type=int, default=4096)
    parser.add_argument("--extra-items", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8192, help="PyArrow scan batch size")
    parser.add_argument("--keep-original-ts", action="store_true")
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def int_or_default(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    return int(value)


def iter_seq_events(seq_obj: Any) -> Iterable[Tuple[int, int, int]]:
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
        value = row.get(fid)
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


def parquet_dataset(parquet_root: Path, name: str) -> ds.Dataset:
    path = parquet_root / name
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet directory: {path}")
    return ds.dataset(str(path), format="parquet")


def dataset_rows(dataset: ds.Dataset) -> int | None:
    try:
        return dataset.count_rows()
    except Exception:
        return None


def iter_rows(
    dataset: ds.Dataset,
    columns: List[str],
    batch_size: int,
    limit: int = 0,
) -> Iterator[Dict[str, Any]]:
    yielded = 0
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        for row in batch.to_pylist():
            if limit > 0 and yielded >= limit:
                return
            yielded += 1
            yield row


def main() -> None:
    args = parse_args()
    parquet_root = Path(args.parquet_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_ds = parquet_dataset(parquet_root, "seq")
    user_ds = parquet_dataset(parquet_root, "user_feat")
    item_ds = parquet_dataset(parquet_root, "item_feat")
    seq_total = dataset_rows(seq_ds)

    scan_total = min(seq_total, args.num_users) if seq_total is not None and args.num_users > 0 else seq_total
    limit_msg = "all" if args.num_users <= 0 else str(args.num_users)
    print(f"[1/6] Scanning local seq parquet users={limit_msg}")

    selected_users: set[int] = set()
    needed_items: set[int] = set()
    for row in tqdm(
        iter_rows(seq_ds, ["user_id", "seq"], args.batch_size, limit=args.num_users),
        total=scan_total,
        desc="Scan seq",
        unit="user",
    ):
        events = list(iter_seq_events(row["seq"]))
        if not events:
            continue
        raw_user_id = int(row["user_id"])
        selected_users.add(raw_user_id)
        needed_items.update(item_id for item_id, _, _ in events)

    if not selected_users:
        raise RuntimeError("No non-empty sequences were sampled.")
    print(f"  users: {len(selected_users)}")
    print(f"  sequence items: {len(needed_items)}")

    print(f"[2/6] Loading local user features for {len(selected_users)} users")
    user_feat_raw: Dict[int, Dict[str, Any]] = {}
    user_total = dataset_rows(user_ds)
    with tqdm(total=user_total, desc="Scan user_feat", unit="row") as pbar:
        for row in iter_rows(user_ds, ["user_id"] + USER_SPARSE_FIDS + USER_ARRAY_FIDS, args.batch_size):
            pbar.update(1)
            raw_user_id = int(row["user_id"])
            if raw_user_id in selected_users:
                user_feat_raw[raw_user_id] = row_to_feat(row, USER_SPARSE_FIDS + USER_ARRAY_FIDS)
                pbar.set_postfix(found=f"{len(user_feat_raw)}/{len(selected_users)}")
                if len(user_feat_raw) == len(selected_users):
                    break

    print(f"[3/6] Loading local item features for sequence items plus {args.extra_items} extras")
    compact = CompactIndex()
    item_feat_dict: Dict[str, Dict[str, Any]] = {}
    extra_count = 0
    needed_found = 0
    item_total = dataset_rows(item_ds)
    with tqdm(total=item_total, desc="Scan item_feat", unit="row") as pbar:
        for row in iter_rows(item_ds, ["item_id"] + ITEM_SPARSE_FIDS, args.batch_size):
            pbar.update(1)
            raw_item_id = int(row["item_id"])
            take = raw_item_id in needed_items or extra_count < args.extra_items
            if not take:
                continue
            if raw_item_id in compact.item:
                continue
            iid = compact.item_id(raw_item_id)
            raw_feat = row_to_feat(row, ITEM_SPARSE_FIDS)
            item_feat_dict[str(iid)] = compact.encode_feat(raw_feat, ITEM_SPARSE_FIDS)
            if raw_item_id in needed_items:
                needed_found += 1
            else:
                extra_count += 1
            pbar.set_postfix(needed=f"{needed_found}/{len(needed_items)}", extras=f"{extra_count}/{args.extra_items}")
            if needed_found >= len(needed_items) and extra_count >= args.extra_items:
                break

    missing_items = needed_items - compact.item.keys()
    if missing_items:
        print(f"[warn] Missing item_feat rows for {len(missing_items)} sequence items; defaults will be used.")

    print("[4/6] Writing train files")
    offsets: List[int] = []
    recent_floor = 1_748_454_400
    seq_path = out_dir / "seq.jsonl"
    with open(seq_path, "wb") as f:
        for row in tqdm(
            iter_rows(seq_ds, ["user_id", "seq"], args.batch_size, limit=args.num_users),
            total=scan_total,
            desc="Write train seq",
            unit="user",
        ):
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
    for row in tqdm(
        iter_rows(seq_ds, ["user_id", "seq"], args.batch_size, limit=args.num_users),
        total=scan_total,
        desc="Build eval rows",
        unit="user",
    ):
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

    meta = {
        "parquet_root": str(parquet_root.resolve()),
        "num_users": args.num_users,
        "eval_users": args.eval_users,
        "extra_items": args.extra_items,
        "users_written": len(offsets),
        "items": len(compact.item),
    }
    with open(out_dir / "build_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

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
