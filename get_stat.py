"""
Flatten seq.jsonl into events and export stats.

Reads: --data_path (defaults to $TRAIN_DATA_PATH); each line is
       a JSON array [user_id, item_id, ..., action, ts].
Writes:
  - $USER_CACHE_PATH/item_freq.csv      # item occurrence counts
  - $USER_CACHE_PATH/item_last_ts.json  # {item_id: last_ts}
Requires: polars
"""

import os
import json
import math
import argparse
from pathlib import Path
import polars as pl
from tqdm import tqdm


def get_args():
    """CLI: --data_path (defaults to $TRAIN_DATA_PATH)."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("TRAIN_DATA_PATH"),
        help="Directory containing seq.jsonl (default: $TRAIN_DATA_PATH)",
    )
    return p.parse_args()


def is_nan_like(x):
    """Return True for blanks/NaN/None-esque values."""
    if x is None:
        return True
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, str):
        s = x.strip().lower()
        return s in {"", "nan", "null", "none", "na", "n/a"}
    return False


def parse_block(line: str):
    """
    Parse one line: a JSON array of events.
    Each event: [user_id, item_id, user_feat, item_feat, action, ts].
    Returns a list of dicts with normalized types.
    """
    events = json.loads(line)
    out = []
    for rec in events:
        if not rec or len(rec) < 6:
            continue
        uid, iid, action, ts = rec[0], rec[1], rec[4], rec[5]
        if is_nan_like(iid):
            continue
        try:
            a = int(float(action))
            t = int(float(ts))
        except (TypeError, ValueError):
            continue
        out.append({
            "user_id": None if is_nan_like(uid) else str(uid),
            "item_id": str(iid),
            "action": a,
            "ts": t,
        })
    return out


def main(data_path: str):
    """Flatten seq.jsonl to a table; write item frequency CSV and last-timestamp JSON."""
    # Resolve paths
    data_dir = Path(data_path)
    cache_dir = os.environ.get("USER_CACHE_PATH")
    if not cache_dir:
        raise EnvironmentError("USER_CACHE_PATH is not set")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seq_jsonl = data_dir / "seq.jsonl"
    item_freq = cache_dir / "item_freq.csv"
    out_json = cache_dir / "item_last_ts.json"

    if not seq_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {seq_jsonl}")

    print(f"Reading: {seq_jsonl}")
    with open(seq_jsonl, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    rows = []
    with open(seq_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Parse seq.jsonl", unit="user"):
            rows.extend(parse_block(line))

    print(f"Parsed {len(rows)} item events from {total_lines} users")
    df = pl.DataFrame(
        rows,
        schema={
            "user_id": pl.Utf8,
            "item_id": pl.Utf8,
            "action": pl.Int64,
            "ts": pl.Int64,
        },
    )

    # (1) item_freq.csv: item occurrence counts (desc), column: occ_total
    print("Aggregating item frequencies")
    item_freq_df = (
        df.group_by("item_id")
          .agg(pl.len().alias("occ_total"))
          .sort("occ_total", descending=True)
    )
    item_freq_df.write_csv(item_freq)
    print(f"Wrote: {item_freq}")

    # (2) item_last_ts_test.json: {item_id: last_ts}
    print("Aggregating item last timestamps")
    item_last = (
        df.select([pl.col("item_id").cast(pl.Utf8), pl.col("ts").cast(pl.Int64)])
          .group_by("item_id")
          .agg(pl.max("ts").alias("last_ts"))
    )
    mapping = {
        row["item_id"]: row["last_ts"]
        for row in tqdm(item_last.iter_rows(named=True), total=item_last.height, desc="Write last_ts", unit="item")
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    print(f"Wrote {len(mapping)} items to: {out_json}")


if __name__ == "__main__":
    args = get_args()
    main(args.data_path)
