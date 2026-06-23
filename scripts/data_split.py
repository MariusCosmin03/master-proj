"""
RL Data Splitter — Weekly episodes preserving datetime index
=============================================================
Loads a single-column price CSV with a datetime index, slices it into
natural ISO calendar weeks (Mon–Sun), shuffles, and splits into
train / val / test. Each week is saved as a CSV preserving its timestamps.

Data format expected:
    ,prices
    2024-01-01 00:00:00,6.13
    2024-01-01 00:15:00,-0.87
    ...

Usage
-----
  python rl_data_split.py --csv_path your_data.csv
  python rl_data_split.py --csv_path your_data.csv --scale 0.001 --out_dir splits
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_price_series(csv_path: str, price_col: str, scale: float = 1.0) -> pd.Series:
    """
    Load a datetime-indexed CSV and return the price column as a pd.Series.
    The datetime index is preserved throughout the pipeline.
    """
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    series = df[price_col] * scale
    print(f"[load] {len(series):,} timesteps | {series.index[0]} → {series.index[-1]}")
    return series


# ---------------------------------------------------------------------------
# 2. Slice by ISO calendar week (natural Mon–Sun boundaries)
# ---------------------------------------------------------------------------

def slice_into_weeks(series: pd.Series) -> list[pd.Series]:
    """
    Split the series on ISO calendar week boundaries (Mon 00:00 → Sun 23:45).
    Each slice retains its original datetime index.
    Incomplete weeks at the start or end of the year are kept.
    """
    weeks = []
    for (year, week), group in series.groupby(
        [series.index.isocalendar().year, series.index.isocalendar().week]
    ):
        if len(group) > 0:
            weeks.append(group.copy())

    lengths = [len(w) for w in weeks]
    print(f"[slice] {len(weeks)} ISO weeks | "
          f"min={min(lengths)} max={max(lengths)} steps/week")
    return weeks


# ---------------------------------------------------------------------------
# 3. Shuffle + split
# ---------------------------------------------------------------------------

def split_weeks(
    weeks: list[pd.Series],
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int          = 42,
) -> tuple[list[pd.Series], list[pd.Series], list[pd.Series]]:
    """
    Shuffle weeks (safe — each is an independent episode) then split.
    Use the same seed for IDC and DAA to keep week alignment.
    """
    rng = random.Random(seed)
    shuffled = weeks.copy()
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    print(f"[split] train={len(train)} | val={len(val)} | test={len(test)} weeks")
    return train, val, test


# ---------------------------------------------------------------------------
# 4. Save — one CSV per week + manifest
# ---------------------------------------------------------------------------

def save_splits(
    train: list[pd.Series],
    val:   list[pd.Series],
    test:  list[pd.Series],
    out_dir: str = "data_splits",
) -> dict:
    """
    Save each week as its own CSV under:
        data_splits/train/week_000.csv
        data_splits/val/week_000.csv
        data_splits/test/week_000.csv

    Also writes manifest.json with start/end dates and step counts.

    Loading a week later:
        series = pd.read_csv(path, parse_dates=True, index_col=0).iloc[:, 0]
        prices = series.values       # numpy array for the env
        index  = series.index        # DatetimeIndex for the market
    """
    manifest = {"train": [], "val": [], "test": []}

    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        split_dir = Path(out_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for i, week in enumerate(split):
            path = split_dir / f"week_{i:03d}.csv"
            week.to_csv(path, header=True)
            manifest[split_name].append({
                "file":  str(path),
                "start": str(week.index[0]),
                "end":   str(week.index[-1]),
                "steps": len(week),
            })

    manifest_path = Path(out_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    for split_name in ("train", "val", "test"):
        entries    = manifest[split_name]
        total_steps = sum(e["steps"] for e in entries)
        print(f"[save] {split_name:5s} → {len(entries):2d} files | "
              f"{total_steps:,} steps | "
              f"{entries[0]['start'][:10]} → {entries[-1]['end'][:10]}")

    print(f"[save] manifest → {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# 5. Loader helper (use this in your training script)
# ---------------------------------------------------------------------------

def load_split(manifest_path: str, split: str) -> list[pd.Series]:
    """
    Reload a saved split from the manifest as a list of pd.Series.

    Usage in training script:
        from rl_data_split import load_split

        train_weeks = load_split("data_splits/manifest.json", "train")
        val_weeks   = load_split("data_splits/manifest.json", "val")
        test_weeks  = load_split("data_splits/manifest.json", "test")

        # Each week as numpy (prices only):
        prices = train_weeks[0].values

        # Each week with its datetime index (for IdcMarket):
        index  = train_weeks[0].index
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    weeks = []
    for entry in manifest[split]:
        s = pd.read_csv(entry["file"], parse_dates=True, index_col=0).iloc[:, 0]
        weeks.append(s)

    print(f"[load_split] '{split}' — {len(weeks)} weeks loaded")
    return weeks


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a datetime-indexed price CSV into weekly RL episodes."
    )
    parser.add_argument("--csv_path",    default="preprocessed_idc_prices_2024.csv")
    parser.add_argument("--price_col",   default="prices")
    parser.add_argument("--scale",       type=float, default=1.0,
                        help="Scale factor, e.g. 0.001 to divide by 1000")
    parser.add_argument("--out_dir",     default="data_splits/idc")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    series           = load_price_series(args.csv_path, args.price_col, args.scale)
    weeks            = slice_into_weeks(series)
    train, val, test = split_weeks(weeks, args.train_ratio, args.val_ratio, args.seed)
    save_splits(train, val, test, args.out_dir)

    print("\n─── Done ─────────────────────────────────────────────────────────")
    print("  Load in your training script:")
    print("    from rl_data_split import load_split")
    print(f"    train_weeks = load_split('{args.out_dir}/manifest.json', 'train')")
    print(f"    val_weeks   = load_split('{args.out_dir}/manifest.json', 'val')")
    print(f"    test_weeks  = load_split('{args.out_dir}/manifest.json', 'test')")
    print("──────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()