"""
Sniper v3.0 — CSV Preprocessor

Reads Dukascopy M1 CSV files from data/raw/ and combines them into a single
parquet file at data/processed/eurusd_m1.parquet.

Supports two CSV formats:
  A) Dukascopy download:  timestamp,open,high,low,close,volume  (lowercase)
  B) Legacy combined:     Timestamp,Open,High,Low,Close,Volume  (capitalized)

Multiple files are auto-detected via glob and concatenated chronologically.

Usage:
    python scripts/01_preprocess.py
    python scripts/01_preprocess.py --raw-dir /path/to/csvs
"""
import os
import sys
import glob
import time
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(_PROJECT_ROOT, 'data', 'raw')
PROCESSED_PATH = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'eurusd_m1.parquet')


def read_csv_auto(path: str) -> pd.DataFrame:
    """Read a single CSV, auto-detecting column name casing."""
    df = pd.read_csv(path, nrows=0)
    cols = set(df.columns.str.lower())

    if 'timestamp' in cols:
        date_col = [c for c in df.columns if c.lower() == 'timestamp'][0]
    elif 'datetime' in cols:
        date_col = [c for c in df.columns if c.lower() == 'datetime'][0]
    elif 'time' in cols:
        date_col = [c for c in df.columns if c.lower() == 'time'][0]
    else:
        date_col = df.columns[0]

    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
    df.columns  = [c.lower() for c in df.columns]
    df.index.name = 'datetime'

    # Ensure required columns exist
    for col in ('open', 'high', 'low', 'close'):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")

    if 'volume' not in df.columns:
        df['volume'] = 0.0

    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Strip timezone if present (normalize to naive UTC)
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)

    return df


def preprocess_data(raw_dir: str, output_path: str):
    start_time = time.time()

    # Find all CSVs
    csv_files = sorted(
        glob.glob(os.path.join(raw_dir, '**', '*.csv'), recursive=True)
    )
    if not csv_files:
        print(f"❌ No CSV files found in {raw_dir}")
        return

    print(f"📂 Found {len(csv_files)} CSV file(s) in {raw_dir}")
    for f in csv_files:
        print(f"   {os.path.relpath(f, _PROJECT_ROOT)}")

    # Read and concatenate
    frames = []
    for path in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = read_csv_auto(path)
            frames.append(df)
        except Exception as e:
            print(f"   ⚠️  Skipping {os.path.basename(path)}: {e}")

    if not frames:
        print("❌ No valid data loaded.")
        return

    df = pd.concat(frames)
    df.sort_index(inplace=True)

    # Drop exact duplicate timestamps (keep last — more recent data wins)
    before = len(df)
    df = df[~df.index.duplicated(keep='last')]
    dupes = before - len(df)
    if dupes:
        print(f"   Removed {dupes:,} duplicate timestamps")

    # Forward-fill gaps (e.g. missing bars during low-liquidity)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Cast to float32 for smaller parquet
    for col in df.columns:
        df[col] = df[col].astype('float32')

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')

    print(f"\n✅ Combined dataset: {len(df):,} bars")
    print(f"   Range: {df.index[0]} → {df.index[-1]}")
    print(f"   Saved: {output_path}")
    print(f"⚡ Time: {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Combine Dukascopy M1 CSVs into parquet.')
    parser.add_argument('--raw-dir', type=str, default=RAW_DIR,
                        help=f'Directory containing CSV files (default: {RAW_DIR})')
    parser.add_argument('--output', type=str, default=PROCESSED_PATH,
                        help=f'Output parquet path (default: {PROCESSED_PATH})')
    args = parser.parse_args()

    preprocess_data(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
