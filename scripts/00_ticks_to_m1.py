import os
import glob
import time
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'eurusd_m1.parquet')
PRICE_SCALE = 100_000.0  # prices stored as 5-decimal integer pips


def load_all_ticks() -> pd.DataFrame:
    pattern = os.path.join(TICK_DIR, '**', 'ticks.parquet')
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No ticks.parquet files found under {TICK_DIR}")
    print(f"Found {len(files)} daily tick files ({files[0].split(os.sep)[-4]} → {files[-1].split(os.sep)[-4]})")

    chunks = []
    for f in tqdm(files, desc="Loading tick files"):
        chunks.append(pd.read_parquet(f, columns=['timestamp', 'mid', 'ask', 'bid', 'ask_volume', 'bid_volume']))
    return pd.concat(chunks, ignore_index=True)


def ticks_to_m1(df: pd.DataFrame) -> pd.DataFrame:
    df['mid_price'] = df['mid'] / PRICE_SCALE
    df['ask_price'] = df['ask'] / PRICE_SCALE
    df['bid_price'] = df['bid'] / PRICE_SCALE
    df['volume'] = df['ask_volume'] + df['bid_volume']

    df = df.set_index('timestamp').sort_index()

    # Drop timezone so downstream pandas/parquet stays tz-naive (matches M1 sample format)
    df.index = df.index.tz_convert('UTC').tz_localize(None)

    ohlc = df['mid_price'].resample('1min').ohlc()
    ohlc['volume'] = df['volume'].resample('1min').sum()
    ohlc = ohlc.dropna(subset=['open', 'high', 'low', 'close'])

    ohlc.index.name = 'datetime'
    return ohlc


def main():
    t0 = time.time()

    print("Loading tick data...")
    ticks = load_all_ticks()
    print(f"Total ticks loaded: {len(ticks):,}")

    print("Resampling to M1 OHLCV...")
    m1 = ticks_to_m1(ticks)
    print(f"M1 bars generated: {len(m1):,}  ({m1.index[0]} → {m1.index[-1]})")

    # Overwrite only the M1 file, not the daily tick folders
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    m1.to_parquet(OUTPUT_PATH, engine='pyarrow', compression='snappy')
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
