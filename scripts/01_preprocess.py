import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm

# --- SETTINGS ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV_PATH = os.path.join(_PROJECT_ROOT, 'data', 'raw', 'EURUSD_M1_Combined_2015_2026.csv')
PROCESSED_PATH = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'eurusd_m1.parquet')

def preprocess_data():
    start_time = time.time()

    if not os.path.exists(RAW_CSV_PATH):
        print(f"❌ Error: File not found at {RAW_CSV_PATH}")
        return

    print(f"🚀 Reading EURUSD M1 file (Timestamp column detected)")

    dtype_dict = {
        'Open': 'float32',
        'High': 'float32',
        'Low': 'float32',
        'Close': 'float32',
        'Volume': 'float32'
    }

    chunks = []
    try:
        # Use 'Timestamp' as the datetime index column
        reader = pd.read_csv(RAW_CSV_PATH,
                            parse_dates=['Timestamp'],
                            index_col='Timestamp',
                            dtype=dtype_dict,
                            chunksize=500000,
                            engine='c')

        for chunk in tqdm(reader, desc="Loading Data"):
            chunks.append(chunk)

        df = pd.concat(chunks)

        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df.index.name = 'datetime'

        # Sort chronologically
        df.sort_index(inplace=True)

        print(f"✅ Loaded and sorted successfully: {len(df):,} rows")

        # --- Basic Feature Engineering ---
        # Forward fill any missing values
        df.ffill(inplace=True)

        # Compute log return for statistical analysis
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Drop NaN rows
        df.dropna(inplace=True)

        # Save as Parquet (Snappy compression balances speed and file size)
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        df.to_parquet(PROCESSED_PATH, engine='pyarrow', compression='snappy')

        print(f"📦 Parquet saved successfully: {PROCESSED_PATH}")
        print(f"⚡ Total time: {time.time() - start_time:.2f}s")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    preprocess_data()
