# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sniper v3.0 is a GPU-accelerated FX mean reversion trading system for EURUSD M1 data. It uses Numba CUDA kernels for feature computation and XGBoost for probabilistic trade filtering. The system prioritizes capital preservation (max drawdown) over raw profit.

## Setup

```bash
pip install -r requirements.txt
cp config/settings.py.example config/settings.py  # then fill in real parameters
```

Requires an NVIDIA GPU with CUDA support. Set `XGB_DEVICE = 'cpu'` in `config/settings.py` if no GPU is available.

## Pipeline

The scripts run sequentially and are numbered by execution order:

```bash
# Path A — from Dukascopy tick data (data/processed/YYYY/MM/DD/ticks.parquet)
python scripts/00_ticks_to_m1.py  # Tick → M1 OHLCV Parquet (writes to data/processed/eurusd_m1.parquet)

# Path B — from raw CSV export
python scripts/01_preprocess.py   # CSV → Parquet (reads from data/raw/, writes to data/processed/)

# Common steps
python scripts/02_train_ml.py     # Train XGBoost models per timeframe (1min, 5min, 15min, 30min)
python scripts/03_backtest.py     # Run backtest and generate performance dashboard
```

All scripts are standalone entry points (`if __name__ == "__main__"`). Run from the project root so `core/` and `config/` imports resolve correctly.

## Architecture

- **`config/settings.py`** — Central configuration (paths, strategy params, risk params, XGBoost hyperparams). Gitignored to protect proprietary values; `settings.py.example` has template defaults.
- **`core/kernels.py`** — Numba `@cuda.jit` kernel computing Z-Score and ATR in a single pass on GPU. Called with 1D grid/block layout: `[(n+255)//256, 256]`.
- **`core/metrics.py`** — Post-trade performance metrics (net profit, profit factor, max drawdown).
- **`scripts/02_train_ml.py`** — Resamples M1 data to multiple timeframes, computes GPU features, trains one XGBoost classifier per timeframe. Models saved to `models/<TF>/MREV_<TF>_v1.json`.
- **`scripts/03_backtest.py`** — CPU-side backtest using rolling Z-Score/ATR (not the GPU kernel), loads XGBoost model via `xgb.Booster`, applies triple-barrier exit (ATR-based TP/SL + time barrier), outputs dashboard plot.

## Key Design Notes

- **GPU kernel vs CPU backtest**: `02_train_ml.py` uses the GPU kernel (`core/kernels.py`) for feature engineering. `03_backtest.py` uses `close.rolling().std()` as a proxy for ATR (not true high-low-close ATR) — these two ATR implementations are intentionally different.
- **Train/test split**: temporal split at `2023-01-01` (train on data before, test on data after).
- **Strategy features**: `z_score`, `atr`, `hour`, `day_of_week` — all four are used by the XGBoost classifier.
- **Triple-barrier exit**: TP at `ATR_TP_MULT × ATR`, SL at `ATR_SL_MULT × ATR`, time exit after `HOLD_BARS` bars.
- **All data paths are derived from `PROJECT_ROOT`** (`os.path.dirname` of the script's location). All three scripts resolve paths relative to the project root — no hardcoded absolute paths.
- **`03_backtest.py` loads only the 1min model** — `MODEL_SAVE_PATH` in `config/settings.py` defaults to `models/MREV_1MIN_v1.json`. To backtest another timeframe, update that path.
- **Dashboard output** is saved to `output/plots/sniper_full_dashboard.png` (created automatically).

## Protected Files (gitignored)

- `config/settings.py` — real strategy parameters
- `models/*.json` — trained model weights
- `data/` — raw and processed market data
- `live/logs/` — live trading logs

Do not commit these files or their contents.
