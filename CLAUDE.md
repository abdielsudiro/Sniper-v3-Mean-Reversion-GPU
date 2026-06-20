# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sniper v3.0 is a GPU-accelerated FX mean reversion trading system for EURUSD M1 data. It uses Numba CUDA kernels for feature computation and XGBoost for probabilistic trade filtering. The system prioritizes capital preservation (max drawdown) over raw profit.

## Setup

```bash
pip install -r requirements.txt
cp config/settings.py.example config/settings.py
cp .env.example .env  # then tune or run 04_optimize.py to auto-fill
```

Requires an NVIDIA GPU with CUDA support for the Numba feature kernel. XGBoost runs on CPU by default (`XGB_DEVICE=cpu` in `.env`) — GPU requires CUDA 12+.

## Pipeline

The scripts run sequentially and are numbered by execution order:

```bash
# Path A — from Dukascopy tick data (data/processed/YYYY/MM/DD/ticks.parquet)
python scripts/00_ticks_to_m1.py  # Tick → M1 OHLCV Parquet (writes to data/processed/eurusd_m1.parquet)

# Path B — from raw CSV export
python scripts/01_preprocess.py   # CSV → Parquet (reads from data/raw/, writes to data/processed/)

# Common steps
python scripts/02_train_ml.py     # Train XGBoost models per timeframe (1min, 5min, 15min, 30min)
python scripts/04_optimize.py     # Search for best strategy params, writes results to .env
python scripts/03_backtest.py     # Run backtest using params from .env, outputs dashboard plot
```

All scripts are standalone entry points (`if __name__ == "__main__"`). Run from the project root so `core/` and `config/` imports resolve correctly.

## Configuration — `.env` and `settings.py`

Strategy parameters live in `.env` (project root, gitignored). `config/settings.py` loads `.env` via `python-dotenv` and falls back to hardcoded defaults if a key is missing.

**Do not edit strategy params in `settings.py` directly** — they will be overridden by `.env` at runtime.

| What to change | Where |
|---|---|
| Strategy params (`Z_THRESHOLD`, `ML_PROB_LIMIT`, `ATR_*`, `HOLD_BARS`, `SPREAD_COST`) | `.env` (or run `04_optimize.py`) |
| Data / model paths (`BASE_DATA_PATH`, `MODEL_SAVE_PATH`) | `.env` |
| XGBoost device (`XGB_DEVICE=cpu` or `cuda`) | `.env` |
| XGBoost hyperparams (`XGB_N_ESTIMATORS`, etc.) | `.env` or `settings.py` |

`.env.example` documents all supported keys with default values.

### Optimizer

```bash
python scripts/04_optimize.py                    # target PF ≥ 0.75, 300 trials
python scripts/04_optimize.py --target 1.2 --trials 500
```

Uses Optuna TPE (Bayesian) search. Prints only improvements and stops as soon as the target Profit Factor is reached. On completion, writes the best parameters directly to `.env` — no manual copy-paste needed.

## Architecture

- **`config/settings.py`** — Loads `.env`, exposes all config as module-level constants. Gitignored; `settings.py.example` is the committed template.
- **`.env`** — Strategy parameters written by `04_optimize.py`. Gitignored.
- **`core/kernels.py`** — Numba `@cuda.jit` kernel computing Z-Score and ATR in a single pass on GPU. Called with 1D grid/block layout: `[(n+255)//256, 256]`.
- **`core/metrics.py`** — Post-trade performance metrics (net profit, profit factor, max drawdown).
- **`scripts/02_train_ml.py`** — Resamples M1 data to multiple timeframes, computes GPU features, trains one XGBoost classifier per timeframe. Models saved to `models/<TF>/MREV_<TF>_v1.json`.
- **`scripts/03_backtest.py`** — CPU-side backtest using rolling Z-Score/ATR (not the GPU kernel), loads XGBoost model via `xgb.Booster`, applies triple-barrier exit (ATR-based TP/SL + time barrier), outputs dashboard plot.
- **`scripts/04_optimize.py`** — Optuna-based parameter optimizer. Loads data and model once, runs trials, writes best params to `.env`.

## Key Design Notes

- **GPU kernel vs CPU backtest**: `02_train_ml.py` uses the GPU kernel (`core/kernels.py`) for feature engineering. `03_backtest.py` uses `close.rolling().std()` as a proxy for ATR (not true high-low-close ATR) — these two ATR implementations are intentionally different.
- **Train/test split**: temporal split at `2025-09-01` (train on data before, test on data after). Adjust in `02_train_ml.py` if the dataset range changes.
- **Strategy features**: `z_score`, `atr`, `hour`, `day_of_week` — all four are used by the XGBoost classifier.
- **Triple-barrier exit**: TP at `ATR_TP_MULT × ATR`, SL at `ATR_SL_MULT × ATR`, time exit after `HOLD_BARS` bars.
- **All data paths are derived from `PROJECT_ROOT`** (`os.path.dirname` of the script's location). All scripts resolve paths relative to the project root — no hardcoded absolute paths.
- **`03_backtest.py` loads only the 1min model** — `MODEL_SAVE_PATH` defaults to `models/1MIN/MREV_1MIN_v1.json`. To backtest another timeframe, set `MODEL_SAVE_PATH` in `.env` (e.g. `models/5MIN/MREV_5MIN_v1.json`).
- **Dashboard output** is saved to `output/plots/sniper_full_dashboard.png` (created automatically).

## Protected Files (gitignored)

- `.env` — tuned strategy parameters
- `config/settings.py` — local config (loads `.env`)
- `models/` — trained model weights
- `data/` — raw and processed market data
- `live/logs/` — live trading logs
