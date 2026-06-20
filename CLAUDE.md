# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sniper v3.0 is a GPU-accelerated FX mean reversion trading system for EURUSD M1 data. It uses Numba CUDA kernels for feature computation and XGBoost for probabilistic trade filtering. The system prioritizes capital preservation (max drawdown) over raw profit.

## Setup

```bash
pip install -r requirements.txt
cp config/settings.py.example config/settings.py
cp .env.example .env  # then tune or run 04_optimize_v1.0.py to auto-fill
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
python scripts/02_train_ml.py         # Train XGBoost models per timeframe (1min, 5min, 15min, 30min)
python scripts/04_optimize_v1.0.py    # Search for best strategy params, writes results to .env
python scripts/03_backtest_v1.0.py    # Run enhanced backtest using params from .env
```

There are also baseline (v0) versions of the last two steps:

```bash
python scripts/04_optimize.py         # Baseline optimizer (vectorized, single TF)
python scripts/03_backtest.py         # Baseline backtest (vectorized, single TF)
```

All scripts are standalone entry points (`if __name__ == "__main__"`). Run from the project root so `core/` and `config/` imports resolve correctly.

## Configuration — `.env` and `settings.py`

Strategy parameters live in `.env` (project root, gitignored). `config/settings.py` loads `.env` via `python-dotenv` and falls back to hardcoded defaults if a key is missing.

**Do not edit strategy params in `settings.py` directly** — they will be overridden by `.env` at runtime.

| What to change | Where |
|---|---|
| Strategy params (`Z_THRESHOLD`, `ML_PROB_LIMIT`, `ML_PROB_LIMIT_5M`, `ATR_*`, `HOLD_BARS`, `SPREAD_COST`) | `.env` (or run optimizer) |
| Breakeven stop (`BREAKEVEN_MULT`) | `.env` (0 = disabled; N = move SL to entry once profit ≥ N×ATR) |
| Enhanced flags (`SESSION_FILTER`, `DYNAMIC_SPREAD`, `MTF_CONFIRM`, `MTF_MODEL_PATH`) | `.env` |
| Data / model paths (`BASE_DATA_PATH`, `MODEL_SAVE_PATH`) | `.env` |
| XGBoost device (`XGB_DEVICE=cpu` or `cuda`) | `.env` |
| XGBoost hyperparams (`XGB_N_ESTIMATORS`, etc.) | `.env` or `settings.py` |

`.env.example` documents all supported keys with default values.

### Optimizers

```bash
# Baseline — vectorized, 1MIN model only
python scripts/04_optimize.py --target 1.2 --trials 500

# Enhanced — matches 03_backtest_v1.0.py logic exactly
python scripts/04_optimize_v1.0.py --target 1.2 --trials 500
```

Both use Optuna TPE (Bayesian) search, print only improvements, stop when the target score is reached, and write the best parameters directly to `.env`. The enhanced optimizer accepts `--metric {pf,sharpe}` to optimize Profit Factor or Sharpe Ratio.

## Architecture

- **`config/settings.py`** — Loads `.env`, exposes all config as module-level constants. Gitignored; `settings.py.example` is the committed template.
- **`.env`** — Strategy parameters written by the optimizer. Gitignored.
- **`core/kernels.py`** — Numba `@cuda.jit` kernel computing Z-Score and ATR in a single pass on GPU. Called with 1D grid/block layout: `[(n+255)//256, 256]`.
- **`core/metrics.py`** — `calculate_metrics()` (backwards-compat) + `calculate_report()` returning Sharpe ratio, Calmar ratio, Recovery Factor, expectancy, avg win/loss.
- **`scripts/02_train_ml.py`** — Resamples M1 data to multiple timeframes, computes GPU features, trains one XGBoost classifier per timeframe. Models saved to `models/<TF>/MREV_<TF>_v1.json`.
- **`scripts/03_backtest.py`** — Baseline backtest: vectorized simulation, rolling std as ATR proxy, 1MIN model only, fixed spread.
- **`scripts/03_backtest_v1.0.py`** — Enhanced backtest: true Wilder ATR, session filter, dynamic spread, concurrent-trade guard, MTF confirmation, correct directional ML filter, breakeven stop, per-session stats, Monte Carlo (3-panel).
- **`scripts/04_optimize.py`** — Baseline Optuna optimizer. Searches `Z_THRESHOLD`, `ML_PROB_LIMIT`, `ATR_*`, `HOLD_BARS`, `SPREAD_COST`.
- **`scripts/04_optimize_v1.0.py`** — Enhanced Optuna optimizer. Mirrors `03_backtest_v1.0.py` exactly; searches `ML_PROB_LIMIT_5M`, `BREAKEVEN_MULT`; supports `--metric {pf,sharpe}`.

## Key Design Notes

- **Baseline vs enhanced ATR**: `03_backtest.py` uses `close.rolling().std()` as an ATR proxy. `03_backtest_v1.0.py` uses true Wilder ATR (`max(H-L, |H-prev_C|, |L-prev_C|)`). Always pair the correct optimizer with its backtest.
- **Directional ML filter**: The XGBoost model predicts P(price goes DOWN). For SHORT signals `prob > ML_PROB_LIMIT` is correct. For LONG signals, `1-prob > ML_PROB_LIMIT` (i.e. `prob < 1-ML_PROB_LIMIT`) is used so both directions are filtered consistently. The baseline `03_backtest.py` does not implement this correction.
- **Breakeven stop**: When `BREAKEVEN_MULT > 0`, the SL is moved to entry (breakeven) once unrealized profit ≥ `BREAKEVEN_MULT × ATR`. Exits under this condition are labeled `BE`. Optimizer searches this parameter automatically.
- **`ML_PROB_LIMIT_5M`**: Separate threshold for the 5MIN MTF model, written by the optimizer and read by the backtest. Defaults to `ML_PROB_LIMIT` if not set.
- **GPU kernel vs CPU backtest**: `02_train_ml.py` uses the GPU kernel (`core/kernels.py`) for feature engineering. Both backtests compute ATR on CPU at runtime.
- **Train/test split**: temporal split at `2025-09-01` (train on data before, test on data after). Adjust in `02_train_ml.py` if the dataset range changes.
- **Strategy features**: `z_score`, `atr`, `hour`, `day_of_week` — all four are used by the XGBoost classifier.
- **Triple-barrier exit**: TP at `ATR_TP_MULT × ATR`, SL at `ATR_SL_MULT × ATR`, time exit after `HOLD_BARS` bars. Optional breakeven stop (`BREAKEVEN_MULT`).
- **All data paths are derived from `PROJECT_ROOT`** — no hardcoded absolute paths.
- **Model paths**: `MODEL_SAVE_PATH` in `.env` defaults to `models/1MIN/MREV_1MIN_v1.json`. `MTF_MODEL_PATH` defaults to `models/5MIN/MREV_5MIN_v1.json`.
- **Enhanced flags default to `true`**: `SESSION_FILTER`, `DYNAMIC_SPREAD`, and `MTF_CONFIRM` are all enabled by default in the v1.0 scripts. Set them to `false` in `.env` to disable.
- **Dashboard outputs**: `03_backtest.py` → `output/plots/sniper_full_dashboard.png`. `03_backtest_v1.0.py` → `output/plots/sniper_v1_dashboard.png` + `output/plots/monte_carlo_paths_v1.0.png` (3-panel: equity fan, final PnL histogram, MDD distribution).

## Protected Files (gitignored)

- `.env` — tuned strategy parameters
- `config/settings.py` — local config (loads `.env`)
- `models/` — trained model weights
- `data/` — raw and processed market data
- `live/logs/` — live trading logs
