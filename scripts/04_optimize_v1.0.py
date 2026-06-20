"""
Sniper v3.0 — Enhanced Parameter Optimizer (v1.0)

Matches the enhanced backtest (03_backtest_v1.0.py) exactly:
  - True Wilder ATR (H-L-C) for TP/SL calibration
  - Session filter (London / NY only)
  - Dynamic spread by session liquidity
  - Bar-by-bar concurrent-trade guard
  - Multi-timeframe confirmation (5MIN model)

New vs 04_optimize.py:
  - ML_PROB_LIMIT_5M searched separately from ML_PROB_LIMIT (1MIN)
  - SPREAD_COST not searched when DYNAMIC_SPREAD=true (session handles it)
  - All structural flags (SESSION_FILTER, DYNAMIC_SPREAD, MTF_CONFIRM)
    read from .env — toggle them there, not here

Usage:
    python scripts/04_optimize_v1.0.py
    python scripts/04_optimize_v1.0.py --target 1.2 --trials 500
"""
import os
import sys
import argparse
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

from core.metrics import calculate_metrics
from config.settings import BASE_DATA_PATH, MODEL_SAVE_PATH, WINDOW_SIZE, PROJECT_ROOT, SPREAD_COST

optuna.logging.set_verbosity(optuna.logging.WARNING)

_ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

TARGET_PF  = 0.75
MAX_TRIALS = 300
MIN_TRADES = 50

# ── Feature flags (from .env) ─────────────────────────────────────────────────
def _bool(key, default): return os.getenv(key, str(default)).lower() in ('1', 'true', 'yes')

SESSION_FILTER = _bool('SESSION_FILTER', True)
DYNAMIC_SPREAD = _bool('DYNAMIC_SPREAD', True)
MTF_CONFIRM    = _bool('MTF_CONFIRM',    True)
MTF_MODEL_PATH = os.path.join(PROJECT_ROOT,
    os.getenv('MTF_MODEL_PATH', 'models/5MIN/MREV_5MIN_v1.json'))

# Session definitions (UTC)
SESSIONS = {'london_ny': (13, 16), 'london': (7, 13), 'ny': (16, 21)}
SPREAD_BY_SESSION = {
    'london_ny': 0.00003, 'london': 0.00005,
    'ny': 0.00007, 'asian': 0.00015, 'sunday': 0.00020,
}


# ── Helpers (mirrors 03_backtest_v1.0.py exactly) ────────────────────────────

def true_atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def session_label(hour: int, dow: int) -> str:
    if dow == 6: return 'sunday'
    if SESSIONS['london_ny'][0] <= hour < SESSIONS['london_ny'][1]: return 'london_ny'
    if SESSIONS['london'][0]    <= hour < SESSIONS['london'][1]:    return 'london'
    if SESSIONS['ny'][0]        <= hour < SESSIONS['ny'][1]:        return 'ny'
    return 'asian'


def is_tradeable(hour: int, dow: int) -> bool:
    return session_label(hour, dow) in ('london', 'london_ny', 'ny')


def compute_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    roll_mean = df['close'].rolling(window).mean()
    roll_std  = df['close'].rolling(window).std()
    df['z_score']     = (df['close'] - roll_mean) / roll_std
    df['atr']         = true_atr(df, window)
    df['hour']        = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    return df.dropna()


def get_ml_prob(df: pd.DataFrame, model: xgb.Booster) -> pd.Series:
    features = ['z_score', 'atr', 'hour', 'day_of_week']
    return pd.Series(model.predict(xgb.DMatrix(df[features])), index=df.index)


def resample_and_predict(df_m1: pd.DataFrame, model: xgb.Booster,
                         tf: str, window: int) -> pd.Series:
    df_tf = df_m1.resample(tf).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    df_tf = compute_features(df_tf, window)
    prob_tf = get_ml_prob(df_tf, model)
    return prob_tf.reindex(df_m1.index, method='ffill')


def simulate_trades(df: pd.DataFrame,
                    z_threshold: float,
                    ml_prob_limit_1m: float,
                    ml_prob_limit_5m: float,
                    atr_tp_mult: float,
                    atr_sl_mult: float,
                    hold_bars: int,
                    dynamic_spread: bool,
                    fixed_spread: float) -> tuple[float, float, float, int]:
    """
    Bar-by-bar simulator with concurrent-trade guard.
    Pre-resolved numpy arrays for speed across hundreds of Optuna trials.
    """
    # Build signal mask from pre-computed columns
    short_mask = (
        (df['z_score'].values  >  z_threshold) &
        (df['prob_1m'].values  >  ml_prob_limit_1m) &
        (df['prob_5m'].values  >  ml_prob_limit_5m) &
        df['tradeable'].values
    )
    long_mask = (
        (df['z_score'].values  < -z_threshold) &
        (df['prob_1m'].values  >  ml_prob_limit_1m) &
        (df['prob_5m'].values  >  ml_prob_limit_5m) &
        df['tradeable'].values
    )
    signals  = np.where(short_mask, -1, np.where(long_mask, 1, 0))
    closes   = df['close'].values
    atrs     = df['atr'].values
    spreads  = df['spread_cost'].values
    hours    = df['hour'].values
    dows     = df['day_of_week'].values

    pnls       = []
    in_trade   = False
    entry_price= 0.0
    direction  = 0
    entry_atr  = 0.0
    entry_sp   = 0.0
    bars_held  = 0

    for i in range(len(df)):
        if in_trade:
            bars_held += 1
            raw_ret    = (closes[i] - entry_price) * direction
            tp_dist    = entry_atr * atr_tp_mult
            sl_dist    = entry_atr * atr_sl_mult

            if raw_ret >= tp_dist or raw_ret <= -sl_dist or bars_held >= hold_bars:
                raw_ret = float(np.clip(raw_ret, -sl_dist, tp_dist))
                pnls.append(raw_ret - entry_sp)
                in_trade = False

        if not in_trade and signals[i] != 0:
            sp = (SPREAD_BY_SESSION[session_label(int(hours[i]), int(dows[i]))]
                  if dynamic_spread else fixed_spread)
            in_trade    = True
            entry_price = closes[i]
            direction   = int(signals[i])
            entry_atr   = atrs[i]
            entry_sp    = sp
            bars_held   = 0

    if len(pnls) < MIN_TRADES:
        return 0.0, 0.0, 0.0, len(pnls)

    pnl_series = pd.Series(pnls)
    net, pf, mdd = calculate_metrics(pnl_series)
    return net, pf, mdd, len(pnls)


# ── Asset loading (once, shared across all trials) ────────────────────────────

def load_assets():
    print(f"📂 Loading data from: {BASE_DATA_PATH}")
    df_raw = pd.read_parquet(BASE_DATA_PATH)

    print(f"⚙️  Computing true ATR features (window={WINDOW_SIZE})...")
    df = compute_features(df_raw, WINDOW_SIZE)

    print(f"🧠 Loading 1MIN model: {MODEL_SAVE_PATH}")
    model_1m = xgb.Booster()
    model_1m.load_model(MODEL_SAVE_PATH)
    df['prob_1m'] = get_ml_prob(df, model_1m)

    if MTF_CONFIRM and os.path.exists(MTF_MODEL_PATH):
        print(f"🧠 Loading 5MIN model: {MTF_MODEL_PATH}")
        model_5m = xgb.Booster()
        model_5m.load_model(MTF_MODEL_PATH)
        df['prob_5m'] = resample_and_predict(df_raw, model_5m, '5min', WINDOW_SIZE)
        df['prob_5m'] = df['prob_5m'].fillna(0.5)
        print(f"   5MIN probs forward-filled onto M1 bars.")
    else:
        df['prob_5m'] = 1.0  # always passes when MTF disabled

    if SESSION_FILTER:
        df['tradeable'] = df.apply(
            lambda r: is_tradeable(int(r['hour']), int(r['day_of_week'])), axis=1)
        pct = (~df['tradeable']).mean() * 100
        print(f"📅 Session filter: {pct:.1f}% of bars excluded")
    else:
        df['tradeable'] = True

    # Pre-compute per-bar spread for the dynamic case (avoids per-trial recompute)
    df['spread_cost'] = df.apply(
        lambda r: SPREAD_BY_SESSION[session_label(int(r['hour']), int(r['day_of_week']))]
        if DYNAMIC_SPREAD else SPREAD_COST, axis=1)

    print(f"✅ Ready — {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")
    print(f"   Flags: SESSION_FILTER={SESSION_FILTER}  "
          f"DYNAMIC_SPREAD={DYNAMIC_SPREAD}  MTF_CONFIRM={MTF_CONFIRM}\n")
    return df


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(df: pd.DataFrame, target_pf: float):
    best = {'pf': 0.0}

    def objective(trial: optuna.Trial) -> float:
        z_threshold      = trial.suggest_float('z_threshold',      0.5,  3.0,  step=0.05)
        ml_prob_limit    = trial.suggest_float('ml_prob_limit',    0.50, 0.80, step=0.01)
        ml_prob_limit_5m = (trial.suggest_float('ml_prob_limit_5m', 0.50, 0.80, step=0.01)
                            if MTF_CONFIRM else 0.0)
        atr_tp_mult      = trial.suggest_float('atr_tp_mult',      0.5,  4.0,  step=0.1)
        atr_sl_mult      = trial.suggest_float('atr_sl_mult',      0.5,  3.0,  step=0.1)
        hold_bars        = trial.suggest_int  ('hold_bars',         2,    30)

        net, pf, mdd, n_trades = simulate_trades(
            df,
            z_threshold      = z_threshold,
            ml_prob_limit_1m = ml_prob_limit,
            ml_prob_limit_5m = ml_prob_limit_5m,
            atr_tp_mult      = atr_tp_mult,
            atr_sl_mult      = atr_sl_mult,
            hold_bars        = hold_bars,
            dynamic_spread   = DYNAMIC_SPREAD,
            fixed_spread     = SPREAD_COST,
        )

        if pf > best['pf']:
            best['pf'] = pf
            mtf_str = f" Prob5m={ml_prob_limit_5m}" if MTF_CONFIRM else ""
            print(f"  Trial {trial.number:>4} | PF={pf:.3f} | Net={net:.6f} | MDD={mdd:.6f} | "
                  f"Trades={n_trades} | Z={z_threshold} Prob={ml_prob_limit}{mtf_str} "
                  f"TP={atr_tp_mult} SL={atr_sl_mult} Hold={hold_bars}")

        return pf

    return objective


class ReachTargetCallback:
    def __init__(self, target_pf: float):
        self.target_pf = target_pf

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if study.best_value >= self.target_pf:
            study.stop()


# ── .env writer ───────────────────────────────────────────────────────────────

def _write_env(params: dict):
    strategy_keys = {k.upper() for k in params}

    preserved = []
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, encoding='utf-8') as f:
            for line in f:
                key = line.split('=')[0].strip()
                if key and not key.startswith('#') and key not in strategy_keys:
                    preserved.append(line.rstrip())

    lines = ['# Sniper v3.0 - Strategy Parameters',
             '# Auto-generated by scripts/04_optimize_v1.0.py - do not edit manually.',
             '']
    if preserved:
        lines += preserved + ['']

    for k, v in params.items():
        val = round(v, 6) if isinstance(v, float) else v
        lines.append(f"{k.upper()}={val}")

    with open(_ENV_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\n📝 Best params written to .env")
    print(f"   Run python scripts/03_backtest_v1.0.py — no manual edits needed.\n")
    for k, v in params.items():
        val = round(v, 6) if isinstance(v, float) else v
        print(f"   {k.upper()}={val}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Optuna optimizer matching 03_backtest_v1.0.py logic.')
    parser.add_argument('--target', type=float, default=TARGET_PF,
                        help=f'Target Profit Factor (default: {TARGET_PF})')
    parser.add_argument('--trials', type=int, default=MAX_TRIALS,
                        help=f'Max Optuna trials (default: {MAX_TRIALS})')
    args = parser.parse_args()

    df = load_assets()

    print(f"🔍 Searching for Profit Factor ≥ {args.target} (max {args.trials} trials)")
    print(f"   Showing only improvements...\n")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='sniper_v3_v1_optimization',
    )

    objective = make_objective(df, args.target)
    callback  = ReachTargetCallback(args.target)

    study.optimize(objective, n_trials=args.trials, callbacks=[callback],
                   show_progress_bar=False)

    best = study.best_trial
    print(f"\n{'='*62}")
    print(f"🏆 BEST RESULT — Profit Factor: {best.value:.3f}")
    print(f"{'='*62}")

    _write_env(best.params)

    if best.value >= args.target:
        print(f"\n✅ Target PF ≥ {args.target} reached!")
    else:
        print(f"\n⚠️  Target not reached after {args.trials} trials. "
              f"Best PF: {best.value:.3f}")
        print("    Try: more data, retrain models, or lower --target.")


if __name__ == '__main__':
    main()
