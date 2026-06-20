"""
Parameter optimizer for Sniper v3.0.
Uses Optuna (Bayesian TPE) to search for parameters that achieve
the target Profit Factor. Data and model are loaded once and reused
across all trials for speed.

Usage:
    python scripts/04_optimize.py
    python scripts/04_optimize.py --target 1.0 --trials 200
"""
import os
import sys
import argparse
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_PROJECT_ROOT, '.env')

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

from core.metrics import calculate_metrics
from config.settings import BASE_DATA_PATH, MODEL_SAVE_PATH, WINDOW_SIZE

optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET_PF = 0.75    # overridden by --target
MAX_TRIALS = 300    # overridden by --trials
MIN_TRADES = 50     # discard parameter sets that produce too few trades


def load_assets() -> tuple[pd.DataFrame, xgb.Booster]:
    print(f"📂 Loading data from: {BASE_DATA_PATH}")
    df = pd.read_parquet(BASE_DATA_PATH)

    # Pre-compute fixed features (window size is not optimized to keep model valid)
    df['z_score'] = (df['close'] - df['close'].rolling(WINDOW_SIZE).mean()) / df['close'].rolling(WINDOW_SIZE).std()
    df['atr'] = df['close'].rolling(WINDOW_SIZE).std()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df = df.dropna()

    print(f"🧠 Loading model from: {MODEL_SAVE_PATH}")
    model = xgb.Booster()
    model.load_model(MODEL_SAVE_PATH)

    features = ['z_score', 'atr', 'hour', 'day_of_week']
    df['prob'] = model.predict(xgb.DMatrix(df[features]))

    print(f"✅ Ready — {len(df):,} bars, {df.index[0]} → {df.index[-1]}\n")
    return df, model


def backtest(df: pd.DataFrame,
             z_threshold: float,
             ml_prob_limit: float,
             atr_tp_mult: float,
             atr_sl_mult: float,
             hold_bars: int,
             spread_cost: float) -> tuple[float, float, float, int]:
    temp = df.copy()
    temp['signal'] = 0
    temp.loc[(temp['z_score'] > z_threshold)  & (temp['prob'] > ml_prob_limit), 'signal'] = -1
    temp.loc[(temp['z_score'] < -z_threshold) & (temp['prob'] > ml_prob_limit), 'signal'] = 1

    temp['raw_ret'] = (temp['close'].shift(-hold_bars) - temp['close']) * temp['signal']
    tp_dist = temp['atr'] * atr_tp_mult
    sl_dist = temp['atr'] * atr_sl_mult
    temp['raw_ret'] = np.clip(temp['raw_ret'], -sl_dist, tp_dist)
    temp['net_pnl'] = np.where(temp['signal'] != 0, temp['raw_ret'] - spread_cost, 0)

    trades = temp[temp['signal'] != 0]['net_pnl']
    if len(trades) < MIN_TRADES:
        return 0.0, 0.0, 0.0, len(trades)

    net, pf, mdd = calculate_metrics(trades)
    return net, pf, mdd, len(trades)


def make_objective(df: pd.DataFrame, target_pf: float):
    best = {'pf': 0.0}

    def objective(trial: optuna.Trial) -> float:
        z_threshold   = trial.suggest_float('z_threshold',   0.5,  3.0, step=0.05)
        ml_prob_limit = trial.suggest_float('ml_prob_limit', 0.50, 0.80, step=0.01)
        atr_tp_mult   = trial.suggest_float('atr_tp_mult',   0.5,  3.0, step=0.1)
        atr_sl_mult   = trial.suggest_float('atr_sl_mult',   0.5,  3.0, step=0.1)
        hold_bars     = trial.suggest_int  ('hold_bars',      2,   30)
        spread_cost   = trial.suggest_float('spread_cost',   0.00005, 0.00020, step=0.00005)

        net, pf, mdd, n_trades = backtest(df, z_threshold, ml_prob_limit,
                                          atr_tp_mult, atr_sl_mult,
                                          hold_bars, spread_cost)

        if pf > best['pf']:
            best['pf'] = pf
            print(f"  Trial {trial.number:>4} | PF={pf:.3f} | Net={net:.6f} | MDD={mdd:.6f} | "
                  f"Trades={n_trades} | "
                  f"Z={z_threshold} Prob={ml_prob_limit} TP={atr_tp_mult} SL={atr_sl_mult} "
                  f"Hold={hold_bars} Spread={spread_cost:.5f}")

        return pf

    return objective


class ReachTargetCallback:
    def __init__(self, target_pf: float):
        self.target_pf = target_pf

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if study.best_value >= self.target_pf:
            study.stop()


def _write_env(params: dict):
    """Write optimized strategy params to .env, preserving any existing non-strategy keys."""
    strategy_keys = {k.upper() for k in params}

    # Read existing .env lines that are NOT strategy params (e.g. paths, device)
    preserved = []
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, encoding='utf-8') as f:
            for line in f:
                key = line.split('=')[0].strip()
                if key and not key.startswith('#') and key not in strategy_keys:
                    preserved.append(line.rstrip())

    lines = ['# Sniper v3.0 - Strategy Parameters',
             '# Auto-generated by scripts/04_optimize.py - do not edit manually.',
             '']
    if preserved:
        lines += preserved + ['']

    for k, v in params.items():
        val = round(v, 6) if isinstance(v, float) else v
        lines.append(f"{k.upper()}={val}")

    with open(_ENV_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\n📝 Best params written to .env")
    print(f"   Run python scripts/03_backtest.py — no settings.py changes needed.\n")
    for k, v in params.items():
        val = round(v, 6) if isinstance(v, float) else v
        print(f"   {k.upper()}={val}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=float, default=TARGET_PF,
                        help='Minimum Profit Factor to stop at (default: 0.75)')
    parser.add_argument('--trials', type=int, default=MAX_TRIALS,
                        help='Maximum number of trials (default: 300)')
    args = parser.parse_args()

    df, _ = load_assets()

    print(f"🔍 Searching for Profit Factor ≥ {args.target} (max {args.trials} trials)")
    print(f"   Showing only improvements...\n")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='sniper_v3_optimization'
    )

    objective = make_objective(df, args.target)
    callback = ReachTargetCallback(args.target)

    study.optimize(objective, n_trials=args.trials, callbacks=[callback],
                   show_progress_bar=False)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"🏆 BEST RESULT — Profit Factor: {best.value:.3f}")
    print(f"{'='*60}")

    _write_env(best.params)

    if best.value >= args.target:
        print(f"\n✅ Target PF ≥ {args.target} reached!")
    else:
        print(f"\n⚠️  Target not reached after {args.trials} trials. Best PF: {best.value:.3f}")
        print("    Try: more data (run 00_ticks_to_m1.py), retrain, or lower --target.")


if __name__ == '__main__':
    main()
