"""
Sniper v3.0 — Enhanced Backtest (v1.0)

Improvements over 03_backtest.py:
  1. True ATR  — Wilder's ATR (H-L-C) instead of close.rolling().std()
  2. Session filter — trades only during London / NY sessions (UTC)
  3. Dynamic spread — spread cost varies by session liquidity
  4. Concurrent-trade guard — only one open position at a time (bar-by-bar loop)
  5. Multi-timeframe confirmation — 5MIN model must agree with 1MIN signal direction

New .env keys (all optional, have defaults):
  SESSION_FILTER=true        # enable session filter
  DYNAMIC_SPREAD=true        # session-aware spread vs fixed SPREAD_COST
  MTF_CONFIRM=true           # require 5MIN model to confirm 1MIN signal
  MTF_MODEL_PATH=models/5MIN/MREV_5MIN_v1.json
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas.plotting import register_matplotlib_converters

from core.metrics import calculate_metrics
from config.settings import (
    BASE_DATA_PATH, MODEL_SAVE_PATH, PROJECT_ROOT,
    Z_THRESHOLD, WINDOW_SIZE, ML_PROB_LIMIT,
    ATR_TP_MULT, ATR_SL_MULT, HOLD_BARS, SPREAD_COST,
)

register_matplotlib_converters()

# ── New feature flags (read from .env / environment) ──────────────────────────
def _bool(key, default): return os.getenv(key, str(default)).lower() in ('1', 'true', 'yes')

SESSION_FILTER   = _bool('SESSION_FILTER',  True)
DYNAMIC_SPREAD   = _bool('DYNAMIC_SPREAD',  True)
MTF_CONFIRM      = _bool('MTF_CONFIRM',     True)
MTF_MODEL_PATH   = os.path.join(
    PROJECT_ROOT,
    os.getenv('MTF_MODEL_PATH', 'models/5MIN/MREV_5MIN_v1.json')
)

# Session windows (UTC hours, inclusive start exclusive end)
SESSIONS = {
    'london_ny': (13, 16),   # tightest spread
    'london':    (7,  13),
    'ny':        (16, 21),
    # anything else = asian / off-hours
}

# Dynamic spread per session (in price points, ~pips × 0.0001)
SPREAD_BY_SESSION = {
    'london_ny': 0.00003,
    'london':    0.00005,
    'ny':        0.00007,
    'asian':     0.00015,
    'sunday':    0.00020,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def true_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Wilder's ATR: avg of max(H-L, |H-prev_C|, |L-prev_C|) over `window` bars."""
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def session_label(hour: int, day_of_week: int) -> str:
    if day_of_week == 6:                            # Sunday
        return 'sunday'
    if SESSIONS['london_ny'][0] <= hour < SESSIONS['london_ny'][1]:
        return 'london_ny'
    if SESSIONS['london'][0] <= hour < SESSIONS['london'][1]:
        return 'london'
    if SESSIONS['ny'][0] <= hour < SESSIONS['ny'][1]:
        return 'ny'
    return 'asian'


def is_tradeable_session(hour: int, day_of_week: int) -> bool:
    lbl = session_label(hour, day_of_week)
    return lbl in ('london', 'london_ny', 'ny')


def compute_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute Z-Score (close-based), true ATR, hour, day_of_week."""
    df = df.copy()
    roll_mean = df['close'].rolling(window).mean()
    roll_std  = df['close'].rolling(window).std()
    df['z_score']    = (df['close'] - roll_mean) / roll_std
    df['atr']        = true_atr(df, window)
    df['hour']       = df.index.hour
    df['day_of_week']= df.index.dayofweek
    return df.dropna()


def get_ml_prob(df: pd.DataFrame, model: xgb.Booster) -> pd.Series:
    features = ['z_score', 'atr', 'hour', 'day_of_week']
    return pd.Series(
        model.predict(xgb.DMatrix(df[features])),
        index=df.index
    )


def resample_and_predict(df_m1: pd.DataFrame, model: xgb.Booster,
                         tf: str, window: int) -> pd.Series:
    """
    Resample M1 → tf, compute features, get probabilities,
    then forward-fill back to M1 index.
    """
    df_tf = df_m1.resample(tf).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    df_tf = compute_features(df_tf, window)
    prob_tf = get_ml_prob(df_tf, model)
    # forward-fill higher-TF probability onto every M1 bar
    return prob_tf.reindex(df_m1.index, method='ffill')


# ── Bar-by-bar trade simulator ────────────────────────────────────────────────

def simulate_trades(df: pd.DataFrame,
                    signal_col: str,
                    atr_tp_mult: float,
                    atr_sl_mult: float,
                    hold_bars: int,
                    dynamic_spread: bool) -> pd.DataFrame:
    """
    Proper event-driven backtest loop.
    - Only one trade open at a time (concurrent-trade guard).
    - Each barrier checked every bar against live price.
    - Spread deducted at entry using session-aware cost.
    """
    records = []
    in_trade    = False
    entry_price = 0.0
    direction   = 0
    entry_atr   = 0.0
    entry_spread= 0.0
    entry_time  = None
    bars_held   = 0

    signals = df[signal_col].values
    closes  = df['close'].values
    atrs    = df['atr'].values
    hours   = df['hour'].values
    dows    = df['day_of_week'].values
    index   = df.index

    for i in range(len(df)):
        if in_trade:
            bars_held += 1
            price    = closes[i]
            raw_ret  = (price - entry_price) * direction
            tp_dist  = entry_atr * atr_tp_mult
            sl_dist  = entry_atr * atr_sl_mult

            hit_tp   = raw_ret >=  tp_dist
            hit_sl   = raw_ret <= -sl_dist
            hit_time = bars_held >= hold_bars

            if hit_tp or hit_sl or hit_time:
                raw_ret  = float(np.clip(raw_ret, -sl_dist, tp_dist))
                net_pnl  = raw_ret - entry_spread
                reason   = 'TP' if hit_tp else ('SL' if hit_sl else 'TIME')
                records.append({
                    'entry_time':   entry_time,
                    'exit_time':    index[i],
                    'direction':    'LONG' if direction == 1 else 'SHORT',
                    'entry_price':  entry_price,
                    'exit_price':   price,
                    'raw_ret':      raw_ret,
                    'net_pnl':      net_pnl,
                    'exit_reason':  reason,
                    'bars_held':    bars_held,
                    'session':      session_label(int(hours[i]), int(dows[i])),
                })
                in_trade = False

        # Only open a new trade if not already in one
        if not in_trade and signals[i] != 0:
            spread = (SPREAD_BY_SESSION[session_label(int(hours[i]), int(dows[i]))]
                      if dynamic_spread else SPREAD_COST)
            in_trade    = True
            entry_price = closes[i]
            direction   = int(signals[i])
            entry_atr   = atrs[i]
            entry_spread= spread
            entry_time  = index[i]
            bars_held   = 0

    return pd.DataFrame(records)


# ── Dashboard ─────────────────────────────────────────────────────────────────

def plot_dashboard(trades: pd.DataFrame, df: pd.DataFrame, config_summary: dict):
    if trades.empty:
        print("⚠️  No trades to plot.")
        return

    trades = trades.set_index('exit_time').sort_index()
    pnl_series = pd.Series(0.0, index=df.index)
    pnl_series.update(trades['net_pnl'])
    cum_pnl  = pnl_series.cumsum()
    peak     = cum_pnl.cummax()
    drawdown = cum_pnl - peak

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Sniper v3.0 Enhanced Backtest — v1.0', fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(cum_pnl.index, cum_pnl.values, color='#2ca02c', linewidth=1.5, label='Cumulative PnL')
    ax1.fill_between(cum_pnl.index, cum_pnl.values, 0,
                     where=(cum_pnl.values >= 0), alpha=0.15, color='#2ca02c')
    ax1.fill_between(cum_pnl.index, cum_pnl.values, 0,
                     where=(cum_pnl.values < 0),  alpha=0.15, color='#d62728')
    ax1.set_ylabel('Cumulative Profit (Points)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='#d62728', alpha=0.5, label='Drawdown')
    ax2.set_ylabel('Drawdown (Points)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 3. Exit reason breakdown
    ax3 = fig.add_subplot(gs[2, 0])
    reason_pnl = trades.groupby('exit_reason')['net_pnl'].sum()
    colors_map = {'TP': '#2ca02c', 'SL': '#d62728', 'TIME': '#f59e0b'}
    bars = ax3.bar(reason_pnl.index,
                   reason_pnl.values,
                   color=[colors_map.get(r, '#888') for r in reason_pnl.index])
    ax3.set_title('Net PnL by Exit Reason')
    ax3.set_ylabel('Points')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, reason_pnl.values):
        ax3.text(bar.get_x() + bar.get_width()/2, val,
                 f'{val:.4f}', ha='center',
                 va='bottom' if val >= 0 else 'top', fontsize=9)

    # 4. Trade count by session
    ax4 = fig.add_subplot(gs[2, 1])
    session_counts = trades.groupby('session')['net_pnl'].agg(['count', 'sum'])
    session_colors = {'london_ny': '#3b82f6', 'london': '#8b5cf6',
                      'ny': '#f59e0b', 'asian': '#6b7280', 'sunday': '#ef4444'}
    s_colors = [session_colors.get(s, '#888') for s in session_counts.index]
    ax4.bar(session_counts.index, session_counts['count'], color=s_colors, alpha=0.8)
    ax4.set_title('Trades by Session')
    ax4.set_ylabel('Trade Count')
    ax4.tick_params(axis='x', rotation=20)
    ax4.grid(True, alpha=0.3, axis='y')

    # Config summary text
    cfg_lines = '\n'.join(f'{k}: {v}' for k, v in config_summary.items())
    fig.text(0.01, 0.01, cfg_lines, fontsize=7.5,
             verticalalignment='bottom', family='monospace',
             color='grey')

    out_path = os.path.join(PROJECT_ROOT, 'output', 'plots', 'sniper_v1_dashboard.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Dashboard saved: {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("📂 Loading M1 data...")
    df_raw = pd.read_parquet(BASE_DATA_PATH)

    print(f"⚙️  Computing true ATR features (window={WINDOW_SIZE})...")
    df = compute_features(df_raw, WINDOW_SIZE)
    print(f"   Bars after dropna: {len(df):,}  ({df.index[0]} → {df.index[-1]})")

    # ── 1MIN model ────────────────────────────────────────────────────────────
    print(f"🧠 Loading 1MIN model: {MODEL_SAVE_PATH}")
    model_1m = xgb.Booster()
    model_1m.load_model(MODEL_SAVE_PATH)
    df['prob_1m'] = get_ml_prob(df, model_1m)

    # ── 5MIN model (MTF confirmation) ─────────────────────────────────────────
    if MTF_CONFIRM:
        if not os.path.exists(MTF_MODEL_PATH):
            print(f"⚠️  MTF model not found at {MTF_MODEL_PATH} — disabling MTF_CONFIRM.")
            df['prob_5m'] = 0.5
        else:
            print(f"🧠 Loading 5MIN model: {MTF_MODEL_PATH}")
            model_5m = xgb.Booster()
            model_5m.load_model(MTF_MODEL_PATH)
            df['prob_5m'] = resample_and_predict(df_raw, model_5m, '5min', WINDOW_SIZE)
            df['prob_5m'] = df['prob_5m'].fillna(0.5)
            print(f"   5MIN probabilities forward-filled onto M1 bars.")
    else:
        df['prob_5m'] = 1.0  # always passes the MTF filter when disabled

    # ── Session filter ────────────────────────────────────────────────────────
    if SESSION_FILTER:
        tradeable = df.apply(
            lambda r: is_tradeable_session(int(r['hour']), int(r['day_of_week'])), axis=1
        )
        n_filtered = (~tradeable).sum()
        df['tradeable'] = tradeable
        print(f"📅 Session filter: {n_filtered:,} bars excluded ({n_filtered/len(df)*100:.1f}%)")
    else:
        df['tradeable'] = True

    # ── Signal generation ─────────────────────────────────────────────────────
    # Both 1MIN and 5MIN must agree on direction (prob > threshold = short bias)
    short_cond = (
        (df['z_score'] >  Z_THRESHOLD) &
        (df['prob_1m'] >  ML_PROB_LIMIT) &
        (df['prob_5m'] >  ML_PROB_LIMIT) &
        df['tradeable']
    )
    long_cond = (
        (df['z_score'] < -Z_THRESHOLD) &
        (df['prob_1m'] >  ML_PROB_LIMIT) &
        (df['prob_5m'] >  ML_PROB_LIMIT) &
        df['tradeable']
    )
    df['signal'] = 0
    df.loc[short_cond, 'signal'] = -1
    df.loc[long_cond,  'signal'] =  1

    raw_signals = (df['signal'] != 0).sum()
    print(f"🎯 Raw signals before concurrent-trade guard: {raw_signals:,}")

    # ── Bar-by-bar simulation (concurrent-trade guard + true barriers) ─────────
    print("🔄 Running bar-by-bar simulation...")
    trades = simulate_trades(
        df,
        signal_col   = 'signal',
        atr_tp_mult  = ATR_TP_MULT,
        atr_sl_mult  = ATR_SL_MULT,
        hold_bars    = HOLD_BARS,
        dynamic_spread = DYNAMIC_SPREAD,
    )

    if trades.empty:
        print("⚠️  No trades generated. Try relaxing Z_THRESHOLD or ML_PROB_LIMIT.")
        return

    # ── Metrics ───────────────────────────────────────────────────────────────
    net, pf, mdd = calculate_metrics(trades['net_pnl'])
    win_rate = (trades['net_pnl'] > 0).mean() * 100

    reason_counts = trades['exit_reason'].value_counts()

    print(f"\n{'─'*44}")
    print(f"  SNIPER v3.0 ENHANCED BACKTEST — v1.0")
    print(f"{'─'*44}")
    print(f"  Total Trades   : {len(trades)}")
    print(f"  Win Rate       : {win_rate:.1f}%")
    print(f"  Net Profit     : {net:.6f} Points")
    print(f"  Profit Factor  : {pf:.3f}")
    print(f"  Max Drawdown   : {mdd:.6f} Points")
    print(f"  Exit — TP      : {reason_counts.get('TP', 0)}")
    print(f"  Exit — SL      : {reason_counts.get('SL', 0)}")
    print(f"  Exit — TIME    : {reason_counts.get('TIME', 0)}")
    print(f"{'─'*44}")
    print(f"  Enhancements active:")
    print(f"    True ATR (H-L-C)    : always on")
    print(f"    Session filter      : {SESSION_FILTER}")
    print(f"    Dynamic spread      : {DYNAMIC_SPREAD}")
    print(f"    MTF confirmation    : {MTF_CONFIRM}")
    print(f"    Concurrent guard    : always on")
    print(f"{'─'*44}\n")

    config_summary = {
        'Z_THRESHOLD': Z_THRESHOLD, 'ML_PROB_LIMIT': ML_PROB_LIMIT,
        'ATR_TP_MULT': ATR_TP_MULT, 'ATR_SL_MULT': ATR_SL_MULT,
        'HOLD_BARS': HOLD_BARS, 'WINDOW_SIZE': WINDOW_SIZE,
        'SESSION_FILTER': SESSION_FILTER, 'DYNAMIC_SPREAD': DYNAMIC_SPREAD,
        'MTF_CONFIRM': MTF_CONFIRM,
    }

    print("📈 Plotting enhanced dashboard...")
    plot_dashboard(trades, df, config_summary)


if __name__ == '__main__':
    run()
