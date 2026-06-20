"""
Sniper v3.0 — Enhanced Backtest (v1.0)

Improvements over 03_backtest.py:
  1. True ATR  — Wilder's ATR (H-L-C) instead of close.rolling().std()
  2. Session filter — trades only during London / NY sessions (UTC)
  3. Dynamic spread — spread cost varies by session liquidity
  4. Concurrent-trade guard — only one open position at a time (bar-by-bar loop)
  5. Multi-timeframe confirmation — 5MIN model must agree with 1MIN signal direction
  6. Correct directional ML filter — LONGs use 1-prob (P(up)) not raw prob (P(down))
  7. Breakeven stop — move SL to entry once profit >= BREAKEVEN_MULT × ATR
  8. Extended metrics — Sharpe ratio, Calmar, Recovery Factor, expectancy

.env keys (all optional):
  SESSION_FILTER=true
  DYNAMIC_SPREAD=true
  MTF_CONFIRM=true
  MTF_MODEL_PATH=models/5MIN/MREV_5MIN_v1.json
  ML_PROB_LIMIT_5M=0.50
  BREAKEVEN_MULT=0.0
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

from core.metrics import calculate_metrics, calculate_report
from config.settings import (
    BASE_DATA_PATH, MODEL_SAVE_PATH, PROJECT_ROOT,
    Z_THRESHOLD, WINDOW_SIZE, ML_PROB_LIMIT, ML_PROB_LIMIT_5M,
    ATR_TP_MULT, ATR_SL_MULT, HOLD_BARS, SPREAD_COST, BREAKEVEN_MULT,
)

register_matplotlib_converters()

# ── Feature flags (from .env) ─────────────────────────────────────────────────
def _bool(key, default): return os.getenv(key, str(default)).lower() in ('1', 'true', 'yes')

SESSION_FILTER = _bool('SESSION_FILTER', True)
DYNAMIC_SPREAD = _bool('DYNAMIC_SPREAD', True)
MTF_CONFIRM    = _bool('MTF_CONFIRM',    True)
MTF_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    os.getenv('MTF_MODEL_PATH', 'models/5MIN/MREV_5MIN_v1.json')
)

# Session windows (UTC hours)
SESSIONS = {
    'london_ny': (13, 16),
    'london':    (7,  13),
    'ny':        (16, 21),
}

# Spread per session in price points (~pip × 0.0001)
SPREAD_BY_SESSION = {
    'london_ny': 0.00003,
    'london':    0.00005,
    'ny':        0.00007,
    'asian':     0.00015,
    'sunday':    0.00020,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def true_atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def session_label(hour: int, day_of_week: int) -> str:
    if day_of_week == 6:
        return 'sunday'
    if SESSIONS['london_ny'][0] <= hour < SESSIONS['london_ny'][1]:
        return 'london_ny'
    if SESSIONS['london'][0] <= hour < SESSIONS['london'][1]:
        return 'london'
    if SESSIONS['ny'][0] <= hour < SESSIONS['ny'][1]:
        return 'ny'
    return 'asian'


def is_tradeable_session(hour: int, day_of_week: int) -> bool:
    return session_label(hour, day_of_week) in ('london', 'london_ny', 'ny')


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
    return pd.Series(
        model.predict(xgb.DMatrix(df[features])),
        index=df.index
    )


def resample_and_predict(df_m1: pd.DataFrame, model: xgb.Booster,
                         tf: str, window: int) -> pd.Series:
    df_tf = df_m1.resample(tf).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    df_tf = compute_features(df_tf, window)
    prob_tf = get_ml_prob(df_tf, model)
    return prob_tf.reindex(df_m1.index, method='ffill')


# ── Bar-by-bar trade simulator ────────────────────────────────────────────────

def simulate_trades(df: pd.DataFrame,
                    signal_col: str,
                    atr_tp_mult: float,
                    atr_sl_mult: float,
                    hold_bars: int,
                    dynamic_spread: bool,
                    breakeven_mult: float = 0.0) -> pd.DataFrame:
    """
    Event-driven backtest with:
    - Concurrent-trade guard (one position at a time)
    - True barrier check every bar against live price
    - Directional session-aware spread at entry
    - Optional breakeven stop: once profit >= breakeven_mult × ATR,
      SL moves to entry (0 risk from that point)
    """
    records = []
    in_trade     = False
    entry_price  = 0.0
    direction    = 0
    entry_atr    = 0.0
    entry_spread = 0.0
    entry_time   = None
    bars_held    = 0
    peak_ret     = 0.0   # highest unrealized PnL seen in current trade
    be_activated = False  # breakeven stop flag

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

            # Track best unrealized profit for breakeven logic
            if raw_ret > peak_ret:
                peak_ret = raw_ret

            # Activate breakeven stop once profit exceeds trigger
            if breakeven_mult > 0 and peak_ret >= entry_atr * breakeven_mult:
                be_activated = True

            # Effective SL distance: 0 when breakeven is active
            effective_sl = 0.0 if be_activated else sl_dist

            hit_tp   = raw_ret >=  tp_dist
            hit_sl   = raw_ret <= -effective_sl
            hit_time = bars_held >= hold_bars

            if hit_tp or hit_sl or hit_time:
                raw_ret  = float(np.clip(raw_ret, -sl_dist, tp_dist))
                net_pnl  = raw_ret - entry_spread
                reason   = 'TP' if hit_tp else ('SL' if hit_sl else 'TIME')
                if be_activated and hit_sl:
                    reason = 'BE'   # breakeven stop
                records.append({
                    'entry_time':  entry_time,
                    'exit_time':   index[i],
                    'direction':   'LONG' if direction == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price':  price,
                    'raw_ret':     raw_ret,
                    'net_pnl':     net_pnl,
                    'exit_reason': reason,
                    'bars_held':   bars_held,
                    'session':     session_label(int(hours[i]), int(dows[i])),
                })
                in_trade     = False
                peak_ret     = 0.0
                be_activated = False

        if not in_trade and signals[i] != 0:
            sp = (SPREAD_BY_SESSION[session_label(int(hours[i]), int(dows[i]))]
                  if dynamic_spread else SPREAD_COST)
            in_trade     = True
            entry_price  = closes[i]
            direction    = int(signals[i])
            entry_atr    = atrs[i]
            entry_spread = sp
            entry_time   = index[i]
            bars_held    = 0
            peak_ret     = 0.0
            be_activated = False

    return pd.DataFrame(records)


# ── Dashboard ─────────────────────────────────────────────────────────────────

def plot_dashboard(trades: pd.DataFrame, df: pd.DataFrame, report: dict):
    if trades.empty:
        print("No trades to plot.")
        return

    t = trades.set_index('exit_time').sort_index()
    pnl_series = pd.Series(0.0, index=df.index)
    pnl_series.update(t['net_pnl'])
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

    # 3. Exit reason net PnL
    ax3 = fig.add_subplot(gs[2, 0])
    reason_pnl = trades.groupby('exit_reason')['net_pnl'].sum()
    colors_map = {'TP': '#2ca02c', 'SL': '#d62728', 'TIME': '#f59e0b', 'BE': '#3b82f6'}
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

    # Key metrics as footer
    cfg_lines = (
        f"PF={report['profit_factor']:.3f}  Sharpe={report['sharpe_ratio']:.2f}  "
        f"Calmar={report['calmar_ratio']:.2f}  RecFactor={report['recovery_factor']:.2f}  "
        f"Expectancy={report['expectancy']:.5f}  "
        f"Z={Z_THRESHOLD}  Prob1m={ML_PROB_LIMIT}  Prob5m={ML_PROB_LIMIT_5M}  "
        f"TP={ATR_TP_MULT}×ATR  SL={ATR_SL_MULT}×ATR  Hold={HOLD_BARS}  BE={BREAKEVEN_MULT}"
    )
    fig.text(0.01, 0.005, cfg_lines, fontsize=7.5,
             verticalalignment='bottom', family='monospace', color='grey')

    out_path = os.path.join(PROJECT_ROOT, 'output', 'plots', 'sniper_v1_dashboard.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Dashboard saved: {out_path}")
    plt.close()


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def plot_monte_carlo(trades: pd.DataFrame, n_paths: int = 5000, seed: int = 42):
    pnl      = trades['net_pnl'].values
    n_trades = len(pnl)
    rng      = np.random.default_rng(seed)

    paths = np.empty((n_paths, n_trades))
    for i in range(n_paths):
        paths[i] = rng.permutation(pnl).cumsum()

    final_pnls = paths[:, -1]
    pct5       = np.percentile(paths, 5,  axis=0)
    pct95      = np.percentile(paths, 95, axis=0)
    pct50      = np.percentile(paths, 50, axis=0)
    actual_cum = pnl.cumsum()
    x          = np.arange(1, n_trades + 1)

    # Max drawdown distribution across paths
    dd_per_path = np.array([
        (paths[i] - np.maximum.accumulate(paths[i])).min()
        for i in range(n_paths)
    ])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Sniper v3.0 v1.0 — Monte Carlo Robustness ({n_paths:,} Paths)',
                 fontsize=14, fontweight='bold')

    # Left: equity path fan
    ax = axes[0]
    sample_idx = rng.choice(n_paths, size=min(500, n_paths), replace=False)
    for idx in sample_idx:
        ax.plot(x, paths[idx], color='#3b82f6', alpha=0.04, linewidth=0.6)
    ax.fill_between(x, pct5, pct95, alpha=0.18, color='#3b82f6', label='5th–95th pct')
    ax.plot(x, pct50,      color='#3b82f6', linewidth=1.2, linestyle='--', label='Median')
    ax.plot(x, actual_cum, color='#f59e0b', linewidth=2.0, label='Actual')
    ax.axhline(0, color='#ef4444', linewidth=0.8, linestyle=':')
    ax.set_title('Equity Path Distribution')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Cumulative PnL (Points)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle='--')

    # Centre: final PnL histogram
    ax2 = axes[1]
    n_positive    = (final_pnls > 0).sum()
    survival_rate = n_positive / n_paths * 100
    n_bins = min(80, max(10, len(np.unique(np.round(final_pnls, 6)))))
    ax2.hist(final_pnls, bins=n_bins, color='#3b82f6', alpha=0.7, edgecolor='none')
    ax2.axvline(0,               color='#ef4444', linewidth=1.5,
                linestyle='--', label='Breakeven')
    ax2.axvline(actual_cum[-1],  color='#f59e0b', linewidth=1.5,
                linestyle='-',  label=f'Actual: {actual_cum[-1]:.5f}')
    ax2.axvline(np.percentile(final_pnls, 5), color='#6b7280', linewidth=1,
                linestyle=':',  label=f'5th pct: {np.percentile(final_pnls, 5):.5f}')
    ax2.set_title(f'Final PnL Distribution\nSurvival Rate: {survival_rate:.1f}%')
    ax2.set_xlabel('Final Cumulative PnL (Points)')
    ax2.set_ylabel('Frequency')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, linestyle='--')

    # Right: max drawdown distribution
    ax3 = axes[2]
    dd_bins = min(60, max(10, len(np.unique(np.round(dd_per_path, 6)))))
    ax3.hist(dd_per_path, bins=dd_bins, color='#ef4444', alpha=0.7, edgecolor='none')
    actual_dd = (actual_cum - np.maximum.accumulate(actual_cum)).min()
    ax3.axvline(actual_dd, color='#f59e0b', linewidth=1.5,
                linestyle='-', label=f'Actual MDD: {actual_dd:.5f}')
    ax3.axvline(np.percentile(dd_per_path, 95), color='#6b7280', linewidth=1,
                linestyle=':', label=f'95th pct: {np.percentile(dd_per_path, 95):.5f}')
    ax3.set_title('Max Drawdown Distribution')
    ax3.set_xlabel('Max Drawdown (Points)')
    ax3.set_ylabel('Frequency')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25, linestyle='--')

    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, 'output', 'plots', 'monte_carlo_paths_v1.0.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Monte Carlo saved: {out_path}")
    plt.close()

    return survival_rate, np.percentile(final_pnls, 5), np.percentile(final_pnls, 95)


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
            MTF_CONFIRM_ACTIVE = False
        else:
            print(f"🧠 Loading 5MIN model: {MTF_MODEL_PATH}")
            model_5m = xgb.Booster()
            model_5m.load_model(MTF_MODEL_PATH)
            df['prob_5m'] = resample_and_predict(df_raw, model_5m, '5min', WINDOW_SIZE)
            df['prob_5m'] = df['prob_5m'].fillna(0.5)
            print(f"   5MIN probs forward-filled onto M1 bars.")
            MTF_CONFIRM_ACTIVE = True
    else:
        MTF_CONFIRM_ACTIVE = False

    # ── Session filter ────────────────────────────────────────────────────────
    if SESSION_FILTER:
        tradeable = df.apply(
            lambda r: is_tradeable_session(int(r['hour']), int(r['day_of_week'])), axis=1
        )
        df['tradeable'] = tradeable
        print(f"📅 Session filter: {(~tradeable).sum():,} bars excluded "
              f"({(~tradeable).mean()*100:.1f}%)")
    else:
        df['tradeable'] = True

    # ── Signal generation ─────────────────────────────────────────────────────
    # Model predicts P(price goes DOWN in next N bars).
    # SHORT trade: we want price to go down → prob > threshold  (correct)
    # LONG  trade: we want price to go up   → (1-prob) > threshold, i.e. prob < (1-threshold)
    if MTF_CONFIRM_ACTIVE:
        short_5m = df['prob_5m'] >  ML_PROB_LIMIT_5M
        long_5m  = df['prob_5m'] < (1 - ML_PROB_LIMIT_5M)
    else:
        short_5m = True
        long_5m  = True

    short_cond = (
        (df['z_score'] >  Z_THRESHOLD) &
        (df['prob_1m'] >  ML_PROB_LIMIT) &
        short_5m &
        df['tradeable']
    )
    long_cond = (
        (df['z_score'] < -Z_THRESHOLD) &
        (df['prob_1m'] < (1 - ML_PROB_LIMIT)) &   # ← fixed: use 1-prob for LONG
        long_5m &
        df['tradeable']
    )

    df['signal'] = 0
    df.loc[short_cond, 'signal'] = -1
    df.loc[long_cond,  'signal'] =  1

    raw_signals = (df['signal'] != 0).sum()
    short_count = (df['signal'] == -1).sum()
    long_count  = (df['signal'] ==  1).sum()
    print(f"🎯 Raw signals: {raw_signals:,}  (SHORT={short_count:,}, LONG={long_count:,})")

    # ── Bar-by-bar simulation ─────────────────────────────────────────────────
    print("🔄 Running bar-by-bar simulation...")
    trades = simulate_trades(
        df,
        signal_col     = 'signal',
        atr_tp_mult    = ATR_TP_MULT,
        atr_sl_mult    = ATR_SL_MULT,
        hold_bars      = HOLD_BARS,
        dynamic_spread = DYNAMIC_SPREAD,
        breakeven_mult = BREAKEVEN_MULT,
    )

    if trades.empty:
        print("⚠️  No trades generated. Try relaxing Z_THRESHOLD or ML_PROB_LIMIT.")
        return

    # ── Extended metrics ──────────────────────────────────────────────────────
    report        = calculate_report(trades)
    reason_counts = trades['exit_reason'].value_counts()

    print(f"\n{'─'*52}")
    print(f"  SNIPER v3.0 ENHANCED BACKTEST — v1.0")
    print(f"{'─'*52}")
    print(f"  Total Trades      : {report['n_trades']}")
    print(f"  Win Rate          : {report['win_rate']*100:.1f}%")
    print(f"  Net Profit        : {report['net_pnl']:.6f} Points")
    print(f"  Profit Factor     : {report['profit_factor']:.3f}")
    print(f"  Sharpe Ratio      : {report['sharpe_ratio']:.2f}")
    print(f"  Calmar Ratio      : {report['calmar_ratio']:.2f}")
    print(f"  Recovery Factor   : {report['recovery_factor']:.2f}")
    print(f"  Max Drawdown      : {report['max_drawdown']:.6f} Points")
    print(f"  Expectancy/Trade  : {report['expectancy']:.6f}")
    print(f"  Avg Win           : {report['avg_win']:.6f}")
    print(f"  Avg Loss          : {report['avg_loss']:.6f}")
    print(f"{'─'*52}")
    print(f"  Exit — TP         : {reason_counts.get('TP', 0)}")
    print(f"  Exit — SL         : {reason_counts.get('SL', 0)}")
    print(f"  Exit — Breakeven  : {reason_counts.get('BE', 0)}")
    print(f"  Exit — TIME       : {reason_counts.get('TIME', 0)}")
    print(f"{'─'*52}")

    # Per-session breakdown
    session_stats = trades.groupby('session').agg(
        count   = ('net_pnl', 'count'),
        net_pnl = ('net_pnl', 'sum'),
        win_rate= ('net_pnl', lambda x: (x > 0).mean()),
    )
    print(f"  {'Session':<12} {'Trades':>7} {'Net PnL':>12} {'WinRate':>9}")
    print(f"  {'-------':<12} {'------':>7} {'-------':>12} {'-------':>9}")
    for sess, row in session_stats.iterrows():
        print(f"  {sess:<12} {row['count']:>7}  {row['net_pnl']:>11.5f}  {row['win_rate']*100:>7.1f}%")
    print(f"{'─'*52}")

    print(f"  Enhancements active:")
    print(f"    True ATR (H-L-C)        : always on")
    print(f"    Correct ML direction    : always on")
    print(f"    Concurrent-trade guard  : always on")
    print(f"    Session filter          : {SESSION_FILTER}")
    print(f"    Dynamic spread          : {DYNAMIC_SPREAD}")
    print(f"    MTF confirmation        : {MTF_CONFIRM_ACTIVE}")
    print(f"    Breakeven stop (mult)   : {BREAKEVEN_MULT if BREAKEVEN_MULT > 0 else 'disabled'}")
    print(f"{'─'*52}\n")

    print("📈 Plotting enhanced dashboard...")
    plot_dashboard(trades, df, report)

    print("🎲 Running Monte Carlo simulation (5,000 paths)...")
    survival, p5, p95 = plot_monte_carlo(trades)
    print(f"   Survival rate (PnL > 0): {survival:.1f}%  |  "
          f"5th pct: {p5:.5f}  |  95th pct: {p95:.5f}")


if __name__ == '__main__':
    run()
