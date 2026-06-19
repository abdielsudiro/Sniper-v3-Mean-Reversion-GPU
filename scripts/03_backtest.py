import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# 1. Import core metrics and config
from core.metrics import calculate_metrics
from config.settings import (
    BASE_DATA_PATH, MODEL_SAVE_PATH, PROJECT_ROOT,
    Z_THRESHOLD, WINDOW_SIZE, ML_PROB_LIMIT,
    ATR_TP_MULT, ATR_SL_MULT, HOLD_BARS, SPREAD_COST,
    XGB_DEVICE
)

register_matplotlib_converters()

def run_full_dashboard():
    print("📂 Loading data and generating Full Dashboard...")

    # 2. Load data (paths from config)
    df = pd.read_parquet(BASE_DATA_PATH)

    # Compute features (window size from config)
    df['z_score'] = (df['close'] - df['close'].rolling(WINDOW_SIZE).mean()) / df['close'].rolling(WINDOW_SIZE).std()
    df['atr'] = df['close'].rolling(WINDOW_SIZE).std()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df = df.dropna()

    # 3. Load AI model (XGBoost)
    print(f"🧠 Loading AI Model from: {MODEL_SAVE_PATH}")
    model = xgb.Booster()
    model.load_model(MODEL_SAVE_PATH)

    # Prepare data for prediction
    features = ['z_score', 'atr', 'hour', 'day_of_week']
    df['prob'] = model.predict(xgb.DMatrix(df[features]))

    # 4. Simulate trades (parameters from config)
    print(f"🎯 Simulating Trades (Z-Thresh: {Z_THRESHOLD}, Prob: {ML_PROB_LIMIT})")
    temp = df.copy()
    temp['signal'] = 0

    # Entry logic
    temp.loc[(temp['z_score'] > Z_THRESHOLD) & (temp['prob'] > ML_PROB_LIMIT), 'signal'] = -1
    temp.loc[(temp['z_score'] < -Z_THRESHOLD) & (temp['prob'] > ML_PROB_LIMIT), 'signal'] = 1

    # Exit logic — time-based exit after HOLD_BARS bars
    temp['raw_ret'] = (temp['close'].shift(-HOLD_BARS) - temp['close']) * temp['signal']

    # Apply dynamic TP/SL clipping
    tp_dist = temp['atr'] * ATR_TP_MULT
    sl_dist = temp['atr'] * ATR_SL_MULT
    temp['raw_ret'] = np.clip(temp['raw_ret'], -sl_dist, tp_dist)

    # Deduct spread cost only on traded bars
    temp['net_pnl'] = np.where(temp['signal'] != 0, temp['raw_ret'] - SPREAD_COST, 0)

    # 5. Compute summary statistics (via core.metrics)
    trades = temp[temp['signal'] != 0]['net_pnl']
    net, pf, mdd = calculate_metrics(trades)

    print(f"\n--- 📊 SNIPER V3.0 SUMMARY ---")
    print(f"Total Trades : {len(trades)}")
    print(f"Net Profit   : {net:.6f} Points")
    print(f"Profit Factor: {pf:.3f}")
    print(f"Max Drawdown : {mdd:.6f} Points")
    print(f"------------------------------\n")

    # 6. Prepare data for plotting
    temp['cum_pnl'] = temp['net_pnl'].cumsum()
    temp['peak'] = temp['cum_pnl'].cummax()
    temp['drawdown'] = temp['cum_pnl'] - temp['peak']

    # --- Build Full Dashboard ---
    print("📈 Plotting Dashboard...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(temp.index, temp['cum_pnl'], color='#2ca02c', linewidth=1.5, label='Sniper v3.0 Performance')
    ax1.set_title('Sniper v3.0: Full Performance Dashboard', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cumulative Profit (Points)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2.fill_between(temp.index, temp['drawdown'], 0, color='#d62728', alpha=0.5, label='Drawdown')
    ax2.set_ylabel('Drawdown (Points)', fontsize=12)
    ax2.set_xlabel('Timeline', fontsize=12)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot_path = os.path.join(PROJECT_ROOT, 'output/plots/sniper_full_dashboard.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    plt.savefig(plot_path, dpi=300)
    print(f"✅ Dashboard saved successfully to: {plot_path}")

if __name__ == "__main__":
    run_full_dashboard()
