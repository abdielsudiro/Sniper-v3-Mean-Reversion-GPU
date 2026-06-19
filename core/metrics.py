import numpy as np

def calculate_metrics(pnl_series):
    """Calculate net profit, profit factor, and max drawdown."""
    total_net = pnl_series.sum()
    gains = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series < 0].sum())
    profit_factor = gains / losses if losses > 0 else 0
    
    cum_pnl = pnl_series.cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_dd = drawdown.min()
    
    return total_net, profit_factor, max_dd