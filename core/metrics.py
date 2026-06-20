import numpy as np
import pandas as pd


def calculate_metrics(pnl_series):
    """Calculate net profit, profit factor, and max drawdown. Kept for backwards compat."""
    total_net = pnl_series.sum()
    gains = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series < 0].sum())
    profit_factor = gains / losses if losses > 0 else 0

    cum_pnl = pnl_series.cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_dd = drawdown.min()

    return total_net, profit_factor, max_dd


def calculate_report(trades_df: pd.DataFrame) -> dict:
    """
    Extended performance report from a trades DataFrame.
    Required column: 'net_pnl'.
    Optional columns: 'entry_time', 'exit_time' (used for annualization).
    Returns a dict with all metrics.
    """
    pnl = trades_df['net_pnl']
    n   = len(pnl)

    winners = pnl[pnl > 0]
    losers  = pnl[pnl < 0]

    total_net     = pnl.sum()
    gross_profit  = winners.sum()
    gross_loss    = losers.abs().sum()
    pf            = gross_profit / gross_loss if gross_loss > 0 else 0.0

    cum_pnl  = pnl.cumsum()
    peak     = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_dd   = drawdown.min()

    win_rate   = len(winners) / n  if n > 0 else 0.0
    avg_win    = winners.mean()    if len(winners) > 0 else 0.0
    avg_loss   = losers.mean()     if len(losers)  > 0 else 0.0  # negative
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    recovery_factor = total_net / abs(max_dd) if max_dd != 0 else 0.0

    # Annualized Sharpe using trade-level returns
    if n > 1 and pnl.std() > 0:
        if ('entry_time' in trades_df.columns and 'exit_time' in trades_df.columns):
            span_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
            span_years = span_days / 365.25
            trades_per_year = n / span_years if span_years > 0 else n
        else:
            trades_per_year = n
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Calmar = annualized net PnL / abs(max drawdown)
    if max_dd != 0 and ('entry_time' in trades_df.columns and 'exit_time' in trades_df.columns):
        span_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
        span_years = span_days / 365.25
        annualized_pnl = total_net / span_years if span_years > 0 else total_net
        calmar = annualized_pnl / abs(max_dd)
    else:
        calmar = 0.0

    return {
        'n_trades':        n,
        'win_rate':        win_rate,
        'net_pnl':         total_net,
        'gross_profit':    gross_profit,
        'gross_loss':      gross_loss,
        'profit_factor':   pf,
        'max_drawdown':    max_dd,
        'recovery_factor': recovery_factor,
        'sharpe_ratio':    sharpe,
        'calmar_ratio':    calmar,
        'expectancy':      expectancy,
        'avg_win':         avg_win,
        'avg_loss':        avg_loss,
    }
