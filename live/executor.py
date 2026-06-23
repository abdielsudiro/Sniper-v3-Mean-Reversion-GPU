"""
Sniper v3.0 — Live MT5 Executor

Trades the same strategy as 03_backtest_v1.0.py against a live MetaTrader 5
terminal. All strategy parameters come from .env automatically.

Required .env keys (add these to your existing .env):
    MT5_LOGIN=12345678
    MT5_PASSWORD=your_password
    MT5_SERVER=YourBroker-Server
    MT5_SYMBOL=EURUSD
    MT5_LOT=0.01
    MT5_MAGIC=203001        # unique int to tag our orders
    MT5_PATH=               # optional: path to terminal64.exe

Optional .env keys (already used by backtest, respected here too):
    Z_THRESHOLD, ML_PROB_LIMIT, ATR_TP_MULT, ATR_SL_MULT,
    HOLD_BARS, BREAKEVEN_MULT, SESSION_WHITELIST, DYNAMIC_SPREAD,
    MTF_CONFIRM, MTF_MODEL_PATH, ML_PROB_LIMIT_5M, WINDOW_SIZE

Usage:
    python live/executor.py                # live trading
    python live/executor.py --dry-run      # signals only, no orders placed
"""

import os
import sys
import time
import logging
import argparse
import signal
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import MetaTrader5 as mt5
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

from config.settings import (
    PROJECT_ROOT, MODEL_SAVE_PATH,
    Z_THRESHOLD, WINDOW_SIZE, ML_PROB_LIMIT, ML_PROB_LIMIT_5M,
    ATR_TP_MULT, ATR_SL_MULT, HOLD_BARS, SPREAD_COST, BREAKEVEN_MULT,
)

# ── MT5 credentials and trading config (from .env) ───────────────────────────
MT5_LOGIN    = int(os.getenv('MT5_LOGIN', '0'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
MT5_SERVER   = os.getenv('MT5_SERVER', '')
MT5_SYMBOL   = os.getenv('MT5_SYMBOL', 'EURUSD')
MT5_LOT      = float(os.getenv('MT5_LOT', '0.01'))
MT5_MAGIC    = int(os.getenv('MT5_MAGIC', '203001'))
_mt5_path_raw = os.getenv('MT5_PATH', '').strip().strip('"').strip("'")
MT5_PATH      = os.path.normpath(_mt5_path_raw) if _mt5_path_raw else None

# ── Feature flags ─────────────────────────────────────────────────────────────
def _bool(key, default):
    return os.getenv(key, str(default)).lower() in ('1', 'true', 'yes')

_sw_raw = os.getenv('SESSION_WHITELIST', '').strip()
if _sw_raw and _sw_raw.lower() != 'all':
    SESSION_WHITELIST = set(s.strip() for s in _sw_raw.lower().split(','))
elif _bool('SESSION_FILTER', True) and not _sw_raw:
    SESSION_WHITELIST = {'london', 'london_ny', 'ny'}
else:
    SESSION_WHITELIST = None

MTF_CONFIRM    = _bool('MTF_CONFIRM',    True)
MTF_MODEL_PATH = os.path.join(
    PROJECT_ROOT, os.getenv('MTF_MODEL_PATH', 'models/5MIN/MREV_5MIN_v1.json')
)

SESSIONS = {'london_ny': (13, 16), 'london': (7, 13), 'ny': (16, 21)}
SPREAD_BY_SESSION = {
    'london_ny': 0.00003, 'london': 0.00005,
    'ny': 0.00007,        'asian':  0.00015, 'sunday': 0.00020,
}

# Poll interval: wake up every N seconds to check for a new M1 bar
POLL_INTERVAL = 10

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(PROJECT_ROOT, 'live', 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)
_log_file = os.path.join(_LOG_DIR, f"executor_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(_log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger('sniper')

# ── Graceful shutdown ─────────────────────────────────────────────────────────
_running = True

def _handle_signal(sig, frame):
    global _running
    log.info("Shutdown signal received — stopping after current bar.")
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Strategy helpers (mirrors 03_backtest_v1.0.py exactly) ───────────────────

def session_label(hour: int, dow: int) -> str:
    if dow == 6:
        return 'sunday'
    if SESSIONS['london_ny'][0] <= hour < SESSIONS['london_ny'][1]:
        return 'london_ny'
    if SESSIONS['london'][0] <= hour < SESSIONS['london'][1]:
        return 'london'
    if SESSIONS['ny'][0] <= hour < SESSIONS['ny'][1]:
        return 'ny'
    return 'asian'


def is_tradeable_session(hour: int, dow: int) -> bool:
    if SESSION_WHITELIST is None:
        return True
    return session_label(hour, dow) in SESSION_WHITELIST


def true_atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


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


# ── MT5 helpers ───────────────────────────────────────────────────────────────

def rates_to_df(rates) -> pd.DataFrame:
    """Convert mt5.copy_rates result to a DataFrame matching the backtest format."""
    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)
    df = df.set_index('datetime')
    df = df.rename(columns={'tick_volume': 'volume'})
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


def fetch_bars(symbol: str, timeframe, count: int) -> pd.DataFrame | None:
    """Fetch the last `count` completed M1 bars from MT5."""
    # start_pos=1 skips the still-forming current bar
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)
    if rates is None or len(rates) == 0:
        log.warning(f"copy_rates_from_pos returned nothing: {mt5.last_error()}")
        return None
    return rates_to_df(rates)


def get_our_position(symbol: str) -> object | None:
    """Return our open position on this symbol (tagged by magic number), or None."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for pos in positions:
        if pos.magic == MT5_MAGIC:
            return pos
    return None


def place_order(symbol: str, direction: int, lot: float,
                tp_price: float, sl_price: float,
                digits: int, dry_run: bool) -> bool:
    """
    Place a market order.
    direction: 1 = LONG (BUY), -1 = SHORT (SELL)
    Returns True on success.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("Cannot get tick — order aborted.")
        return False

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price      = tick.ask               if direction == 1 else tick.bid
    tp_price   = round(tp_price, digits)
    sl_price   = round(sl_price, digits)

    side_str = "LONG  (BUY) " if direction == 1 else "SHORT (SELL)"
    log.info(f"  → Signal: {side_str}  price={price:.{digits}f}  "
             f"TP={tp_price:.{digits}f}  SL={sl_price:.{digits}f}  lot={lot}")

    if dry_run:
        log.info("  [DRY-RUN] Order NOT sent.")
        return True

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      lot,
        "type":        order_type,
        "price":       price,
        "sl":          sl_price,
        "tp":          tp_price,
        "deviation":   20,
        "magic":       MT5_MAGIC,
        "comment":     "Sniper v3.0",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        # Some brokers only support FOK filling — retry
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        log.error(f"Order failed: retcode={retcode}  {mt5.last_error()}")
        return False

    log.info(f"  ✅ Order placed: ticket={result.order}  deal={result.deal}")
    return True


def close_position(pos, symbol: str, digits: int,
                   reason: str, dry_run: bool) -> bool:
    """Close an open position at market price."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid             if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    log.info(f"  → Closing position #{pos.ticket} ({reason})  price={close_price:.{digits}f}")

    if dry_run:
        log.info("  [DRY-RUN] Close NOT sent.")
        return True

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      pos.volume,
        "type":        close_type,
        "position":    pos.ticket,
        "price":       close_price,
        "deviation":   20,
        "magic":       MT5_MAGIC,
        "comment":     f"Sniper v3.0 {reason}",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        log.error(f"Close failed: retcode={retcode}  {mt5.last_error()}")
        return False

    log.info(f"  ✅ Position closed.")
    return True


def modify_sl(pos, new_sl: float, digits: int, dry_run: bool) -> bool:
    """Modify the SL of an open position (used for breakeven)."""
    new_sl = round(new_sl, digits)
    log.info(f"  → Breakeven: moving SL to {new_sl:.{digits}f} (entry price)")

    if dry_run:
        log.info("  [DRY-RUN] Modify NOT sent.")
        return True

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "sl":       new_sl,
        "tp":       pos.tp,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"SL modify failed: {mt5.last_error()}")
        return False

    log.info(f"  ✅ SL moved to breakeven.")
    return True


# ── Main executor ─────────────────────────────────────────────────────────────

def run(dry_run: bool = False):
    # ── Connect to MT5 ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Sniper v3.0 — Live Executor starting")
    log.info("=" * 60)
    log.info(f"  Symbol      : {MT5_SYMBOL}")
    log.info(f"  Lot         : {MT5_LOT}")
    log.info(f"  Z_THRESHOLD : {Z_THRESHOLD}")
    log.info(f"  ML_PROB_LIMIT: {ML_PROB_LIMIT}")
    log.info(f"  ATR TP/SL   : {ATR_TP_MULT}× / {ATR_SL_MULT}×")
    log.info(f"  HOLD_BARS   : {HOLD_BARS}")
    log.info(f"  BREAKEVEN   : {BREAKEVEN_MULT if BREAKEVEN_MULT > 0 else 'disabled'}")
    wl_str = ','.join(sorted(SESSION_WHITELIST)) if SESSION_WHITELIST else 'all'
    log.info(f"  SESSION_WHITELIST: {wl_str}")
    log.info(f"  MTF_CONFIRM : {MTF_CONFIRM}")
    log.info(f"  DRY RUN     : {dry_run}")
    log.info("-" * 60)

    init_kwargs = dict(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if MT5_PATH:
        init_kwargs['path'] = MT5_PATH

    if not mt5.initialize(**init_kwargs):
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return

    account = mt5.account_info()
    if account is None:
        log.error("Cannot get account info — check credentials.")
        mt5.shutdown()
        return

    log.info(f"  Account     : #{account.login}  {account.name}")
    log.info(f"  Broker      : {account.company}")
    log.info(f"  Balance     : {account.balance:.2f} {account.currency}")
    log.info(f"  MT5 Build   : {mt5.version()}")
    log.info("-" * 60)

    # ── Symbol info ───────────────────────────────────────────────────────────
    sym_info = mt5.symbol_info(MT5_SYMBOL)
    if sym_info is None:
        log.error(f"Symbol '{MT5_SYMBOL}' not found — check MT5_SYMBOL in .env")
        mt5.shutdown()
        return

    if not sym_info.visible:
        mt5.symbol_select(MT5_SYMBOL, True)

    DIGITS = sym_info.digits
    POINT  = sym_info.point
    log.info(f"  Symbol info : digits={DIGITS}  point={POINT}")
    log.info("=" * 60)

    # ── Load ML models ────────────────────────────────────────────────────────
    log.info(f"Loading 1MIN model: {MODEL_SAVE_PATH}")
    model_1m = xgb.Booster()
    model_1m.load_model(MODEL_SAVE_PATH)

    model_5m = None
    mtf_active = False
    if MTF_CONFIRM and os.path.exists(MTF_MODEL_PATH):
        log.info(f"Loading 5MIN model: {MTF_MODEL_PATH}")
        model_5m = xgb.Booster()
        model_5m.load_model(MTF_MODEL_PATH)
        mtf_active = True
    elif MTF_CONFIRM:
        log.warning(f"5MIN model not found at {MTF_MODEL_PATH} — MTF disabled.")

    log.info("Models loaded. Entering main loop (Ctrl+C to stop).")
    log.info("=" * 60)

    # ── State ─────────────────────────────────────────────────────────────────
    last_bar_time   = None   # timestamp of last processed bar
    bars_held       = 0      # bars elapsed since position entry
    entry_atr       = None   # ATR at position entry (for breakeven check)
    be_activated    = False  # breakeven stop already moved

    # Bars needed: WINDOW_SIZE for features + extra buffer
    BARS_NEEDED = WINDOW_SIZE + 60

    # ── Main loop ─────────────────────────────────────────────────────────────
    while _running:
        try:
            # ── Fetch completed M1 bars ───────────────────────────────────────
            df_m1 = fetch_bars(MT5_SYMBOL, mt5.TIMEFRAME_M1, BARS_NEEDED)
            if df_m1 is None or len(df_m1) < WINDOW_SIZE + 2:
                time.sleep(POLL_INTERVAL)
                continue

            current_bar_time = df_m1.index[-1]

            # Skip if we already processed this bar
            if current_bar_time == last_bar_time:
                time.sleep(POLL_INTERVAL)
                continue

            last_bar_time = current_bar_time
            now_utc       = datetime.now(timezone.utc).replace(tzinfo=None)
            hour          = current_bar_time.hour
            dow           = current_bar_time.dayofweek

            log.info(f"── New bar: {current_bar_time}  (UTC now: {now_utc.strftime('%H:%M:%S')})")

            # ── Compute features on completed bars ────────────────────────────
            df = compute_features(df_m1, WINDOW_SIZE)
            if df.empty:
                log.warning("Not enough bars after dropna — skipping.")
                time.sleep(POLL_INTERVAL)
                continue

            last = df.iloc[-1]
            z_score = last['z_score']
            atr     = last['atr']
            log.info(f"   Z-Score={z_score:.4f}  ATR={atr:.6f}  "
                     f"Session={session_label(hour, dow)}")

            # ── Check open position ───────────────────────────────────────────
            pos = get_our_position(MT5_SYMBOL)

            if pos is not None:
                bars_held += 1
                tick       = mt5.symbol_info_tick(MT5_SYMBOL)
                cur_price  = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                direction  = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                raw_ret    = (cur_price - pos.price_open) * direction

                log.info(f"   Position #{pos.ticket} | {bars_held} bars held | "
                         f"unrealized={raw_ret:.6f}")

                # Breakeven stop: move SL to entry once profit ≥ trigger
                if (BREAKEVEN_MULT > 0
                        and entry_atr is not None
                        and not be_activated
                        and raw_ret >= entry_atr * BREAKEVEN_MULT):
                    be_activated = True
                    new_sl = pos.price_open + (POINT if direction == 1 else -POINT)
                    modify_sl(pos, new_sl, DIGITS, dry_run)

                # Time exit: close after HOLD_BARS completed bars
                if bars_held >= HOLD_BARS:
                    closed = close_position(pos, MT5_SYMBOL, DIGITS, "TIME", dry_run)
                    if closed:
                        bars_held    = 0
                        entry_atr    = None
                        be_activated = False

            else:
                # No open position — reset state and look for signal
                bars_held    = 0
                entry_atr    = None
                be_activated = False

                # ── Session filter ────────────────────────────────────────────
                if not is_tradeable_session(hour, dow):
                    log.info(f"   Outside tradeable session — no signal.")
                    time.sleep(POLL_INTERVAL)
                    continue

                # ── ML probabilities ──────────────────────────────────────────
                prob_1m = float(get_ml_prob(df.tail(1), model_1m).iloc[-1])

                prob_5m_val = None
                if mtf_active:
                    df_5m = fetch_bars(MT5_SYMBOL, mt5.TIMEFRAME_M5, WINDOW_SIZE + 10)
                    if df_5m is not None and len(df_5m) >= WINDOW_SIZE + 2:
                        df_5m_feat = compute_features(df_5m, WINDOW_SIZE)
                        if not df_5m_feat.empty:
                            prob_5m_val = float(
                                get_ml_prob(df_5m_feat.tail(1), model_5m).iloc[-1]
                            )

                log.info(f"   prob_1m={prob_1m:.4f}" +
                         (f"  prob_5m={prob_5m_val:.4f}" if prob_5m_val is not None else ""))

                # ── Signal logic (mirrors 03_backtest_v1.0.py) ────────────────
                short_5m = (prob_5m_val > ML_PROB_LIMIT_5M)    if mtf_active and prob_5m_val is not None else True
                long_5m  = (prob_5m_val < (1 - ML_PROB_LIMIT_5M)) if mtf_active and prob_5m_val is not None else True

                is_short = (z_score >  Z_THRESHOLD and prob_1m >  ML_PROB_LIMIT       and short_5m)
                is_long  = (z_score < -Z_THRESHOLD and prob_1m < (1 - ML_PROB_LIMIT)  and long_5m)

                if is_short or is_long:
                    direction = -1 if is_short else 1
                    tick      = mt5.symbol_info_tick(MT5_SYMBOL)
                    entry     = tick.ask if direction == 1 else tick.bid

                    tp_price = entry + direction * atr * ATR_TP_MULT
                    sl_price = entry - direction * atr * ATR_SL_MULT

                    placed = place_order(
                        MT5_SYMBOL, direction, MT5_LOT,
                        tp_price, sl_price, DIGITS, dry_run
                    )
                    if placed:
                        entry_atr    = atr
                        be_activated = False
                        bars_held    = 0
                else:
                    log.info(f"   No signal.")

        except Exception as exc:
            log.exception(f"Unexpected error in main loop: {exc}")
            time.sleep(POLL_INTERVAL)

        time.sleep(POLL_INTERVAL)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("Shutting down MT5 connection...")
    mt5.shutdown()
    log.info("Executor stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sniper v3.0 Live MT5 Executor')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate signals but do NOT place real orders')
    args = parser.parse_args()

    if not MT5_LOGIN:
        print("ERROR: MT5_LOGIN not set in .env")
        print("Add the following to your .env:")
        print("  MT5_LOGIN=your_account_number")
        print("  MT5_PASSWORD=your_password")
        print("  MT5_SERVER=YourBroker-Server")
        print("  MT5_SYMBOL=EURUSD")
        print("  MT5_LOT=0.01")
        sys.exit(1)

    run(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
