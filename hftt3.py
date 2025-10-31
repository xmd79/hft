import asyncio
import os
import sys
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import time
import math
from collections import deque, defaultdict
import json
import statistics
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import requests
from binance import AsyncClient, BinanceSocketManager
from binance.client import Client
from binance.exceptions import BinanceWebsocketQueueOverflow

# ---------------------------
# Logging configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration (tune me)
# ---------------------------
SYMBOL = 'BTCUSDC'                 # Target perpetual futures symbol
TIMEFRAME_SECONDS = 15             # Pseudo-candle length in seconds
LEVERAGE = 75                      # Leverage to set
INITIAL_BALANCE = 10.0             # Virtual starting USDC balance
PNL_PERCENT = 0.05                 # Fraction of INITIAL_BALANCE to risk per trade
PIP_VALUE = 1.0                    # $1 per pip for BTC
BASE_SPIKE_THRESHOLD = 10.0        # Base $ amount to consider a spike
BASE_SMALL_CANDLE_THRESHOLD = 5.0  # Base $ threshold for small "base" candles
VOLUME_THRESHOLD = 0.5             # Relative volume threshold for pattern confirmation
MOMENTUM_THRESHOLD = 0.5           # $/s threshold for momentum confirmation
WEBHOOK_URL = None                 # Webhook URL for notifications
MIN_ORDER_SIZE = 0.001             # Minimum quantity (BTC) allowed
SLEEP_INTERVAL = 5                 # Seconds sleep between iterations
WINDOW_SIZE = 5                    # Number of pseudo-candles used for detection
BASE_CANDLE_TIME_LIMIT = 45        # Max seconds for base candles duration

# Backtesting parameters
BACKTEST_MODE = False              # Set to True to run in backtest mode
BACKTEST_DAYS = 30                 # Number of days to backtest
BACKTEST_FILE = 'backtest_results.json'  # File to store backtest results

# Market regime parameters
ADX_PERIOD = 14                    # Period for ADX calculation
TREND_THRESHOLD = 25               # ADX value above which market is considered trending

# Position management parameters
ATR_PERIOD = 14                    # Period for ATR calculation
STOP_LOSS_ATR_MULTIPLIER = 1.5     # Stop loss distance in ATR units
TAKE_PROFIT_ATR_MULTIPLIER = 3.0   # Take profit distance in ATR units
TRAILING_STOP_ATR_MULTIPLIER = 1.0 # Trailing stop distance in ATR units

# Rounding defaults
DEFAULT_QUANTITY_STEP = 0.000001
DEFAULT_PRICE_TICK = 0.01

# Trade analysis parameters
TRADE_FILE = 'trades.csv'
ANALYSIS_FILE = 'trade_analysis.txt'
ANALYSIS_INTERVAL = 3600  # Run analysis every hour (in seconds)
PERIODIC_REPORT_INTERVAL = 300  # Run periodic report every 5 minutes (in seconds)

# ---------------------------
# Global runtime state
# ---------------------------
open_trades = []                   # List of open trades
virtual_balance = INITIAL_BALANCE  # Track virtual balance
candle_data = []                   # List of pseudo-candle dicts
symbol_info_cache = {}             # Cache symbol exchange_info
AUTH = False                       # Whether API keys were provided
backtest_results = []              # Store backtest results
market_regime = "ranging"          # Current market regime (trending/ranging)
volatility_state = "normal"        # Current volatility state (low/normal/high)

# For adaptive thresholds
volatility_history = deque(maxlen=50)  # Store recent volatility measurements
atr_values = deque(maxlen=ATR_PERIOD)  # Store ATR values for calculation

# For market regime detection
adx_values = deque(maxlen=ADX_PERIOD)  # Store ADX values
plus_di_values = deque(maxlen=ADX_PERIOD)
minus_di_values = deque(maxlen=ADX_PERIOD)

# Message processing optimization
message_queue = asyncio.Queue(maxsize=1000)  # Queue for processing messages
processing_task = None  # Reference to the processing task
last_processed_time = time.time()  # Track last processing time
processing_stats = {
    'messages_received': 0,
    'messages_processed': 0,
    'overflows': 0,
    'processing_time': 0
}  # Track processing statistics

# Trade analysis variables
trade_count = 0
last_analysis_time = time.time()
last_periodic_report_time = time.time()  # Track last periodic report time

# ---------------------------
# Read credentials from api.txt
# ---------------------------
def load_credentials():
    """Load API credentials from api.txt file."""
    api_key = None
    api_secret = None
    
    try:
        if os.path.exists("api.txt"):
            with open("api.txt", "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if len(lines) >= 2:
                    api_key = lines[0]
                    api_secret = lines[1]
                    logger.info("API credentials loaded from api.txt")
    except Exception as e:
        logger.error(f"Error reading api.txt: {e}")
    
    # Override with environment variables if present
    env_key = os.getenv("BINANCE_API_KEY")
    env_secret = os.getenv("BINANCE_API_SECRET")
    
    if env_key and env_secret:
        api_key = env_key
        api_secret = env_secret
        logger.info("API credentials loaded from environment variables")
    
    return api_key, api_secret, os.getenv("WEBHOOK_URL")

API_KEY, API_SECRET, WEBHOOK_URL = load_credentials()

if API_KEY and API_SECRET:
    AUTH = True
    logger.info("API credentials found — will attempt real trading")
else:
    AUTH = False
    logger.info("No API credentials found — running in simulation-only mode")

# ---------------------------
# Utilities: rounding to step-size
# ---------------------------
def floor_to_step(q: float, step: float) -> float:
    """Floor quantity to the nearest step size using Decimal for precision safety."""
    if step is None or step == 0:
        return float(q)
    q_dec = Decimal(str(q))
    step_dec = Decimal(str(step))
    steps = (q_dec // step_dec)
    quant = (steps * step_dec).quantize(step_dec, rounding=ROUND_DOWN)
    return float(quant)

async def fetch_symbol_info(async_client, symbol: str):
    """Fetch and cache lot step and price tick details for the symbol."""
    global symbol_info_cache
    key = ("futures", symbol)
    if key in symbol_info_cache:
        return symbol_info_cache[key]
    info = None
    try:
        ex_info = await async_client.futures_exchange_info()
        for s in ex_info.get("symbols", []):
            if s.get("symbol") == symbol:
                info = s
                break
    except Exception as e:
        logger.warning(f"Could not get futures exchange_info: {e}")
    if not info:
        result = {
            "stepSize": DEFAULT_QUANTITY_STEP,
            "minQty": MIN_ORDER_SIZE,
            "tickSize": DEFAULT_PRICE_TICK,
        }
        symbol_info_cache[key] = result
        return result

    filters = {f['filterType']: f for f in info.get('filters', [])}
    lot = filters.get('LOT_SIZE', {})
    price_filter = filters.get('PRICE_FILTER', {})
    stepSize = lot.get('stepSize') or DEFAULT_QUANTITY_STEP
    minQty = lot.get('minQty') or MIN_ORDER_SIZE
    tickSize = price_filter.get('tickSize') or DEFAULT_PRICE_TICK
    try:
        stepSize = float(stepSize)
    except:
        stepSize = DEFAULT_QUANTITY_STEP
    try:
        minQty = float(minQty)
    except:
        minQty = MIN_ORDER_SIZE
    try:
        tickSize = float(tickSize)
    except:
        tickSize = DEFAULT_PRICE_TICK

    result = {"stepSize": stepSize, "minQty": minQty, "tickSize": tickSize}
    symbol_info_cache[key] = result
    return result

# ---------------------------
# Enhanced Market Analysis
# ---------------------------
def calculate_atr(df: pd.DataFrame, period=ATR_PERIOD) -> float:
    """Calculate Average True Range (ATR) for volatility measurement."""
    if len(df) < period + 1:
        return 0.0
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(window=period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else 0.0

def calculate_adx(df: pd.DataFrame, period=ADX_PERIOD) -> tuple:
    """Calculate Average Directional Index (ADX) for trend strength."""
    if len(df) < period + 1:
        return 0.0, 0.0, 0.0
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_smooth = tr.rolling(window=period).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
    
    # Calculate Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Calculate ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0, \
           float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0.0, \
           float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0.0

def update_market_regime(df: pd.DataFrame):
    """Update market regime based on ADX and volatility."""
    global market_regime, volatility_state
    
    if len(df) < ADX_PERIOD + 1:
        return
    
    # Calculate ADX
    adx, plus_di, minus_di = calculate_adx(df)
    adx_values.append(adx)
    plus_di_values.append(plus_di)
    minus_di_values.append(minus_di)
    
    # Determine market regime
    if adx > TREND_THRESHOLD:
        market_regime = "trending"
        if plus_di > minus_di:
            market_regime += "_bullish"
        else:
            market_regime += "_bearish"
    else:
        market_regime = "ranging"
    
    # Calculate and store volatility
    atr = calculate_atr(df)
    atr_values.append(atr)
    
    if len(atr_values) >= ATR_PERIOD:
        current_atr = atr_values[-1]
        avg_atr = statistics.mean(atr_values)
        
        if current_atr > avg_atr * 1.5:
            volatility_state = "high"
        elif current_atr < avg_atr * 0.5:
            volatility_state = "low"
        else:
            volatility_state = "normal"

def get_adaptive_thresholds():
    """Calculate adaptive thresholds based on current volatility."""
    if len(atr_values) < ATR_PERIOD:
        return BASE_SPIKE_THRESHOLD, BASE_SMALL_CANDLE_THRESHOLD
    
    current_atr = atr_values[-1]
    avg_atr = statistics.mean(atr_values)
    
    # Adjust thresholds based on volatility
    volatility_factor = current_atr / avg_atr if avg_atr > 0 else 1.0
    
    spike_threshold = BASE_SPIKE_THRESHOLD * volatility_factor
    small_candle_threshold = BASE_SMALL_CANDLE_THRESHOLD * volatility_factor
    
    return spike_threshold, small_candle_threshold

# ---------------------------
# Enhanced Energy Flow Model
# ---------------------------
def calculate_market_energy(df: pd.DataFrame) -> dict:
    """
    Calculate market energy based on price momentum, volume, and volatility.
    Returns a dictionary with energy components and total energy.
    """
    if len(df) < 3:
        return {"total": 0.0, "momentum": 0.0, "volume": 0.0, "volatility": 0.0}
    
    # Price momentum (rate of change)
    price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
    time_change = (df['start_time'].iloc[-1] - df['start_time'].iloc[-2]) / 1000  # Convert to seconds
    momentum = price_change / time_change if time_change > 0 else 0.0
    
    # Volume momentum
    volume_change = df['volume'].iloc[-1] - df['volume'].iloc[-2]
    volume_momentum = volume_change / time_change if time_change > 0 else 0.0
    
    # Volatility (ATR)
    volatility = calculate_atr(df)
    
    # Normalize components
    max_momentum = max(abs(momentum), 1.0)  # Avoid division by zero
    normalized_momentum = momentum / max_momentum
    
    max_volume = max(df['volume'].max(), 1.0)
    normalized_volume = volume_momentum / max_volume
    
    max_volatility = max(volatility, 1.0)
    normalized_volatility = volatility / max_volatility
    
    # Combine components with weights
    momentum_weight = 0.5
    volume_weight = 0.3
    volatility_weight = 0.2
    
    total_energy = (normalized_momentum * momentum_weight + 
                   normalized_volume * volume_weight + 
                   normalized_volatility * volatility_weight)
    
    return {
        "total": total_energy,
        "momentum": normalized_momentum,
        "volume": normalized_volume,
        "volatility": normalized_volatility
    }

def energy_forecast_delta(df: pd.DataFrame, scale=1.0) -> float:
    """
    Convert market energy into a forecast delta ($).
    Uses recent energy pattern to estimate short-term movement.
    """
    if len(df) < 5:
        return 0.0
    
    energy = calculate_market_energy(df)
    
    # Calculate energy trend
    if len(df) >= 5:
        recent_energies = []
        for i in range(1, 6):
            if len(df) >= i:
                e = calculate_market_energy(df.iloc[:-i])
                recent_energies.append(e["total"])
        
        if recent_energies:
            energy_trend = energy["total"] - statistics.mean(recent_energies)
        else:
            energy_trend = 0.0
    else:
        energy_trend = 0.0
    
    # Combine current energy and trend
    spike_threshold, _ = get_adaptive_thresholds()
    delta = (energy["total"] * 0.7 + energy_trend * 0.3) * spike_threshold * scale
    
    return float(delta)

# ---------------------------
# Candle aggregation & indicators
# ---------------------------
def aggregate_candle(candle_data, trade):
    """Aggregate a single trade message into pseudo-candles."""
    try:
        price = float(trade.get('p', trade.get('price') or 0.0))
        volume = float(trade.get('q', trade.get('qty') or 0.0))
        timestamp = int(trade.get('T', trade.get('time') or int(time.time() * 1000)))
    except Exception as e:
        logger.debug(f"Invalid trade format: {e} - trade: {trade}")
        return candle_data

    if not candle_data:
        candle_data.append({
            'start_time': timestamp,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume
        })
    else:
        current = candle_data[-1]
        if timestamp - current['start_time'] < TIMEFRAME_SECONDS * 1000:
            current['high'] = max(current['high'], price)
            current['low'] = min(current['low'], price)
            current['close'] = price
            current['volume'] += volume
        else:
            candle_data.append({
                'start_time': timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            })
            if len(candle_data) > WINDOW_SIZE:
                candle_data.pop(0)
    return candle_data

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators including EMA, Stochastic, etc."""
    if df.empty:
        return df
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=3, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=8, adjust=False).mean()
    low_min = df['low'].rolling(window=5, min_periods=1).min()
    high_max = df['high'].rolling(window=5, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df['stoch_k'] = 100 * (df['close'] - low_min) / denom
    df['stoch_k'] = df['stoch_k'].fillna(50.0)
    df['stoch_d'] = df['stoch_k'].rolling(window=2, min_periods=1).mean()
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = (df['high'] - df['low']).abs()
    df['body_ratio'] = (df['body'] / df['range'].replace(0, np.nan)).fillna(0.0)
    
    # Get adaptive thresholds
    spike_threshold, small_candle_threshold = get_adaptive_thresholds()
    
    def spike_dir(row):
        if row['range'] > spike_threshold:
            return -1 if row['close'] < row['open'] else 1
        return 0
    df['spike'] = df.apply(spike_dir, axis=1)
    df['momentum'] = df['close'].diff().fillna(0.0) / TIMEFRAME_SECONDS
    df['volume_mean'] = df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Calculate ATR for position management
    df['atr'] = df['high'].rolling(window=ATR_PERIOD).max() - df['low'].rolling(window=ATR_PERIOD).min()
    
    return df

# ---------------------------
# Pattern detection with intensity scoring
# ---------------------------
def score_between(val, low, high):
    """Normalized score (0..1) of val inside low..high; outside clipped."""
    if low == high:
        return 1.0 if val == low else 0.0
    return float(max(0.0, min(1.0, (val - low) / (high - low))))

def pattern_intensity(df, idx):
    """Compute a composite intensity between 0 and 1 based on multiple features."""
    # Weights for different sub-criteria
    w_body_ratio = 0.25
    w_volume = 0.2
    w_momentum = 0.2
    w_ema = 0.15
    w_stoch = 0.2

    i = idx
    scores = []

    # body_ratio closeness (prefer >0.7 for spike/rejection)
    br = df['body_ratio'].iloc[i-1] if i-1 >= 0 else 0.0
    br_score = score_between(br, 0.4, 1.0)  # 0.4..1.0 maps to 0..1
    scores.append(w_body_ratio * br_score)

    # volume: how large spike/base volumes are relative to mean
    vol_mean = df['volume_mean'].iloc[i-1] if i-1 >= 0 else 0.0
    vol = df['volume'].iloc[i-1] if i-1 >= 0 else 0.0
    if vol_mean > 0:
        vol_ratio = vol / vol_mean
        vol_score = score_between(vol_ratio, 0.5, 3.0)  # 0.5..3.0 -> 0..1
    else:
        vol_score = 0.0
    scores.append(w_volume * vol_score)

    # momentum magnitude
    mom = abs(df['momentum'].iloc[i-1]) if i-1 >= 0 else 0.0
    mom_score = score_between(mom, 0.0, MOMENTUM_THRESHOLD * 4)
    scores.append(w_momentum * mom_score)

    # EMA alignment: prefer fast < slow for sell, > for buy
    ema_diff = df['ema_fast'].iloc[i] - df['ema_slow'].iloc[i]
    ema_score = score_between(abs(ema_diff), 0.0, max(abs(df['close'].iloc[i]) * 0.01, 1.0))
    scores.append(w_ema * ema_score)

    # stochastic: extremes are supportive (near 0 or 100)
    sk = df['stoch_k'].iloc[i]
    stoch_extreme = max(score_between(sk, 0, 20), score_between(sk, 80, 100))
    scores.append(w_stoch * stoch_extreme)

    total_score = sum(scores)
    # clamp 0..1
    return float(max(0.0, min(1.0, total_score)))

def detect_patterns_with_intensity(df):
    """Detect patterns and produce tuples with action, entry price, intensity, and pattern name."""
    signals = []
    if df is None or len(df) < 4:
        return signals
    i = len(df) - 1

    # early guard
    if i < 3:
        return signals

    # Get adaptive thresholds
    spike_threshold, small_candle_threshold = get_adaptive_thresholds()
    
    # reused booleans similar to earlier logic
    high_volume = df['volume'].iloc[i-3] > df['volume_mean'].iloc[i-3] * VOLUME_THRESHOLD
    low_base_volume = df['volume'].iloc[i-2] < df['volume_mean'].iloc[i-2] * VOLUME_THRESHOLD
    high_momentum = abs(df['momentum'].iloc[i-3]) > MOMENTUM_THRESHOLD
    low_base_momentum = abs(df['momentum'].iloc[i-2]) < MOMENTUM_THRESHOLD
    time_valid = (df['start_time'].iloc[i-1] - df['start_time'].iloc[i-3]) <= BASE_CANDLE_TIME_LIMIT * 1000

    spike_ok = df['range'].iloc[i-3] > spike_threshold and high_volume and high_momentum
    base_ok = df['body_ratio'].iloc[i-2] < 0.25 and df['range'].iloc[i-2] < small_candle_threshold and low_base_volume and low_base_momentum
    confirm_candle_ok = df['body_ratio'].iloc[i-1] > 0.7 and df['range'].iloc[i-1] > 0

    # Adjust pattern detection based on market regime
    if market_regime.startswith("trending"):
        # In trending markets, we want to be more selective with reversals
        stoch_threshold = 15 if market_regime.endswith("bullish") else 85
    else:
        # In ranging markets, standard thresholds
        stoch_threshold = 20 if market_regime.endswith("bullish") else 80

    # DBD
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] < df['open'].iloc[i-3] and
            df['close'].iloc[i-1] < df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] > stoch_threshold):
        intensity = pattern_intensity(df, i)
        signals.append(('sell', float(df['close'].iloc[i]), int(intensity * 100), 'DBD'))

    # RBD
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] > df['open'].iloc[i-3] and
            df['close'].iloc[i-1] < df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] > stoch_threshold):
        intensity = pattern_intensity(df, i)
        signals.append(('sell', float(df['close'].iloc[i]), int(intensity * 100), 'RBD'))

    # DBR
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] < df['open'].iloc[i-3] and
            df['close'].iloc[i-1] > df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] < (100 - stoch_threshold)):
        intensity = pattern_intensity(df, i)
        signals.append(('buy', float(df['close'].iloc[i]), int(intensity * 100), 'DBR'))

    # RBR
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] > df['open'].iloc[i-3] and
            df['close'].iloc[i-1] > df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] < (100 - stoch_threshold)):
        intensity = pattern_intensity(df, i)
        signals.append(('buy', float(df['close'].iloc[i]), int(intensity * 100), 'RBR'))

    # Spike Buy (downward spike rejection)
    if (df['spike'].iloc[i-1] == -1 and
            df['open'].iloc[i] > df['low'].iloc[i-1] and
            df['volume'].iloc[i-1] > df['volume_mean'].iloc[i-1] * VOLUME_THRESHOLD and
            abs(df['momentum'].iloc[i-1]) > MOMENTUM_THRESHOLD and
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] < (100 - stoch_threshold)):
        intensity = pattern_intensity(df, i)
        # bump intensity for spike-specific criteria
        intensity = min(1.0, intensity + 0.15)
        signals.append(('buy', float(df['close'].iloc[i]), int(intensity * 100), 'SpikeBuy'))

    # Spike Sell (upward spike rejection)
    if (df['spike'].iloc[i-1] == 1 and
            df['open'].iloc[i] < df['high'].iloc[i-1] and
            df['volume'].iloc[i-1] > df['volume_mean'].iloc[i-1] * VOLUME_THRESHOLD and
            abs(df['momentum'].iloc[i-1]) > MOMENTUM_THRESHOLD and
            df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] > stoch_threshold):
        intensity = pattern_intensity(df, i)
        intensity = min(1.0, intensity + 0.15)
        signals.append(('sell', float(df['close'].iloc[i]), int(intensity * 100), 'SpikeSell'))

    # Extra quick spike predictor: large sudden momentum on last tick
    last_mom = abs(df['momentum'].iloc[-1])
    if last_mom > MOMENTUM_THRESHOLD * 3:
        direction = 'buy' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'sell'
        intensity = min(100, int(score_between(last_mom, MOMENTUM_THRESHOLD*3, MOMENTUM_THRESHOLD*8) * 100 + 20))
        signals.append((direction, float(df['close'].iloc[-1]), intensity, 'QuickMomentumSpike'))

    # Deduplicate signals preferring higher intensity for same action
    dedup = {}
    for s in signals:
        action, price, intensity, name = s
        key = (action)
        if key not in dedup or intensity > dedup[key][1]:
            dedup[key] = (name, intensity, price)
    final = [(action, dedup[action][2], dedup[action][1], dedup[action][0]) for action in dedup]
    return final

# ---------------------------
# Enhanced Position Management
# ---------------------------
async def calculate_quantity_and_levels(entry_price: float, action: str, async_client, df: pd.DataFrame) -> tuple:
    """
    Compute quantity (floored to step), stop_loss, take_profit, and can_trade boolean.
    Uses ATR for dynamic stop-loss and take-profit levels.
    """
    global virtual_balance

    risk_amount = virtual_balance * PNL_PERCENT
    
    # Calculate ATR for dynamic position sizing
    atr = calculate_atr(df) if not df.empty else 0.0
    if atr == 0.0:
        atr = BASE_SPIKE_THRESHOLD / 2.0  # Fallback to fixed value
    
    # Use ATR for stop distance
    stop_distance = atr * STOP_LOSS_ATR_MULTIPLIER
    take_profit_distance = atr * TAKE_PROFIT_ATR_MULTIPLIER
    
    raw_quantity = risk_amount / (stop_distance * PIP_VALUE) if stop_distance > 0 else 0.0
    margin_required = (raw_quantity * entry_price) / LEVERAGE if entry_price > 0 else float('inf')

    stepSize = DEFAULT_QUANTITY_STEP
    minQty = MIN_ORDER_SIZE
    try:
        info = await fetch_symbol_info(async_client, SYMBOL)
        stepSize = info.get('stepSize', DEFAULT_QUANTITY_STEP)
        minQty = info.get('minQty', MIN_ORDER_SIZE)
    except Exception:
        pass

    max_q_by_margin = (virtual_balance * LEVERAGE) / entry_price if entry_price > 0 else 0.0
    max_q_by_margin *= 0.995

    quantity = min(raw_quantity, max_q_by_margin)
    quantity = floor_to_step(quantity, stepSize)

    if quantity < minQty or quantity < MIN_ORDER_SIZE:
        can_trade = False
    else:
        can_trade = True if AUTH and (margin_required <= virtual_balance) else False
        margin_required = (quantity * entry_price) / LEVERAGE if entry_price > 0 else float('inf')
        if AUTH and margin_required > virtual_balance:
            can_trade = False

    # Set dynamic stop-loss and take-profit based on ATR
    if action == 'buy':
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + take_profit_distance
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - take_profit_distance

    quantity = float(round(quantity, 8))
    return quantity, stop_loss, take_profit, can_trade

async def execute_trade(async_client, action, entry_price, quantity, stop_loss, take_profit, can_trade, df: pd.DataFrame, pattern_name='', intensity=0):
    """
    Execute a real trade (async) or simulate it. Adds trade to open_trades.
    Implements trailing stop for trending markets.
    """
    global virtual_balance, open_trades, trade_count
    
    # Determine if we should use trailing stop
    use_trailing_stop = market_regime.startswith("trending")
    trailing_distance = calculate_atr(df) * TRAILING_STOP_ATR_MULTIPLIER if not df.empty else 0.0
    
    trade_info = {
        'action': action,
        'entry_price': entry_price,
        'quantity': quantity,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'is_simulated': not can_trade,
        'entry_time': datetime.now(timezone.utc).isoformat(),
        'use_trailing_stop': use_trailing_stop,
        'trailing_distance': trailing_distance,
        'highest_price': entry_price if action == 'buy' else 0.0,
        'lowest_price': entry_price if action == 'sell' else float('inf'),
        'market_regime': market_regime,
        'volatility_state': volatility_state,
        'pattern': pattern_name,
        'intensity': intensity
    }

    if can_trade and async_client and AUTH:
        try:
            side_open = 'BUY' if action == 'buy' else 'SELL'
            await async_client.futures_create_order(
                symbol=SYMBOL,
                side=side_open,
                type='MARKET',
                quantity=str(quantity)
            )
            
            # For trailing stop, we'll monitor and update manually
            if not use_trailing_stop:
                side_sl = 'SELL' if action == 'buy' else 'BUY'
                await async_client.futures_create_order(
                    symbol=SYMBOL,
                    side=side_sl,
                    type='STOP_MARKET',
                    stopPrice=str(round(stop_loss, 8)),
                    closePosition=False,
                    reduceOnly=True,
                    quantity=str(quantity)
                )
            
            # Always set take profit
            side_tp = 'SELL' if action == 'buy' else 'BUY'
            await async_client.futures_create_order(
                symbol=SYMBOL,
                side=side_tp,
                type='TAKE_PROFIT_MARKET',
                stopPrice=str(round(take_profit, 8)),
                closePosition=False,
                reduceOnly=True,
                quantity=str(quantity)
            )
            
            margin_used = (quantity * entry_price) / LEVERAGE
            virtual_balance -= margin_used
            logger.info(f"Real trade placed: {action.upper()} {quantity} {SYMBOL} at {entry_price:.2f} | SL {stop_loss:.2f} TP {take_profit:.2f} | margin used: ${margin_used:.4f}")
        except Exception as e:
            logger.error(f"Real trade execution failed: {e}")
            trade_info['is_simulated'] = True
    else:
        reason = "not authorized" if not AUTH else "insufficient margin or size"
        logger.info(f"Simulating trade due to {reason}: {action.upper()} {quantity} {SYMBOL} at ${entry_price:.2f}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")

    open_trades.append(trade_info)
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] ENTRY -> {action.upper():4} | Qty {quantity:.6f} | Price {entry_price:.2f} | SL {stop_loss:.2f} | TP {take_profit:.2f} | Simulated: {trade_info['is_simulated']} | Pattern: {pattern_name}")
    
    # Write trade entry to file
    write_trade_result(trade_info)
    trade_count += 1

def send_webhook(action, entry_price, stop_loss, take_profit, quantity, is_simulated):
    """Send webhook (synchronous)."""
    if not WEBHOOK_URL:
        return
    payload = {
        'action': action,
        'symbol': SYMBOL,
        'leverage': LEVERAGE,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'quantity': quantity,
        'is_simulated': is_simulated,
        'timeframe': f"{TIMEFRAME_SECONDS}s",
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'market_regime': market_regime,
        'volatility_state': volatility_state
    }
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        resp.raise_for_status()
        logger.info(f"Webhook sent successfully: {payload}")
    except Exception as e:
        logger.error(f"Webhook send failed: {e}")

# ---------------------------
# Enhanced Trade Monitoring
# ---------------------------
async def track_trades(current_price):
    """
    Check open_trades against current_price and close ones that hit SL or TP.
    Implements trailing stop for trending markets.
    """
    global virtual_balance, open_trades, trade_count, last_analysis_time, last_periodic_report_time
    if current_price is None:
        return
    closed = []
    for trade in open_trades:
        action = trade['action']
        entry_price = trade['entry_price']
        quantity = trade['quantity']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        is_simulated = trade['is_simulated']
        use_trailing_stop = trade.get('use_trailing_stop', False)
        trailing_distance = trade.get('trailing_distance', 0.0)

        # Update trailing stop if enabled
        if use_trailing_stop and trailing_distance > 0:
            if action == 'buy':
                # Update highest price seen
                if current_price > trade['highest_price']:
                    trade['highest_price'] = current_price
                    # Update stop loss to trail
                    new_stop_loss = current_price - trailing_distance
                    if new_stop_loss > stop_loss:
                        trade['stop_loss'] = new_stop_loss
                        stop_loss = new_stop_loss
            else:  # sell
                # Update lowest price seen
                if current_price < trade['lowest_price']:
                    trade['lowest_price'] = current_price
                    # Update stop loss to trail
                    new_stop_loss = current_price + trailing_distance
                    if new_stop_loss < stop_loss:
                        trade['stop_loss'] = new_stop_loss
                        stop_loss = new_stop_loss

        # Check if trade should be closed
        exit_price = None
        pnl = 0
        if action == 'buy':
            if current_price <= stop_loss:
                exit_price = stop_loss
                pnl = (stop_loss - entry_price) * quantity * PIP_VALUE
                closed.append(trade)
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] EXIT  -> BUY closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Simulated: {is_simulated}")
                if not is_simulated:
                    virtual_balance += pnl
            elif current_price >= take_profit:
                exit_price = take_profit
                pnl = (take_profit - entry_price) * quantity * PIP_VALUE
                closed.append(trade)
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] EXIT  -> BUY closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Simulated: {is_simulated}")
                if not is_simulated:
                    virtual_balance += pnl
        else:  # sell
            if current_price >= stop_loss:
                exit_price = stop_loss
                pnl = (entry_price - stop_loss) * quantity * PIP_VALUE
                closed.append(trade)
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] EXIT  -> SELL closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Simulated: {is_simulated}")
                if not is_simulated:
                    virtual_balance += pnl
            elif current_price <= take_profit:
                exit_price = take_profit
                pnl = (entry_price - take_profit) * quantity * PIP_VALUE
                closed.append(trade)
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] EXIT  -> SELL closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Simulated: {is_simulated}")
                if not is_simulated:
                    virtual_balance += pnl
        
        # Record closed trade
        if exit_price is not None:
            trade['exit_price'] = exit_price
            trade['pnl'] = pnl
            write_trade_result(trade)
            trade_count += 1

    open_trades[:] = [t for t in open_trades if t not in closed]
    
    # Run analysis periodically (hourly)
    current_time = time.time()
    if current_time - last_analysis_time > ANALYSIS_INTERVAL:
        analyze_trades()
    
    # Run periodic report (every 5 minutes)
    if current_time - last_periodic_report_time > PERIODIC_REPORT_INTERVAL:
        print_periodic_report()
        last_periodic_report_time = current_time

# ---------------------------
# Trade Recording and Analysis
# ---------------------------
def write_trade_result(trade):
    """Write trade result to CSV file."""
    file_exists = Path(TRADE_FILE).exists()
    
    with open(TRADE_FILE, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'action', 'entry_price', 'exit_price', 
            'quantity', 'pnl', 'balance', 'market_regime', 
            'volatility_state', 'pattern', 'intensity'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': trade['action'],
            'entry_price': trade['entry_price'],
            'exit_price': trade.get('exit_price', 0),
            'quantity': trade['quantity'],
            'pnl': trade.get('pnl', 0),
            'balance': virtual_balance,
            'market_regime': trade.get('market_regime', market_regime),
            'volatility_state': trade.get('volatility_state', volatility_state),
            'pattern': trade.get('pattern', ''),
            'intensity': trade.get('intensity', 0)
        })

def analyze_trades():
    """Analyze batch of trades and save results."""
    global trade_count, last_analysis_time
    
    if not Path(TRADE_FILE).exists():
        logger.info("No trades to analyze")
        return
    
    try:
        df = pd.read_csv(TRADE_FILE)
        
        if df.empty:
            logger.info("No trades to analyze")
            return
        
        # Calculate metrics
        total_trades = len(df)
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        max_win = df['pnl'].max()
        max_loss = df['pnl'].min()
        
        # Calculate profit factor
        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        df['cum_pnl'] = df['pnl'].cumsum()
        df['peak'] = df['cum_pnl'].cummax()
        df['drawdown'] = df['peak'] - df['cum_pnl']
        max_drawdown = df['drawdown'].max()
        
        # Prepare analysis results
        analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_win': max_win,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'final_balance': virtual_balance
        }
        
        # Save to file
        file_exists = Path(ANALYSIS_FILE).exists()
        with open(ANALYSIS_FILE, 'a' if file_exists else 'w', newline='') as f:
            if not file_exists:
                f.write("timestamp,total_trades,winning_trades,losing_trades,win_rate,total_pnl,avg_pnl,max_win,max_loss,profit_factor,max_drawdown,final_balance\n")
            
            f.write(
                f"{analysis['timestamp']},{analysis['total_trades']},{analysis['winning_trades']},"
                f"{analysis['losing_trades']},{analysis['win_rate']:.2f},{analysis['total_pnl']:.4f},"
                f"{analysis['avg_pnl']:.4f},{analysis['max_win']:.4f},{analysis['max_loss']:.4f},"
                f"{analysis['profit_factor']:.2f},{analysis['max_drawdown']:.4f},{analysis['final_balance']:.4f}\n"
            )
        
        # Print to console
        print("\n" + "="*50)
        print("TRADE ANALYSIS REPORT")
        print("="*50)
        print(f"Analysis Time: {analysis['timestamp']}")
        print(f"Total Trades: {analysis['total_trades']}")
        print(f"Winning Trades: {analysis['winning_trades']} ({analysis['win_rate']:.2f}%)")
        print(f"Losing Trades: {analysis['losing_trades']}")
        print(f"Total PnL: ${analysis['total_pnl']:.4f}")
        print(f"Average PnL: ${analysis['avg_pnl']:.4f}")
        print(f"Max Win: ${analysis['max_win']:.4f}")
        print(f"Max Loss: ${analysis['max_loss']:.4f}")
        print(f"Profit Factor: {analysis['profit_factor']:.2f}")
        print(f"Max Drawdown: ${analysis['max_drawdown']:.4f}")
        print(f"Final Balance: ${analysis['final_balance']:.4f}")
        print("="*50 + "\n")
        
        # Update last analysis time
        last_analysis_time = time.time()
        
    except Exception as e:
        logger.error(f"Error analyzing trades: {e}")

def print_periodic_report():
    """Print a periodic report every 5 minutes with trading performance metrics."""
    global virtual_balance
    
    if not Path(TRADE_FILE).exists():
        print("\n" + "="*50)
        print("5-MINUTE PERFORMANCE REPORT")
        print("="*50)
        print("No trades recorded yet.")
        print(f"Initial Balance: ${INITIAL_BALANCE:.4f}")
        print(f"Current Balance: ${virtual_balance:.4f}")
        print(f"Net PnL: ${virtual_balance - INITIAL_BALANCE:.4f}")
        print("="*50 + "\n")
        return
    
    try:
        df = pd.read_csv(TRADE_FILE)
        
        if df.empty:
            print("\n" + "="*50)
            print("5-MINUTE PERFORMANCE REPORT")
            print("="*50)
            print("No trades recorded yet.")
            print(f"Initial Balance: ${INITIAL_BALANCE:.4f}")
            print(f"Current Balance: ${virtual_balance:.4f}")
            print(f"Net PnL: ${virtual_balance - INITIAL_BALANCE:.4f}")
            print("="*50 + "\n")
            return
        
        # Get closed trades (with exit price)
        closed_trades = df[df['exit_price'] != 0]
        
        # Calculate metrics
        total_closed_trades = len(closed_trades)
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_closed_trades * 100 if total_closed_trades > 0 else 0
        total_pnl = closed_trades['pnl'].sum()
        avg_pnl = closed_trades['pnl'].mean() if total_closed_trades > 0 else 0
        max_win = closed_trades['pnl'].max() if not winning_trades.empty else 0
        max_loss = closed_trades['pnl'].min() if not losing_trades.empty else 0
        
        # Calculate profit factor
        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Get open trades count
        open_trades_count = len(open_trades)
        
        # Print report
        print("\n" + "="*50)
        print("5-MINUTE PERFORMANCE REPORT")
        print("="*50)
        print(f"Report Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial Balance: ${INITIAL_BALANCE:.4f}")
        print(f"Current Balance: ${virtual_balance:.4f}")
        print(f"Net PnL: ${virtual_balance - INITIAL_BALANCE:.4f}")
        print(f"Total Closed Trades: {total_closed_trades}")
        print(f"Open Trades: {open_trades_count}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.2f}%)")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Total PnL from Closed Trades: ${total_pnl:.4f}")
        print(f"Average PnL per Trade: ${avg_pnl:.4f}")
        print(f"Max Win: ${max_win:.4f}")
        print(f"Max Loss: ${max_loss:.4f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error printing periodic report: {e}")

# ---------------------------
# Backtesting Functionality
# ---------------------------
async def run_backtest(async_client):
    """Run backtest using historical data."""
    global virtual_balance, backtest_results, candle_data
    
    logger.info(f"Starting backtest for the last {BACKTEST_DAYS} days...")
    
    # Reset state
    virtual_balance = INITIAL_BALANCE
    open_trades.clear()
    backtest_results.clear()
    candle_data.clear()
    
    # Get historical klines
    klines = await async_client.futures_klines(
        symbol=SYMBOL,
        interval=AsyncClient.KLINE_INTERVAL_1MINUTE,
        limit=BACKTEST_DAYS * 1440  # 1440 minutes per day
    )
    
    # Process klines
    for kline in klines:
        # Convert kline to trade format
        trade = {
            'p': float(kline[4]),  # Close price
            'q': float(kline[5]),  # Volume
            'T': int(kline[6])     # Close time
        }
        
        # Aggregate into pseudo-candles
        candle_data = aggregate_candle(candle_data, trade)
        
        # Build df & indicators when enough data
        df = pd.DataFrame(candle_data)
        if not df.empty:
            df = calculate_indicators(df)
            update_market_regime(df)
        
        # Detect patterns with intensities
        signals = detect_patterns_with_intensity(df) if not df.empty else []
        
        # Execute signals
        for action, entry_price, intensity_pct, pname in signals:
            quantity, stop_loss, take_profit, can_trade = await calculate_quantity_and_levels(entry_price, action, async_client, df)
            await execute_trade(async_client, action, entry_price, quantity, stop_loss, take_profit, False, df, pname, intensity_pct)
        
        # Track trades on current price
        current_price = float(trade['p'])
        await track_trades(current_price)
    
    # Save backtest results
    with open(BACKTEST_FILE, 'w') as f:
        json.dump(backtest_results, f, indent=2)
    
    # Calculate performance metrics
    if backtest_results:
        total_trades = len(backtest_results)
        winning_trades = sum(1 for r in backtest_results if r['pnl'] > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(r['pnl'] for r in backtest_results)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        logger.info(f"Backtest completed. Total trades: {total_trades}, Win rate: {win_rate:.2f}%, Total PnL: ${total_pnl:.4f}, Avg PnL: ${avg_pnl:.4f}")
    else:
        logger.info("Backtest completed. No trades were executed.")

# ---------------------------
# Enhanced message processing
# ---------------------------
async def process_message(message, async_client):
    """Process a single trade message with error handling."""
    global processing_stats, candle_data
    processing_stats['messages_processed'] += 1
    start_time = time.time()
    
    try:
        # Check if message is valid
        if not message or ('p' not in message and 'price' not in message):
            return
        
        # Aggregate into pseudo-candles
        candle_data = aggregate_candle(candle_data, message)
        
        # Only build df & indicators when we have enough data and periodically
        # to avoid excessive processing
        if len(candle_data) >= 3 and (processing_stats['messages_processed'] % 5 == 0):
            df = pd.DataFrame(candle_data)
            if not df.empty:
                df = calculate_indicators(df)
                update_market_regime(df)
                
                # Forecast price delta from energy model
                forecast_delta = energy_forecast_delta(df)
                current_price = float(message.get('p', df['close'].iloc[-1] if not df.empty else 0.0))
                forecast_price = current_price + forecast_delta
                
                # Detect patterns with intensities
                signals = detect_patterns_with_intensity(df) if not df.empty else []
                
                # Print status periodically
                if processing_stats['messages_processed'] % 20 == 0:
                    latest = df.iloc[-1] if not df.empty else None
                    print_main_status(latest, forecast_price, forecast_delta, signals)
                
                # Execute signals (for each, size and attempt trade)
                for action, entry_price, intensity_pct, pname in signals:
                    # For extremely high intensity (>=80) allow larger price_move scaling
                    quantity, stop_loss, take_profit, can_trade = await calculate_quantity_and_levels(entry_price, action, async_client, df)
                    # slight scale for intensity: bring TP closer if intensity is high (capture quick fill)
                    if intensity_pct >= 80:
                        # shrink price_move by 50% to make TP close; recalc TP/SL around entry
                        atr = calculate_atr(df) if not df.empty else 0.0
                        if atr > 0:
                            price_move = atr * TAKE_PROFIT_ATR_MULTIPLIER
                            price_move *= 0.5
                            if action == 'buy':
                                stop_loss = entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER)
                                take_profit = entry_price + price_move
                            else:
                                stop_loss = entry_price + (atr * STOP_LOSS_ATR_MULTIPLIER)
                                take_profit = entry_price - price_move
                    await execute_trade(async_client, action, entry_price, quantity, stop_loss, take_profit, can_trade, df, pname, intensity_pct)
                    send_webhook(action, entry_price, stop_loss, take_profit, quantity, not can_trade)
                
                # Track trades on current price
                await track_trades(current_price)
                
                # ensure candle_data trimmed
                if len(candle_data) > WINDOW_SIZE:
                    candle_data = candle_data[-WINDOW_SIZE:]
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
    finally:
        # Update processing statistics
        processing_time = time.time() - start_time
        processing_stats['processing_time'] += processing_time

async def message_processor(async_client):
    """Process messages from the queue with rate limiting."""
    global last_processed_time
    
    while True:
        try:
            # Get a message from the queue with timeout
            message = await asyncio.wait_for(message_queue.get(), timeout=1.0)
            
            # Process the message
            await process_message(message, async_client)
            
            # Update last processed time
            last_processed_time = time.time()
            
            # Small delay to prevent CPU overload
            await asyncio.sleep(0.001)
            
        except asyncio.TimeoutError:
            # No messages in queue, continue loop
            continue
        except Exception as e:
            logger.error(f"Error in message processor: {e}")
            await asyncio.sleep(0.1)

async def process_trade_stream(async_client):
    """
    Consume trade websocket, queue messages, and process them asynchronously.
    """
    global processing_stats, processing_task
    
    # Run backtest if enabled
    if BACKTEST_MODE:
        await run_backtest(async_client)
        return
    
    # Start the message processor task
    processing_task = asyncio.create_task(message_processor(async_client))
    
    bsm = BinanceSocketManager(async_client)
    ws_symbol = SYMBOL.lower()
    
    # Create the trade socket without queue_size parameter
    ts = bsm.trade_socket(ws_symbol)
    
    logger.info(f"Listening trade socket for {SYMBOL}...")
    
    async with ts:
        while True:
            try:
                # Receive message with timeout to allow for queue processing
                trade = await asyncio.wait_for(ts.recv(), timeout=5.0)
                
                # Update statistics
                processing_stats['messages_received'] += 1
                
                # Try to put message in queue, but don't block if queue is full
                try:
                    message_queue.put_nowait(trade)
                except asyncio.QueueFull:
                    # Queue is full, log warning and drop message
                    processing_stats['overflows'] += 1
                    logger.warning(f"Message queue full, dropping message. Overflows: {processing_stats['overflows']}")
                    
                    # If we have too many overflows, log more details
                    if processing_stats['overflows'] % 100 == 0:
                        queue_size = message_queue.qsize()
                        logger.warning(f"Queue statistics: Received={processing_stats['messages_received']}, "
                                     f"Processed={processing_stats['messages_processed']}, "
                                     f"Overflows={processing_stats['overflows']}, "
                                     f"QueueSize={queue_size}, "
                                     f"AvgProcessTime={processing_stats['processing_time'] / max(1, processing_stats['messages_processed']):.4f}s")
                
                # Periodically log processing statistics
                if processing_stats['messages_received'] % 1000 == 0:
                    queue_size = message_queue.qsize()
                    logger.info(f"Processing statistics: Received={processing_stats['messages_received']}, "
                              f"Processed={processing_stats['messages_processed']}, "
                              f"Overflows={processing_stats['overflows']}, "
                              f"QueueSize={queue_size}, "
                              f"AvgProcessTime={processing_stats['processing_time'] / max(1, processing_stats['messages_processed']):.4f}s")
                
            except asyncio.TimeoutError:
                # No message received within timeout, continue loop
                continue
            except BinanceWebsocketQueueOverflow as e:
                # Handle the specific overflow exception
                processing_stats['overflows'] += 1
                logger.error(f"Binance WebSocket queue overflow: {e}. Overflows: {processing_stats['overflows']}")
                
                # Sleep briefly to allow the queue to drain
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                # Sleep before retrying
                await asyncio.sleep(SLEEP_INTERVAL)
    
    # Clean up the processing task when done
    if processing_task:
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass

# ---------------------------
# Printing / UI helpers
# ---------------------------
def format_signals_for_print(signals):
    if not signals:
        return "none"
    parts = []
    for action, price, intensity, name in signals:
        parts.append(f"{name}:{action}@{price:.2f} ({intensity}%)")
    return " | ".join(parts)

def print_main_status(latest_row, forecast_price, forecast_delta, signals):
    try:
        ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
        if latest_row is None:
            print(f"[{ts}] No candle yet | ForecastΔ: {forecast_delta:+.4f} | ForecastPrice: {forecast_price:.2f} | Signals: none | Balance: ${virtual_balance:.4f}")
            return
        price = float(latest_row['close'])
        vol = float(latest_row['volume'])
        momentum = float(latest_row.get('momentum', 0.0))
        ema_f = float(latest_row.get('ema_fast', 0.0))
        ema_s = float(latest_row.get('ema_slow', 0.0))
        stoch_k = float(latest_row.get('stoch_k', 0.0))
        open_count = len(open_trades)
        sig_text = format_signals_for_print(signals)
        print(f"[{ts}] Price: {price:.2f} | Vol: {vol:.6f} | Mom: {momentum:.4f} | EMAf: {ema_f:.2f} | EMAs: {ema_s:.2f} | StochK: {stoch_k:.1f} | Regime: {market_regime} | Volatility: {volatility_state} | ForecastΔ: {forecast_delta:+.4f} | Forecast: {forecast_price:.2f} | Signals: {sig_text} | Open: {open_count} | Bal: ${virtual_balance:.4f}")
    except Exception as e:
        logger.debug(f"print_main_status error: {e}")

# ---------------------------
# Main
# ---------------------------
async def main():
    async_client = None
    try:
        if AUTH:
            async_client = await AsyncClient.create(API_KEY, API_SECRET)
        else:
            async_client = await AsyncClient.create()

        await set_leverage(async_client)
        await process_trade_stream(async_client)

    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
    finally:
        # Run final analysis
        analyze_trades()
        
        # Clean up processing task
        if processing_task:
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
                
        if async_client:
            try:
                if hasattr(async_client, "close_connection"):
                    await async_client.close_connection()
            except Exception:
                pass
            try:
                if hasattr(async_client, "close"):
                    await async_client.close()
            except Exception:
                pass
        logger.info("Client closed. Exiting.")

async def set_leverage(async_client):
    """Set leverage for the symbol (async)."""
    if not AUTH or async_client is None:
        logger.info("Skipping leverage setting (not authenticated or client missing).")
        return
    try:
        await async_client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        logger.info(f"Leverage set to {LEVERAGE}x for {SYMBOL}")
    except Exception as e:
        logger.error(f"Failed to set leverage: {e}")

# Entrypoint
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")