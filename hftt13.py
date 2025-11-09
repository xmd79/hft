import asyncio
import os
import sys
import logging
import time
import math
import json
import csv
import traceback
import signal
import statistics
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from binance import AsyncClient, BinanceSocketManager
from binance.client import Client
from binance.exceptions import BinanceWebsocketQueueOverflow
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# For health check server
from aiohttp import web

# ---------------------------
# Logging configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------------------
# Hurst Exponent Calculation
# ---------------------------
try:
    from hurst import compute_Hc
    HURST_AVAILABLE = True
    logger.info("Hurst module imported successfully")
except ImportError:
    HURST_AVAILABLE = False
    logger.warning("Hurst module not available. Install with: pip install hurst")

def compute_hurst(series, window_size=200, preferred_kind='random_walk'):
    """
    Compute Hurst exponent for a time series window.
    
    Parameters:
    - series: numpy array, input time series
    - window_size: int, size of the window to analyze
    - preferred_kind: str, preferred kind for compute_Hc ('price', 'random_walk', 'change')
    
    Returns:
    - H: float, Hurst exponent or None if computation fails
    """
    if not HURST_AVAILABLE:
        logger.debug("Hurst module not available")
        return None
        
    if len(series) < window_size:
        logger.debug(f"Series length {len(series)} is less than window_size {window_size}")
        return None

    window = series[-window_size:]  # Take the most recent window

    # Check if window is valid
    if np.std(window) < 1e-10 or np.max(window) == np.min(window):
        logger.debug("Window has zero range or standard deviation")
        return None

    if np.any(np.isnan(window)) or np.any(np.isinf(window)):
        logger.debug("Window contains NaN or Inf values")
        return None

    kinds = [preferred_kind] + [k for k in ['price', 'random_walk', 'change'] if k != preferred_kind]

    for kind in kinds:
        try:
            adjusted_window = window
            if kind == 'price' and np.any(window <= 0):
                adjusted_window = window - np.min(window) + 1e-10

            H, c, data = compute_Hc(adjusted_window, kind=kind, simplified=True)
            logger.debug(f"Successfully computed Hurst exponent: {H:.3f} with kind='{kind}'")
            return H
        except Exception as e:
            logger.debug(f"Error computing Hurst with kind='{kind}': {e}")
            continue

    logger.debug("Failed to compute Hurst exponent with all kinds")
    return None

def hurst_cycle_analysis(series, window_size=200, sampling_rate=20, preferred_kind='random_walk'):
    """
    Perform Hurst cycle analysis on a time series, computing Hurst exponents and detecting the most extreme peak/trough per window.

    Parameters:
    - series: numpy array, input time series
    - window_size: int, size of the sliding window
    - sampling_rate: int, step size for sliding window
    - preferred_kind: str, preferred kind for compute_Hc ('price', 'random_walk', 'change')

    Returns:
    - cycles: list of Hurst exponents for each valid window
    - peaks: list of indices where the most extreme peaks occur
    - troughs: list of indices where the most extreme troughs occur
    """
    if len(series) < window_size:
        logger.debug(f"Series length {len(series)} is less than window_size {window_size}")
        return [], [], []

    cycles = []
    peaks = []
    troughs = []
    length = len(series)

    # Define order of kinds to try, prioritizing preferred_kind
    kinds = [preferred_kind] + [k for k in ['price', 'random_walk', 'change'] if k != preferred_kind]

    for start in range(0, length - window_size + 1, sampling_rate):
        window = series[start:start + window_size]

        # Check if window is valid
        if len(window) < window_size:
            logger.debug(f"Skipping window at index {start}: insufficient data points")
            continue

        # Validate window data
        if np.std(window) < 1e-10 or np.max(window) == np.min(window):
            logger.debug(f"Skipping window at index {start}: zero range or standard deviation")
            continue

        if np.any(np.isnan(window)) or np.any(np.isinf(window)):
            logger.debug(f"Skipping window at index {start}: contains NaN or Inf values")
            continue

        # Compute Hurst exponent
        hurst_computed = False
        for kind in kinds:
            try:
                # Adjust window for 'price' kind to ensure positive values
                adjusted_window = window
                if kind == 'price' and np.any(window <= 0):
                    adjusted_window = window - np.min(window) + 1e-10

                H, c, data = compute_Hc(adjusted_window, kind=kind, simplified=True)
                cycles.append(H)
                logger.debug(f"Success at index {start} with kind='{kind}', Hurst={H:.3f}")
                hurst_computed = True
                break
            except Exception as e:
                logger.debug(f"Error computing Hurst in window at index {start} with kind='{kind}': {e}")

        if not hurst_computed:
            logger.debug(f"Failed to compute Hurst for window at index {start}")
            continue

        # Detect the most extreme peak and trough
        max_idx = np.argmax(adjusted_window)
        min_idx = np.argmin(adjusted_window)
        peaks.append(start + max_idx)
        troughs.append(start + min_idx)

    if not cycles:
        logger.debug("No valid windows processed")
        return [], [], []

    return cycles, peaks, troughs

# ---------------------------
# Magic Square Diagonal Reversal Forecasting
# ---------------------------
class MagicSquareForecaster:
    """
    Utilizes 4x4 magic square diagonal reversals to forecast cyclic patterns.
    Implements state transitions and pattern detection for market forecasting.
    """
    
    def __init__(self):
        # Define the 4x4 magic square states
        self.state_A = np.array([
            [16, 2, 3, 13],
            [5, 11, 10, 8],
            [9, 7, 6, 12],
            [4, 14, 15, 1]
        ])
        
        # State B: Main diagonal reversed
        self.state_B = np.array([
            [1, 2, 3, 13],
            [5, 6, 10, 8],
            [9, 7, 11, 12],
            [16, 14, 15, 4]
        ])
        
        # State C: Anti-diagonal reversed on State B
        self.state_C = np.array([
            [4, 2, 3, 16],
            [5, 7, 10, 8],
            [9, 6, 11, 12],
            [13, 14, 15, 1]
        ])
        
        # State D: Main diagonal reversed on State C
        self.state_D = np.array([
            [1, 2, 3, 16],
            [5, 11, 10, 8],
            [9, 7, 6, 12],
            [4, 14, 15, 13]
        ])
        
        # State transition cycle: A -> B -> C -> D -> A
        self.states = [self.state_A, self.state_B, self.state_C, self.state_D]
        self.state_names = ['A', 'B', 'C', 'D']
        
        # Cache for recent forecasts
        self.forecast_cache = deque(maxlen=10)
        
    def get_rank_grid(self, prices):
        """Convert price array to rank grid (1-16)"""
        if len(prices) < 16:
            return None
            
        # Take last 16 prices
        last_16 = np.array(prices[-16:])
        
        # Create rank grid (1 for smallest, 16 for largest)
        temp = last_16.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(last_16))
        rank_grid = (ranks + 1).reshape(4, 4)  # Convert to 1-16 ranking
        
        return rank_grid
    
    def detect_current_state(self, rank_grid):
        """Detect current magic square state by comparing rank patterns"""
        if rank_grid is None:
            return None, float('inf')
            
        min_distance = float('inf')
        best_state = None
        best_state_idx = -1
        
        for i, state in enumerate(self.states):
            # Calculate sum of squared differences
            diff = rank_grid - state
            distance = np.sum(diff ** 2)
            
            if distance < min_distance:
                min_distance = distance
                best_state = state
                best_state_idx = i
                
        return best_state_idx, min_distance
    
    def get_next_state(self, current_state_idx):
        """Get next state in the cycle"""
        return (current_state_idx + 1) % 4
    
    def forecast_pattern(self, prices):
        """
        Forecast next pattern using magic square diagonal reversals.
        Returns: (signal, confidence, next_state_name)
        """
        if len(prices) < 16:
            return 0, 0.0, "insufficient_data"
            
        # Get rank grid for recent prices
        rank_grid = self.get_rank_grid(prices)
        if rank_grid is None:
            return 0, 0.0, "rank_error"
            
        # Detect current state
        current_state_idx, distance = self.detect_current_state(rank_grid)
        if current_state_idx is None:
            return 0, 0.0, "detection_failed"
            
        # Get next state in cycle
        next_state_idx = self.get_next_state(current_state_idx)
        next_state = self.states[next_state_idx]
        next_state_name = self.state_names[next_state_idx]
        
        # Calculate trend signal based on next state
        # Compare first row (earlier) vs last row (later) in the grid
        first_row_avg = np.mean(next_state[0, :])
        last_row_avg = np.mean(next_state[3, :])
        
        # Signal: 1 for bullish, -1 for bearish
        signal = 1 if last_row_avg > first_row_avg else -1 if last_row_avg < first_row_avg else 0
        
        # Calculate confidence based on distance (lower distance = higher confidence)
        max_possible_distance = 4 * 4 * (16**2)  # Worst case scenario
        confidence = max(0.0, 1.0 - (distance / max_possible_distance))
        
        # Cache forecast
        self.forecast_cache.append({
            'timestamp': time.time(),
            'signal': signal,
            'confidence': confidence,
            'state': next_state_name,
            'distance': distance
        })
        
        return signal, confidence, next_state_name
    
    def get_forecast_trend(self):
        """Get recent forecast trend for additional confirmation"""
        if len(self.forecast_cache) < 3:
            return 0  # Not enough data
            
        # Get last 3 forecasts
        recent = list(self.forecast_cache)[-3:]
        
        # Count bullish vs bearish signals
        bullish_count = sum(1 for f in recent if f['signal'] == 1)
        bearish_count = sum(1 for f in recent if f['signal'] == -1)
        
        # Determine trend
        if bullish_count > bearish_count:
            return 1  # Bullish trend
        elif bearish_count > bullish_count:
            return -1  # Bearish trend
        else:
            return 0  # Neutral

# Initialize Magic Square Forecaster
magic_forecaster = MagicSquareForecaster()

# ---------------------------
# Configuration (tune me)
# ---------------------------
SYMBOL = 'BTCUSDC'                 # Target perpetual futures symbol
TIMEFRAME_SECONDS = 15             # Pseudo-candle length in seconds
LEVERAGE = 125                     # Leverage to set
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
MIN_BALANCE_THRESHOLD = 0.5        # Minimum balance to allow trading
TRADE_COOLDOWN_SECONDS = 60        # Cooldown period between same pattern trades

# Backtesting parameters
BACKTEST_MODE = False              # Set to True to run in backtest mode
BACKTEST_DAYS = 30                 # Number of days to backtest

# Market regime parameters
ADX_PERIOD = 14                    # Period for ADX calculation
TREND_THRESHOLD = 25               # ADX value above which market is considered trending

# Position management parameters
ATR_PERIOD = 14                    # Period for ATR calculation
STOP_LOSS_ATR_MULTIPLIER = 1.5     # Stop loss distance in ATR units
TAKE_PROFIT_ATR_MULTIPLIER = 3.0   # Take profit distance in ATR units
TRAILING_STOP_ATR_MULTIPLIER = 1.0 # Trailing stop distance in ATR units

# Fee adjustment
FEE_RATE = 0.0004                 # 0.04% fee per side (0.08% round trip)

# Linear Regression Channel parameters
LRC_PERIOD = 200                   # Period for Linear Regression Channel
LRC_DEVIATION_MULTIPLIER = 2.0     # Deviation multiplier for channel width

# Rounding defaults
DEFAULT_QUANTITY_STEP = 0.000001
DEFAULT_PRICE_TICK = 0.01

# WebSocket configuration
WS_QUEUE_SIZE = 2000               # Internal queue size for processing
WS_RECONNECT_DELAY = 10            # Seconds to wait before reconnecting
WS_MAX_RECONNECT_ATTEMPTS = 20     # Maximum reconnection attempts
WS_PROCESSING_TIMEOUT = 0.001      # Timeout for message processing (seconds)
WS_HEARTBEAT_INTERVAL = 30        # Seconds between heartbeat checks

# Debugging options
DEBUG_MODE = True                  # Enable detailed debug logging

# Hurst Exponent parameters
HURST_WINDOW_SIZE = 200            # Window size for Hurst calculation
HURST_UPDATE_FREQUENCY = 20        # Update Hurst every N candles
USE_HURST = True                  # Enable/disable Hurst analysis

# Extreme Reversal Detection parameters
EXTREME_REVERSAL_WINDOW = 200      # Window size for extreme reversal detection
EXTREME_REVERSAL_THRESHOLD = 0.02  # Threshold for extreme reversal (2%)
EXTREME_REVERSAL_UPDATE_FREQ = 20 # Update extreme reversals every N candles

# Magic Square Forecasting parameters
USE_MAGIC_SQUARE = True           # Enable/disable magic square forecasting
MAGIC_SQUARE_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for magic square signals

# ---------------------------
# Global runtime state
# ---------------------------
open_trades = []                   # List of open trades
current_balance = INITIAL_BALANCE   # Track current balance (real or virtual)
candle_data = []                   # List of pseudo-candle dicts
symbol_info_cache = {}             # Cache symbol exchange_info
AUTH = False                       # Whether API keys were provided
market_regime = "ranging"          # Current market regime (trending/ranging)
volatility_state = "normal"        # Current volatility state (low/normal/high)
last_pattern_time = {}             # Track last pattern execution time for cooldown
bot_start_time = time.time()       # Track when the bot started

# For adaptive thresholds
volatility_history = deque(maxlen=50)  # Store recent volatility measurements
atr_values = deque(maxlen=ATR_PERIOD)  # Store ATR values for calculation

# For market regime detection
adx_values = deque(maxlen=ADX_PERIOD)  # Store ADX values
plus_di_values = deque(maxlen=ADX_PERIOD)
minus_di_values = deque(maxlen=ADX_PERIOD)

# For arctanh reversal detection
price_history = deque(maxlen=200)  # Store last 200 price points
last_argmin_index = -1             # Index of last argmin in price_history
last_argmax_index = -1             # Index of last argmax in price_history

# For Linear Regression Channel
lrc_prices = deque(maxlen=LRC_PERIOD)  # Store prices for LRC calculation
lrc_middle = None                  # Current LRC middle line value
lrc_upper = None                   # Current LRC upper line value
lrc_lower = None                   # Current LRC lower line value
lrc_signal = None                  # Current LRC signal (long/short)

# For Hurst Exponent
current_hurst = None               # Current Hurst exponent value
hurst_update_counter = 0           # Counter for Hurst updates

# For Extreme Reversal Detection
extreme_peak_prices = []           # List of detected extreme peak prices
extreme_trough_prices = []         # List of detected extreme trough prices
extreme_peak_indices = []          # List of detected extreme peak indices
extreme_trough_indices = []        # List of detected extreme trough indices
extreme_reversal_counter = 0       # Counter for extreme reversal updates

# Message processing optimization
message_queue = asyncio.Queue(maxsize=WS_QUEUE_SIZE)  # Queue for processing messages
processing_task = None  # Reference to the processing task
last_processed_time = time.time()  # Track last processing time
last_heartbeat_time = time.time()  # Track last heartbeat time
processing_stats = {
    'messages_received': 0,
    'messages_processed': 0,
    'overflows': 0,
    'processing_time': 0
}  # Track processing statistics

# Trade analysis variables
trade_count = 0

# WebSocket management
ws_reconnect_attempts = 0
ws_connected = False
ws_manager = None
ws_socket = None
ws_active = False

# Balance updater task
balance_updater_task = None

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
# Balance Management Functions
# ---------------------------
async def get_real_balance(async_client):
    """Fetch real USDC balance from Binance Futures account."""
    if not AUTH or async_client is None:
        return None
    try:
        account_info = await async_client.futures_account()
        for asset in account_info.get('assets', []):
            if asset.get('asset') == 'USDC':
                return float(asset.get('availableBalance', 0.0))
    except Exception as e:
        logger.error(f"Failed to fetch real balance: {e}")
    return None

async def update_current_balance(async_client):
    """Update current balance with real balance if authenticated."""
    global current_balance
    if AUTH and async_client:
        real_balance = await get_real_balance(async_client)
        if real_balance is not None:
            current_balance = real_balance
            logger.debug(f"Updated current_balance to {current_balance} (real)")
        else:
            logger.warning("Failed to fetch real balance, keeping current_balance")

async def balance_updater(async_client):
    """Periodically update current balance with real balance."""
    while True:
        try:
            await asyncio.sleep(60)  # Update every 60 seconds
            await update_current_balance(async_client)
        except asyncio.CancelledError:
            logger.info("Balance updater task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in balance updater: {e}")
            await asyncio.sleep(10)  # Wait before retrying

# ---------------------------
# Linear Regression Channel Calculation
# ---------------------------
def calculate_lrc(prices, period=LRC_PERIOD, deviation_multiplier=LRC_DEVIATION_MULTIPLIER):
    """
    Calculate Linear Regression Channel (LRC) for a given price series.
    Returns: (middle_line, upper_line, lower_line)
    """
    if len(prices) < period:
        return None, None, None
    
    # Use logarithmic scale for better results with trending markets
    log_prices = np.log(prices)
    
    # Create x values (time points)
    x = np.arange(period)
    
    # Calculate linear regression (y = mx + b)
    slope, intercept = np.polyfit(x, log_prices, 1)
    
    # Calculate the regression line values
    regression_line = slope * x + intercept
    
    # Calculate standard deviation
    std_dev = np.std(log_prices - regression_line)
    
    # Calculate upper and lower channel lines
    upper_line = regression_line + (deviation_multiplier * std_dev)
    lower_line = regression_line - (deviation_multiplier * std_dev)
    
    # Convert back from log scale
    middle_line = np.exp(regression_line)
    upper_line = np.exp(upper_line)
    lower_line = np.exp(lower_line)
    
    return middle_line[-1], upper_line[-1], lower_line[-1]

def update_lrc(current_price):
    """
    Update Linear Regression Channel values and signal.
    Returns: (middle, upper, lower, signal)
    """
    global lrc_prices, lrc_middle, lrc_upper, lrc_lower, lrc_signal
    
    # Add current price to history
    lrc_prices.append(current_price)
    
    # Calculate LRC if we have enough data
    if len(lrc_prices) >= LRC_PERIOD:
        lrc_middle, lrc_upper, lrc_lower = calculate_lrc(list(lrc_prices), LRC_PERIOD, LRC_DEVIATION_MULTIPLIER)
        
        # Determine signal based on current price position relative to channel
        if current_price < lrc_lower:
            lrc_signal = "long"
        elif current_price > lrc_upper:
            lrc_signal = "short"
        else:
            # Price is within the channel, maintain previous signal or set to neutral
            # We'll keep the previous signal to avoid flipping too frequently
            pass
    
    return lrc_middle, lrc_upper, lrc_lower, lrc_signal

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
# FFT Cycle Detection
# ---------------------------
def fft_cycle_detection(prices):
    """
    Detect dominant cycles in price data using FFT.
    Returns dominant period and phase.
    """
    if len(prices) < 50:  # Need enough data for meaningful FFT
        return None, None
    
    # Convert to numpy array
    prices = np.array(prices)
    
    # Detrend the prices by subtracting a linear fit
    x = np.arange(len(prices))
    y = prices
    coef = np.polyfit(x, y, 1)
    trend = np.polyval(coef, x)
    detrended = y - trend
    
    # Apply FFT
    fft_result = fft(detrended)
    fft_freq = fftfreq(len(detrended))
    
    # Get the amplitudes (ignore negative frequencies and zero frequency)
    amplitudes = np.abs(fft_result)
    positive_freq_idx = np.where(fft_freq > 0)[0]
    
    if len(positive_freq_idx) == 0:
        return None, None
    
    # Find the frequency with the maximum amplitude
    max_amp_idx = positive_freq_idx[np.argmax(amplitudes[positive_freq_idx])]
    dominant_freq = fft_freq[max_amp_idx]
    dominant_period = 1 / dominant_freq if dominant_freq != 0 else None
    
    # Get the phase at the dominant frequency
    phase = np.angle(fft_result[max_amp_idx])
    
    return dominant_period, phase

# ---------------------------
# Arctanh Reversal Detection
# ---------------------------
def update_arctanh_reversals(current_price):
    """
    Update arctanh reversal detection logic.
    Returns reversal signal if detected.
    """
    global price_history, last_argmin_index, last_argmax_index
    
    # Add current price to history
    price_history.append(current_price)
    
    # Need at least 50 points for meaningful analysis
    if len(price_history) < 50:
        return None
    
    # Find argmin and argmax in the price history
    prices_array = np.array(price_history)
    argmin_index = np.argmin(prices_array)
    argmax_index = np.argmax(prices_array)
    
    # Check if we have new extremes
    new_argmin = (argmin_index != last_argmin_index)
    new_argmax = (argmax_index != last_argmax_index)
    
    # Update last known indices
    if new_argmin:
        last_argmin_index = argmin_index
    if new_argmax:
        last_argmax_index = argmax_index
    
    # Determine which extreme is more recent
    if argmin_index > argmax_index:
        # Most recent extreme is a minimum (dip)
        # This suggests a long opportunity
        return "long"
    elif argmax_index > argmin_index:
        # Most recent extreme is a maximum (peak)
        # This suggests a short opportunity
        return "short"
    
    return None

# ---------------------------
# Extreme Reversal Detection with Hurst Cycle Analysis
# ---------------------------
def update_hurst_and_extremes(current_price):
    """
    Update Hurst exponent and extreme reversal detection using Hurst cycle analysis.
    Returns reversal signal if detected.
    """
    global current_hurst, extreme_peak_prices, extreme_trough_prices, extreme_peak_indices, extreme_trough_indices
    
    # Add current price to price history
    price_history.append(current_price)
    
    # Need at least EXTREME_REVERSAL_WINDOW points for meaningful analysis
    if len(price_history) < EXTREME_REVERSAL_WINDOW:
        return None
    
    # Convert price history to numpy array
    prices_array = np.array(price_history)
    
    # Run Hurst cycle analysis
    try:
        cycles, peaks, troughs = hurst_cycle_analysis(
            prices_array, 
            window_size=EXTREME_REVERSAL_WINDOW,
            sampling_rate=10,  # Smaller sampling rate for more frequent updates
            preferred_kind='random_walk'
        )
        
        if not cycles:
            logger.debug("No Hurst cycles computed")
            return None
        
        # Update current Hurst exponent with the most recent value
        current_hurst = cycles[-1]
        logger.debug(f"Updated Hurst exponent: {current_hurst:.3f}")
        
        # Get the most recent peak and trough
        if peaks:
            last_peak_idx = peaks[-1]
            if last_peak_idx < len(prices_array):
                extreme_peak_prices.append(prices_array[last_peak_idx])
                extreme_peak_indices.append(last_peak_idx)
        
        if troughs:
            last_trough_idx = troughs[-1]
            if last_trough_idx < len(prices_array):
                extreme_trough_prices.append(prices_array[last_trough_idx])
                extreme_trough_indices.append(last_trough_idx)
        
        # Keep only the last 5 extremes to avoid memory issues
        if len(extreme_peak_prices) > 5:
            extreme_peak_prices = extreme_peak_prices[-5:]
            extreme_peak_indices = extreme_peak_indices[-5:]
        
        if len(extreme_trough_prices) > 5:
            extreme_trough_prices = extreme_trough_prices[-5:]
            extreme_trough_indices = extreme_trough_indices[-5:]
        
        # Check if current price is near an extreme
        current_price = prices_array[-1]
        
        # Check for peak (short opportunity)
        if extreme_peak_prices:
            last_peak = extreme_peak_prices[-1]
            if abs(current_price - last_peak) / last_peak < EXTREME_REVERSAL_THRESHOLD:
                logger.debug(f"Current price {current_price} near peak {last_peak}")
                return "short"
        
        # Check for trough (long opportunity)
        if extreme_trough_prices:
            last_trough = extreme_trough_prices[-1]
            if abs(current_price - last_trough) / last_trough < EXTREME_REVERSAL_THRESHOLD:
                logger.debug(f"Current price {current_price} near trough {last_trough}")
                return "long"
        
        return None
    
    except Exception as e:
        logger.error(f"Error in Hurst and extreme reversal detection: {e}")
        return None

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
    
    # Calculate Linear Regression Channel
    if len(df) >= LRC_PERIOD:
        prices = df['close'].values[-LRC_PERIOD:]
        middle, upper, lower = calculate_lrc(prices, LRC_PERIOD, LRC_DEVIATION_MULTIPLIER)
        df['lrc_middle'] = middle
        df['lrc_upper'] = upper
        df['lrc_lower'] = lower
    else:
        df['lrc_middle'] = np.nan
        df['lrc_upper'] = np.nan
        df['lrc_lower'] = np.nan
    
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

def detect_patterns_with_intensity(df, current_price):
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

    # Adjust pattern detection based on market regime and Hurst exponent
    if market_regime.startswith("trending"):
        # In trending markets, we want to be more selective with reversals
        stoch_threshold = 15 if market_regime.endswith("bullish") else 85
    else:
        # In ranging markets, standard thresholds
        stoch_threshold = 20 if market_regime.endswith("bullish") else 80
    
    # Further adjust based on Hurst exponent if available
    if USE_HURST and current_hurst is not None:
        if current_hurst > 0.5:  # Trending market
            # Make it harder to trigger reversals in trending markets
            stoch_threshold = 10 if market_regime.endswith("bullish") else 90
        else:  # Mean-reverting market
            # Make it easier to trigger reversals
            stoch_threshold = 30 if market_regime.endswith("bullish") else 70

    # Check for arctanh reversal signal
    arctanh_signal = update_arctanh_reversals(current_price)
    
    # Check for extreme reversal signal
    extreme_signal = update_hurst_and_extremes(current_price)
    
    # Check for FFT cycle confirmation
    fft_period, fft_phase = fft_cycle_detection(price_history)
    fft_confirms_long = False
    fft_confirms_short = False
    
    if fft_period is not None and fft_phase is not None:
        # Normalize phase to [0, 2π]
        normalized_phase = (fft_phase + np.pi) % (2 * np.pi)
        
        # If phase is near 0 or 2π, it suggests a peak (short opportunity)
        # If phase is near π, it suggests a trough (long opportunity)
        if normalized_phase < np.pi/4 or normalized_phase > 7*np.pi/4:
            fft_confirms_short = True
        elif np.pi/2 < normalized_phase < 3*np.pi/2:
            fft_confirms_long = True

    # Check for Linear Regression Channel signal
    lrc_signal = update_lrc(current_price)
    
    # Check for Magic Square signal
    magic_signal = 0
    magic_confidence = 0.0
    magic_state = "none"
    if USE_MAGIC_SQUARE and len(price_history) >= 16:
        magic_signal, magic_confidence, magic_state = magic_forecaster.forecast_pattern(list(price_history))
        # Only use if confidence is above threshold
        if magic_confidence < MAGIC_SQUARE_CONFIDENCE_THRESHOLD:
            magic_signal = 0

    # DBD
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] < df['open'].iloc[i-3] and
            df['close'].iloc[i-1] < df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] > stoch_threshold and
            (arctanh_signal == "short" or extreme_signal == "short" or fft_confirms_short or lrc_signal == "short" or magic_signal == -1)):
        intensity = pattern_intensity(df, i)
        # Adjust intensity based on Hurst exponent
        if USE_HURST and current_hurst is not None:
            if current_hurst > 0.5:  # Trending market
                intensity *= 1.2  # Boost intensity for trend patterns
            else:  # Mean-reverting market
                intensity *= 0.8  # Reduce intensity for reversals
        signals.append(('sell', float(df['close'].iloc[i]), int(intensity * 100), 'DBD'))

    # RBD
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] > df['open'].iloc[i-3] and
            df['close'].iloc[i-1] < df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] > stoch_threshold and
            (arctanh_signal == "short" or extreme_signal == "short" or fft_confirms_short or lrc_signal == "short" or magic_signal == -1)):
        intensity = pattern_intensity(df, i)
        # Adjust intensity based on Hurst exponent
        if USE_HURST and current_hurst is not None:
            if current_hurst > 0.5:  # Trending market
                intensity *= 1.2  # Boost intensity for trend patterns
            else:  # Mean-reverting market
                intensity *= 0.8  # Reduce intensity for reversals
        signals.append(('sell', float(df['close'].iloc[i]), int(intensity * 100), 'RBD'))

    # DBR
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] < df['open'].iloc[i-3] and
            df['close'].iloc[i-1] > df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] < (100 - stoch_threshold) and
            (arctanh_signal == "long" or extreme_signal == "long" or fft_confirms_long or lrc_signal == "long" or magic_signal == 1)):
        intensity = pattern_intensity(df, i)
        # Adjust intensity based on Hurst exponent
        if USE_HURST and current_hurst is not None:
            if current_hurst > 0.5:  # Trending market
                intensity *= 1.2  # Boost intensity for trend patterns
            else:  # Mean-reverting market
                intensity *= 0.8  # Reduce intensity for reversals
        signals.append(('buy', float(df['close'].iloc[i]), int(intensity * 100), 'DBR'))

    # RBR
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] > df['open'].iloc[i-3] and
            df['close'].iloc[i-1] > df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] < (100 - stoch_threshold) and
            (arctanh_signal == "long" or extreme_signal == "long" or fft_confirms_long or lrc_signal == "long" or magic_signal == 1)):
        intensity = pattern_intensity(df, i)
        # Adjust intensity based on Hurst exponent
        if USE_HURST and current_hurst is not None:
            if current_hurst > 0.5:  # Trending market
                intensity *= 1.2  # Boost intensity for trend patterns
            else:  # Mean-reverting market
                intensity *= 0.8  # Reduce intensity for reversals
        signals.append(('buy', float(df['close'].iloc[i]), int(intensity * 100), 'RBR'))

    # Spike Buy (downward spike rejection)
    if (df['spike'].iloc[i-1] == -1 and
            df['open'].iloc[i] > df['low'].iloc[i-1] and
            df['volume'].iloc[i-1] > df['volume_mean'].iloc[i-1] * VOLUME_THRESHOLD and
            abs(df['momentum'].iloc[i-1]) > MOMENTUM_THRESHOLD and
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] < (100 - stoch_threshold) and
            (arctanh_signal == "long" or extreme_signal == "long" or fft_confirms_long or lrc_signal == "long" or magic_signal == 1)):
        intensity = pattern_intensity(df, i)
        # bump intensity for spike-specific criteria
        intensity = min(1.0, intensity + 0.15)
        # Adjust intensity based on Hurst exponent
        if USE_HURST and current_hurst is not None:
            if current_hurst < 0.5:  # Mean-reverting market
                intensity *= 1.3  # Boost intensity for reversals
            else:  # Trending market
                intensity *= 0.9  # Reduce intensity for reversals
        signals.append(('buy', float(df['close'].iloc[i]), int(intensity * 100), 'SpikeBuy'))

    # Spike Sell (upward spike rejection)
    if (df['spike'].iloc[i-1] == 1 and
            df['open'].iloc[i] < df['high'].iloc[i-1] and
            df['volume'].iloc[i-1] > df['volume_mean'].iloc[i-1] * VOLUME_THRESHOLD and
            abs(df['momentum'].iloc[i-1]) > MOMENTUM_THRESHOLD and
            df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] > stoch_threshold and
            (arctanh_signal == "short" or extreme_signal == "short" or fft_confirms_short or lrc_signal == "short" or magic_signal == -1)):
        intensity = pattern_intensity(df, i)
        intensity = min(1.0, intensity + 0.15)
        # Adjust intensity based on Hurst exponent
        if USE_HURST and current_hurst is not None:
            if current_hurst < 0.5:  # Mean-reverting market
                intensity *= 1.3  # Boost intensity for reversals
            else:  # Trending market
                intensity *= 0.9  # Reduce intensity for reversals
        signals.append(('sell', float(df['close'].iloc[i]), int(intensity * 100), 'SpikeSell'))

    # Extra quick spike predictor: large sudden momentum on last tick
    last_mom = abs(df['momentum'].iloc[-1])
    if last_mom > MOMENTUM_THRESHOLD * 3:
        direction = 'buy' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'sell'
        intensity = min(100, int(score_between(last_mom, MOMENTUM_THRESHOLD*3, MOMENTUM_THRESHOLD*8) * 100 + 20))
        
        # Adjust intensity based on Hurst exponent
        if USE_HURST and current_hurst is not None:
            if current_hurst < 0.5:  # Mean-reverting market
                intensity = min(100, int(intensity * 1.2))
            else:  # Trending market
                intensity = min(100, int(intensity * 0.8))
        
        # Only add if confirmed by arctanh, extreme, FFT, LRC, or Magic Square
        if (direction == 'buy' and (arctanh_signal == "long" or extreme_signal == "long" or fft_confirms_long or lrc_signal == "long" or magic_signal == 1)) or \
           (direction == 'sell' and (arctanh_signal == "short" or extreme_signal == "short" or fft_confirms_short or lrc_signal == "short" or magic_signal == -1)):
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
    Uses ATR for dynamic stop-loss and take-profit levels, adjusted for fees and Hurst exponent.
    """
    global current_balance

    # Check if balance is too low to trade
    if current_balance < MIN_BALANCE_THRESHOLD:
        return 0.0, 0.0, 0.0, False

    risk_amount = current_balance * PNL_PERCENT
    
    # Calculate ATR for dynamic position sizing
    atr = calculate_atr(df) if not df.empty else 0.0
    if atr == 0.0:
        atr = BASE_SPIKE_THRESHOLD / 2.0  # Fallback to fixed value
    
    # Adjust ATR multipliers based on Hurst exponent
    stop_multiplier = STOP_LOSS_ATR_MULTIPLIER
    take_multiplier = TAKE_PROFIT_ATR_MULTIPLIER
    
    if USE_HURST and current_hurst is not None:
        if current_hurst > 0.5:  # Trending market
            # Wider stops and targets for trends
            stop_multiplier *= 1.2
            take_multiplier *= 1.5
        else:  # Mean-reverting market
            # Tighter stops and targets for reversals
            stop_multiplier *= 0.8
            take_multiplier *= 0.7
    
    # Use ATR for stop distance
    stop_distance = atr * stop_multiplier
    take_profit_distance = atr * take_multiplier
    
    # Adjust for fees
    fee_adjusted_stop_distance = stop_distance * (1 + FEE_RATE)
    fee_adjusted_take_profit_distance = take_profit_distance * (1 + FEE_RATE)
    
    raw_quantity = risk_amount / (fee_adjusted_stop_distance * PIP_VALUE) if fee_adjusted_stop_distance > 0 else 0.0
    margin_required = (raw_quantity * entry_price) / LEVERAGE if entry_price > 0 else float('inf')

    stepSize = DEFAULT_QUANTITY_STEP
    minQty = MIN_ORDER_SIZE
    try:
        info = await fetch_symbol_info(async_client, SYMBOL)
        stepSize = info.get('stepSize', DEFAULT_QUANTITY_STEP)
        minQty = info.get('minQty', MIN_ORDER_SIZE)
    except Exception:
        pass

    max_q_by_margin = (current_balance * LEVERAGE) / entry_price if entry_price > 0 else 0.0
    max_q_by_margin *= 0.995

    quantity = min(raw_quantity, max_q_by_margin)
    quantity = floor_to_step(quantity, stepSize)

    if quantity < minQty or quantity < MIN_ORDER_SIZE:
        can_trade = False
    else:
        can_trade = True if AUTH and (margin_required <= current_balance) else False
        margin_required = (quantity * entry_price) / LEVERAGE if entry_price > 0 else float('inf')
        if AUTH and margin_required > current_balance:
            can_trade = False

    # Set dynamic stop-loss and take-profit based on ATR, adjusted for fees
    if action == 'buy':
        stop_loss = entry_price - fee_adjusted_stop_distance
        take_profit = entry_price + fee_adjusted_take_profit_distance
    else:
        stop_loss = entry_price + fee_adjusted_stop_distance
        take_profit = entry_price - fee_adjusted_take_profit_distance

    quantity = float(round(quantity, 8))
    return quantity, stop_loss, take_profit, can_trade

async def execute_trade(async_client, action, entry_price, quantity, stop_loss, take_profit, can_trade, df: pd.DataFrame, pattern_name='', intensity=0):
    """
    Execute a real trade (async) or simulate it. Adds trade to open_trades.
    Implements trailing stop for trending markets.
    """
    global current_balance, open_trades, trade_count, last_pattern_time
    
    # Check if quantity is 0 and skip if so
    if quantity <= 0:
        logger.info(f"Skipping trade with 0 quantity: {action.upper()} {SYMBOL} at ${entry_price:.2f}")
        return
    
    # Check cooldown period for this pattern
    current_time = time.time()
    if pattern_name in last_pattern_time:
        time_since_last = current_time - last_pattern_time[pattern_name]
        if time_since_last < TRADE_COOLDOWN_SECONDS:
            logger.debug(f"Skipping {pattern_name} due to cooldown ({time_since_last:.1f}s < {TRADE_COOLDOWN_SECONDS}s)")
            return
    
    # Update last pattern time
    last_pattern_time[pattern_name] = current_time
    
    # Determine if we should use trailing stop
    use_trailing_stop = market_regime.startswith("trending")
    if USE_HURST and current_hurst is not None and current_hurst > 0.5:
        use_trailing_stop = True  # Always use trailing stop in trending markets (H > 0.5)
    
    trailing_distance = calculate_atr(df) * TRAILING_STOP_ATR_MULTIPLIER if not df.empty else 0.0
    
    # Calculate margin required for this trade
    margin_required = (quantity * entry_price) / LEVERAGE if entry_price > 0 else 0.0
    
    # Update current balance for simulated trades
    if not can_trade:
        current_balance -= margin_required
    
    trade_info = {
        'action': action,
        'entry_price': entry_price,
        'quantity': quantity,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'is_simulated': not can_trade,
        'entry_time': datetime.now().isoformat(),
        'use_trailing_stop': use_trailing_stop,
        'trailing_distance': trailing_distance,
        'highest_price': entry_price if action == 'buy' else 0.0,
        'lowest_price': entry_price if action == 'sell' else float('inf'),
        'market_regime': market_regime,
        'volatility_state': volatility_state,
        'pattern': pattern_name,
        'intensity': intensity,
        'margin_used': margin_required,
        'hurst_exponent': current_hurst if USE_HURST else None
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
            
            logger.info(f"Real trade placed: {action.upper()} {quantity} {SYMBOL} at {entry_price:.2f} | SL {stop_loss:.2f} TP {take_profit:.2f} | margin used: ${margin_required:.4f} | Balance: ${current_balance:.4f}")
        except Exception as e:
            logger.error(f"Real trade execution failed: {e}")
            trade_info['is_simulated'] = True
    else:
        reason = "not authorized" if not AUTH else "insufficient margin or size"
        logger.info(f"Simulating trade due to {reason}: {action.upper()} {quantity} {SYMBOL} at ${entry_price:.2f}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f} | Balance: ${current_balance:.4f}")

    open_trades.append(trade_info)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ENTRY -> {action.upper():4} | Qty {quantity:.6f} | Price {entry_price:.2f} | SL {stop_loss:.2f} | TP {take_profit:.2f} | Simulated: {trade_info['is_simulated']} | Pattern: {pattern_name} | Balance: ${current_balance:.4f}")
    
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
        'timestamp': datetime.now().isoformat(),
        'market_regime': market_regime,
        'volatility_state': volatility_state,
        'balance': current_balance,
        'hurst_exponent': current_hurst if USE_HURST else None
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
    global current_balance, open_trades, trade_count
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
        margin_used = trade.get('margin_used', 0.0)

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
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT  -> BUY closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Balance: ${current_balance + pnl + margin_used:.4f}")
                # Update current balance for simulated trades
                if is_simulated:
                    current_balance += pnl + margin_used
            elif current_price >= take_profit:
                exit_price = take_profit
                pnl = (take_profit - entry_price) * quantity * PIP_VALUE
                closed.append(trade)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT  -> BUY closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Balance: ${current_balance + pnl + margin_used:.4f}")
                # Update current balance for simulated trades
                if is_simulated:
                    current_balance += pnl + margin_used
        else:  # sell
            if current_price >= stop_loss:
                exit_price = stop_loss
                pnl = (entry_price - stop_loss) * quantity * PIP_VALUE
                closed.append(trade)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT  -> SELL closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Balance: ${current_balance + pnl + margin_used:.4f}")
                # Update current balance for simulated trades
                if is_simulated:
                    current_balance += pnl + margin_used
            elif current_price <= take_profit:
                exit_price = take_profit
                pnl = (entry_price - take_profit) * quantity * PIP_VALUE
                closed.append(trade)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT  -> SELL closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Balance: ${current_balance + pnl + margin_used:.4f}")
                # Update current balance for simulated trades
                if is_simulated:
                    current_balance += pnl + margin_used
        
        # Record closed trade
        if exit_price is not None:
            trade['exit_price'] = exit_price
            trade['pnl'] = pnl
            trade['exit_time'] = datetime.now().isoformat()  # Add exit time
            trade_count += 1

    open_trades[:] = [t for t in open_trades if t not in closed]

# ---------------------------
# Backtesting Functionality
# ---------------------------
async def run_backtest(async_client):
    """Run backtest using historical data."""
    global current_balance, candle_data, price_history, lrc_prices, current_hurst
    
    logger.info(f"Starting backtest for the last {BACKTEST_DAYS} days...")
    
    # Reset state
    current_balance = INITIAL_BALANCE
    open_trades.clear()
    candle_data.clear()
    price_history.clear()
    lrc_prices.clear()
    current_hurst = None
    
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
            
            # Update Hurst exponent periodically
            if USE_HURST and len(candle_data) >= HURST_WINDOW_SIZE:
                hurst_update_counter += 1
                if hurst_update_counter >= HURST_UPDATE_FREQUENCY:
                    hurst_update_counter = 0
                    prices = np.array([c['close'] for c in candle_data])
                    current_hurst = compute_hurst(prices, window_size=HURST_WINDOW_SIZE)
                    if current_hurst is not None:
                        logger.debug(f"Backtest Hurst exponent: {current_hurst:.3f}")
            
            # Update extreme reversals periodically
            if len(candle_data) >= EXTREME_REVERSAL_WINDOW:
                extreme_reversal_counter += 1
                if extreme_reversal_counter >= EXTREME_REVERSAL_UPDATE_FREQ:
                    extreme_reversal_counter = 0
                    current_price = float(trade['p'])
                    update_hurst_and_extremes(current_price)
        
        # Detect patterns with intensities
        current_price = float(trade['p'])
        signals = detect_patterns_with_intensity(df, current_price) if not df.empty else []
        
        # Execute signals
        for action, entry_price, intensity_pct, pname in signals:
            quantity, stop_loss, take_profit, can_trade = await calculate_quantity_and_levels(entry_price, action, async_client, df)
            await execute_trade(async_client, action, entry_price, quantity, stop_loss, take_profit, False, df, pname, intensity_pct)
        
        # Track trades on current price
        await track_trades(current_price)
    
    # Calculate performance metrics
    if trade_count > 0:
        logger.info(f"Backtest completed. Total trades: {trade_count}, Final Balance: ${current_balance:.4f}")
    else:
        logger.info("Backtest completed. No trades were executed.")

# ---------------------------
# Enhanced message processing
# ---------------------------
async def process_message(message, async_client):
    """Process a single trade message with error handling."""
    global processing_stats, candle_data, last_heartbeat_time, price_history, hurst_update_counter, current_hurst, extreme_reversal_counter
    processing_stats['messages_processed'] += 1
    start_time = time.time()
    
    try:
        # Check if message is valid
        if not message or ('p' not in message and 'price' not in message):
            return
        
        # Update heartbeat time
        last_heartbeat_time = time.time()
        
        # Get current price
        current_price = float(message.get('p', 0.0))
        
        # Update price history for arctanh reversal detection
        price_history.append(current_price)
        
        # Update LRC with current price
        update_lrc(current_price)
        
        # Aggregate into pseudo-candles
        candle_data = aggregate_candle(candle_data, message)
        
        # Only build df & indicators when we have enough data and periodically
        # to avoid excessive processing
        if len(candle_data) >= 3 and (processing_stats['messages_processed'] % 5 == 0):
            df = pd.DataFrame(candle_data)
            if not df.empty:
                df = calculate_indicators(df)
                update_market_regime(df)
                
                # Update Hurst exponent periodically
                if USE_HURST and len(candle_data) >= HURST_WINDOW_SIZE:
                    hurst_update_counter += 1
                    if hurst_update_counter >= HURST_UPDATE_FREQUENCY:
                        hurst_update_counter = 0
                        prices = np.array([c['close'] for c in candle_data])
                        current_hurst = compute_hurst(prices, window_size=HURST_WINDOW_SIZE)
                        if current_hurst is not None:
                            logger.debug(f"Updated Hurst exponent: {current_hurst:.3f}")
                
                # Update extreme reversals periodically
                if len(candle_data) >= EXTREME_REVERSAL_WINDOW:
                    extreme_reversal_counter += 1
                    if extreme_reversal_counter >= EXTREME_REVERSAL_UPDATE_FREQ:
                        extreme_reversal_counter = 0
                        update_hurst_and_extremes(current_price)
                
                # Forecast price delta from energy model
                forecast_delta = energy_forecast_delta(df)
                forecast_price = current_price + forecast_delta
                
                # Detect patterns with intensities
                signals = detect_patterns_with_intensity(df, current_price) if not df.empty else []
                
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
                                stop_loss = entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER * (1 + FEE_RATE))
                                take_profit = entry_price + (price_move * (1 + FEE_RATE))
                            else:
                                stop_loss = entry_price + (atr * STOP_LOSS_ATR_MULTIPLIER * (1 + FEE_RATE))
                                take_profit = entry_price - (price_move * (1 + FEE_RATE))
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
            await asyncio.sleep(WS_PROCESSING_TIMEOUT)
            
        except asyncio.TimeoutError:
            # No messages in queue, continue loop
            continue
        except Exception as e:
            logger.error(f"Error in message processor: {e}")
            await asyncio.sleep(0.1)

async def websocket_listener(async_client):
    """Listen to WebSocket messages with enhanced error handling."""
    global ws_connected, ws_manager, ws_socket, ws_active, ws_reconnect_attempts, last_heartbeat_time
    
    ws_symbol = SYMBOL.lower()
    
    while ws_reconnect_attempts < WS_MAX_RECONNECT_ATTEMPTS:
        try:
            logger.info(f"Creating BinanceSocketManager for {SYMBOL}...")
            ws_manager = BinanceSocketManager(async_client)
            
            logger.info(f"Creating trade socket for {ws_symbol}...")
            ws_socket = ws_manager.trade_socket(ws_symbol)
            
            logger.info(f"Connecting to WebSocket for {SYMBOL} (attempt {ws_reconnect_attempts + 1}/{WS_MAX_RECONNECT_ATTEMPTS})...")
            
            # Connect to the WebSocket
            await ws_socket.__aenter__()
            ws_connected = True
            ws_active = True
            ws_reconnect_attempts = 0  # Reset counter on successful connection
            
            logger.info(f"WebSocket connection established for {SYMBOL}")
            
            # Listen for messages
            while ws_active:
                try:
                    # Receive message with timeout
                    trade = await asyncio.wait_for(ws_socket.recv(), timeout=5.0)
                    
                    # Update statistics and heartbeat time
                    processing_stats['messages_received'] += 1
                    last_heartbeat_time = time.time()
                    
                    # Try to put message in queue
                    try:
                        message_queue.put_nowait(trade)
                    except asyncio.QueueFull:
                        processing_stats['overflows'] += 1
                        logger.warning(f"Message queue full, dropping message. Overflows: {processing_stats['overflows']}")
                    
                    # Periodically log processing statistics
                    if processing_stats['messages_received'] % 1000 == 0:
                        queue_size = message_queue.qsize()
                        logger.info(f"Processing statistics: Received={processing_stats['messages_received']}, "
                                  f"Processed={processing_stats['messages_processed']}, "
                                  f"Overflows={processing_stats['overflows']}, "
                                  f"QueueSize={queue_size}")
                
                except asyncio.TimeoutError:
                    # No message received within timeout, check if we need to reconnect
                    if time.time() - last_heartbeat_time > WS_HEARTBEAT_INTERVAL:
                        logger.warning(f"No messages received for {WS_HEARTBEAT_INTERVAL} seconds, reconnecting...")
                        break
                    continue
                
                except BinanceWebsocketQueueOverflow as e:
                    processing_stats['overflows'] += 1
                    logger.error(f"Binance WebSocket queue overflow: {e}. Overflows: {processing_stats['overflows']}")
                    await asyncio.sleep(0.1)
                    continue
                
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    break
            
            # Clean up the socket
            try:
                await ws_socket.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing WebSocket socket: {e}")
            
            # Clean up the socket manager
            try:
                if hasattr(ws_manager, 'close'):
                    await ws_manager.close()
                elif hasattr(ws_manager, 'stop'):
                    await ws_manager.stop()
            except Exception as e:
                logger.error(f"Error closing WebSocket manager: {e}")
            
            ws_connected = False
            
            # Wait before reconnecting
            if ws_active and ws_reconnect_attempts < WS_MAX_RECONNECT_ATTEMPTS:
                logger.info(f"Waiting {WS_RECONNECT_DELAY} seconds before reconnecting...")
                await asyncio.sleep(WS_RECONNECT_DELAY)
        
        except Exception as e:
            logger.error(f"Error creating WebSocket connection: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            ws_connected = False
            
            # Clean up the socket manager if it exists
            if ws_manager:
                try:
                    if hasattr(ws_manager, 'close'):
                        await ws_manager.close()
                    elif hasattr(ws_manager, 'stop'):
                        await ws_manager.stop()
                except Exception as e:
                    logger.error(f"Error closing WebSocket manager during cleanup: {e}")
            
            # Wait before reconnecting
            if ws_reconnect_attempts < WS_MAX_RECONNECT_ATTEMPTS:
                logger.info(f"Waiting {WS_RECONNECT_DELAY} seconds before reconnecting...")
                await asyncio.sleep(WS_RECONNECT_DELAY)
        
        finally:
            ws_reconnect_attempts += 1
            logger.info(f"Reconnection attempt {ws_reconnect_attempts}/{WS_MAX_RECONNECT_ATTEMPTS}")
    
    # If we've exhausted all reconnection attempts
    if ws_reconnect_attempts >= WS_MAX_RECONNECT_ATTEMPTS:
        logger.error(f"Maximum WebSocket reconnection attempts ({WS_MAX_RECONNECT_ATTEMPTS}) reached. Giving up.")
        ws_connected = False
        ws_active = False

async def rest_fallback_poller(async_client):
    """Poll recent trades using REST API when WebSocket is not available."""
    global message_queue
    
    logger.info("Starting REST API fallback poller...")
    
    last_id = None
    
    while True:
        try:
            # Get recent trades
            if last_id is None:
                trades = await async_client.futures_recent_trades(symbol=SYMBOL, limit=100)
            else:
                trades = await async_client.futures_recent_trades(symbol=SYMBOL, limit=100, fromId=last_id)
            
            if trades:
                # Process each trade
                for trade in trades:
                    trade_id = int(trade['id'])
                    if last_id is None or trade_id > last_id:
                        last_id = trade_id
                        # Put trade in queue
                        try:
                            message_queue.put_nowait(trade)
                        except asyncio.QueueFull:
                            logger.warning("Message queue full in REST fallback, dropping trade")
            
            # Wait before next poll
            await asyncio.sleep(1.0)  # Poll every second
            
        except Exception as e:
            logger.error(f"Error in REST fallback poller: {e}")
            await asyncio.sleep(5.0)  # Wait longer on error

async def process_trade_stream(async_client):
    """
    Consume trade websocket, queue messages, and process them asynchronously.
    Falls back to REST API polling if WebSocket fails.
    """
    global processing_task, ws_active, balance_updater_task
    
    # Run backtest if enabled
    if BACKTEST_MODE:
        await run_backtest(async_client)
        return
    
    # Start the balance updater task if authenticated
    if AUTH:
        balance_updater_task = asyncio.create_task(balance_updater(async_client))
    
    # Start the message processor task
    processing_task = asyncio.create_task(message_processor(async_client))
    
    # Start the WebSocket listener
    ws_active = True
    websocket_task = asyncio.create_task(websocket_listener(async_client))
    
    # Start a fallback REST poller if WebSocket fails
    fallback_task = None
    
    try:
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [processing_task, websocket_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # If the WebSocket task completed (likely due to connection failure), start fallback
        if websocket_task in done:
            logger.warning("WebSocket connection failed, starting REST API fallback...")
            fallback_task = asyncio.create_task(rest_fallback_poller(async_client))
        
        # Wait for any remaining task to complete
        if pending:
            done, pending = await asyncio.wait(
                pending + ([fallback_task] if fallback_task else []),
                return_when=asyncio.FIRST_COMPLETED
            )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        logger.error(f"Error in trade stream processing: {e}")
    finally:
        # Clean up tasks
        if processing_task and not processing_task.done():
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
        
        if websocket_task and not websocket_task.done():
            websocket_task.cancel()
            try:
                await websocket_task
            except asyncio.CancelledError:
                pass
        
        if fallback_task and not fallback_task.done():
            fallback_task.cancel()
            try:
                await fallback_task
            except asyncio.CancelledError:
                pass
        
        if balance_updater_task and not balance_updater_task.done():
            balance_updater_task.cancel()
            try:
                await balance_updater_task
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
        ts = datetime.now().strftime('%H:%M:%S')
        if latest_row is None:
            hurst_text = f"Hurst: {current_hurst:.3f}" if current_hurst is not None else "Hurst: N/A"
            magic_text = f"Magic: N/A"
            print(f"[{ts}] No candle yet | ForecastΔ: {forecast_delta:+.4f} | ForecastPrice: {forecast_price:.2f} | Signals: none | Balance: ${current_balance:.4f} | {hurst_text} | {magic_text}")
            return
        price = float(latest_row['close'])
        vol = float(latest_row['volume'])
        momentum = float(latest_row.get('momentum', 0.0))
        ema_f = float(latest_row.get('ema_fast', 0.0))
        ema_s = float(latest_row.get('ema_slow', 0.0))
        stoch_k = float(latest_row.get('stoch_k', 0.0))
        open_count = len(open_trades)
        sig_text = format_signals_for_print(signals)
        
        # Get arctanh reversal signal
        arctanh_signal = update_arctanh_reversals(price)
        arctanh_text = f"Arctanh: {arctanh_signal}" if arctanh_signal else "Arctanh: none"
        
        # Get extreme reversal signal
        extreme_signal = update_hurst_and_extremes(price)
        extreme_text = f"Extreme: {extreme_signal}" if extreme_signal else "Extreme: none"
        
        # Get FFT cycle info
        fft_period, fft_phase = fft_cycle_detection(price_history)
        fft_text = f"FFT Period: {fft_period:.1f}" if fft_period else "FFT: none"
        
        # Get LRC info
        lrc_text = f"LRC: {lrc_signal}" if lrc_signal else "LRC: neutral"
        if lrc_middle is not None and lrc_upper is not None and lrc_lower is not None:
            lrc_text += f" (M:{lrc_middle:.2f} U:{lrc_upper:.2f} L:{lrc_lower:.2f})"
        
        # Get Hurst exponent info
        hurst_text = f"Hurst: {current_hurst:.3f}" if current_hurst is not None else "Hurst: N/A"
        if current_hurst is not None:
            if current_hurst > 0.5:
                hurst_text += " (Trending)"
            else:
                hurst_text += " (Mean-reverting)"
        
        # Get Magic Square info
        magic_signal = 0
        magic_confidence = 0.0
        magic_state = "none"
        if USE_MAGIC_SQUARE and len(price_history) >= 16:
            magic_signal, magic_confidence, magic_state = magic_forecaster.forecast_pattern(list(price_history))
        magic_text = f"Magic: {magic_state} ({magic_confidence:.2f})" if magic_confidence > 0 else "Magic: none"
        
        print(f"[{ts}] Price: {price:.2f} | Vol: {vol:.6f} | Mom: {momentum:.4f} | EMAf: {ema_f:.2f} | EMAs: {ema_s:.2f} | StochK: {stoch_k:.1f} | Regime: {market_regime} | Volatility: {volatility_state} | {arctanh_text} | {extreme_text} | {fft_text} | {lrc_text} | {hurst_text} | {magic_text} | ForecastΔ: {forecast_delta:+.4f} | Forecast: {forecast_price:.2f} | Signals: {sig_text} | Open: {open_count} | Balance: ${current_balance:.4f}")
    except Exception as e:
        logger.debug(f"print_main_status error: {e}")

# ---------------------------
# Health Check Server
# ---------------------------
async def health_check(request):
    """Simple health check endpoint."""
    return web.Response(text="OK", status=200)

async def start_health_server():
    """Start a simple HTTP server for health checks."""
    app = web.Application()
    app.router.add_get('/health', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    logger.info("Health check server started on port 8080")
    return runner

# ---------------------------
# WebSocket Keep-Alive
# ---------------------------
async def websocket_keep_alive():
    """Monitor WebSocket connection and restart if needed."""
    global ws_active, ws_reconnect_attempts
    
    while True:
        try:
            # Check if WebSocket is active
            if ws_active and time.time() - last_heartbeat_time > WS_HEARTBEAT_INTERVAL * 2:
                logger.warning("WebSocket heartbeat timeout, reconnecting...")
                ws_active = False  # Trigger reconnection
                ws_reconnect_attempts = 0  # Reset counter
            
            await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
        except Exception as e:
            logger.error(f"Error in WebSocket keep-alive: {e}")
            await asyncio.sleep(10)

# ---------------------------
# Signal Handlers
# ---------------------------
def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        # Cancel all tasks
        for task in asyncio.all_tasks():
            task.cancel()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ---------------------------
# Main
# ---------------------------
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

async def main():
    setup_signal_handlers()
    
    # Add startup delay
    logger.info("Bot starting up, waiting 5 seconds...")
    await asyncio.sleep(5)
    
    global bot_start_time, balance_updater_task
    bot_start_time = time.time()  # Record when the bot starts
    
    async_client = None
    health_runner = None
    
    try:
        if AUTH:
            async_client = await AsyncClient.create(API_KEY, API_SECRET)
        else:
            async_client = await AsyncClient.create()

        await set_leverage(async_client)
        
        # Start health check server
        health_runner = await start_health_server()
        
        # Start the keep-alive task
        keep_alive_task = asyncio.create_task(websocket_keep_alive())
        
        # Main loop with error handling
        while True:
            try:
                logger.info("Starting trade stream processing...")
                await process_trade_stream(async_client)
                
                # If we get here, the process_trade_stream completed
                # Wait a bit before restarting
                logger.info("Trade stream processing completed, restarting in 10 seconds...")
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                logger.info("Main task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                
                # Wait before retrying
                logger.info("Waiting 30 seconds before retrying...")
                await asyncio.sleep(30)
    finally:
        # Cancel the keep-alive task
        if 'keep_alive_task' in locals():
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Clean up health server
        if health_runner:
            await health_runner.cleanup()
        
        # Clean up
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

# Entrypoint
if __name__ == "__main__":
    try:
        logger.info("Starting trading bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")