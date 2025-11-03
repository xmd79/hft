import asyncio
import os
import sys
import logging
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
import time
import math
from collections import deque, defaultdict
import json
import statistics

import numpy as np
import pandas as pd
import requests
from binance import AsyncClient, BinanceSocketManager
from binance.client import Client
from binance.exceptions import BinanceWebsocketQueueOverflow
from scipy.fft import fft, fftfreq
import talib
import pywt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

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
# Configuration (tune me)
# ---------------------------
SYMBOL = 'BTCUSDC'                 # Target perpetual futures symbol
TIMEFRAME_SECONDS = 15             # Pseudo-candle length in seconds
LEVERAGE = 125                     # Default leverage to set (using max from start)
MAX_LEVERAGE = 125                 # Maximum leverage to use if needed (BTC max is 125x)
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
BACKTEST_FILE = 'backtest_results.json'  # File to store backtest results

# Market regime parameters
ADX_PERIOD = 14                    # Period for ADX calculation
DMI_PERIOD = 14                    # Period for DMI calculation (same as ADX)
TREND_THRESHOLD = 25               # ADX value above which market is considered trending

# Position management parameters
ATR_PERIOD = 14                    # Period for ATR calculation
STOP_LOSS_ATR_MULTIPLIER = 1.5     # Stop loss distance in ATR units
TAKE_PROFIT_ATR_MULTIPLIER = 3.0   # Take profit distance in ATR units
TRAILING_STOP_ATR_MULTIPLIER = 1.0 # Trailing stop distance in ATR units

# Fee adjustment
FEE_RATE = 0.0006                 # 0.06% fee per side (0.12% round trip)

# Linear Regression Channel parameters
LRC_PERIOD = 200                   # Period for Linear Regression Channel
LRC_DEVIATION_MULTIPLIER = 2.0     # Deviation multiplier for channel width

# ML Forecasting parameters
ML_LOOKBACK = 200                  # Lookback period for ML forecasting

# Rounding defaults
DEFAULT_QUANTITY_STEP = 0.000001
DEFAULT_PRICE_TICK = 0.01

# WebSocket configuration
WS_QUEUE_SIZE = 2000               # Internal queue size for processing
WS_RECONNECT_DELAY = 5            # Seconds to wait before reconnecting
WS_MAX_RECONNECT_ATTEMPTS = 10    # Maximum reconnection attempts
WS_PROCESSING_TIMEOUT = 0.001      # Timeout for message processing (seconds)
WS_HEARTBEAT_INTERVAL = 30        # Seconds between heartbeat checks

# Balance update configuration
BALANCE_UPDATE_INTERVAL = 300     # Update balance every 5 minutes (in seconds)

# Support/Resistance parameters
SUPPORT_RESISTANCE_TOLERANCE = 0.005  # 0.5% tolerance for support/resistance

# ---------------------------
# Global runtime state
# ---------------------------
open_trades = []                   # List of open trades
virtual_balance = INITIAL_BALANCE  # Track virtual balance
real_balance = 0.0                 # Track real balance
available_balances = {}            # Track all available asset balances
candle_data = []                   # List of pseudo-candle dicts
symbol_info_cache = {}             # Cache symbol exchange_info
AUTH = False                       # Whether API keys were provided
backtest_results = []              # Store backtest results
market_regime = "ranging"          # Current market regime (trending/ranging)
volatility_state = "normal"        # Current volatility state (low/normal/high)
last_pattern_time = {}             # Track last pattern execution time for cooldown
bot_start_time = time.time()       # Track when the bot started
last_balance_update_time = 0        # Track last balance update time
current_leverage = LEVERAGE        # Track current leverage setting
max_allowed_leverage = MAX_LEVERAGE # Track maximum allowed leverage
trade_count = 0                    # Track number of trades

# For adaptive thresholds
volatility_history = deque(maxlen=50)  # Store recent volatility measurements
atr_values = deque(maxlen=ATR_PERIOD)  # Store ATR values for calculation

# For market regime detection
adx_values = deque(maxlen=ADX_PERIOD)  # Store ADX values
plus_di_values = deque(maxlen=DMI_PERIOD)
minus_di_values = deque(maxlen=DMI_PERIOD)

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

# WebSocket management
ws_reconnect_attempts = 0
ws_connected = False
ws_manager = None
ws_socket = None
ws_active = False

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

def ceil_to_step(q: float, step: float) -> float:
    """Ceil quantity to the nearest step size using Decimal for precision safety."""
    if step is None or step == 0:
        return float(q)
    q_dec = Decimal(str(q))
    step_dec = Decimal(str(step))
    # Round up to the next step
    steps = (q_dec / step_dec).to_integral_value(rounding=ROUND_UP)
    quant = (steps * step_dec).quantize(step_dec, rounding=ROUND_DOWN)
    return float(quant)

async def fetch_symbol_info(async_client, symbol: str):
    """Fetch and cache lot step and price tick details for the symbol."""
    global symbol_info_cache, max_allowed_leverage
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
    
    # Get the maximum leverage allowed for this symbol
    max_allowed_leverage = int(info.get('maxLeverage', MAX_LEVERAGE))
    logger.info(f"Maximum allowed leverage for {symbol}: {max_allowed_leverage}x")

    result = {"stepSize": stepSize, "minQty": minQty, "tickSize": tickSize}
    symbol_info_cache[key] = result
    return result

# ---------------------------
# Data Cleaning
# ---------------------------
def clean_data(data, min_val=1e-10):
    """Replace NaN, Inf, and zero values with a small positive value"""
    if isinstance(data, list):
        data = np.array(data)
    
    # Replace NaN and Inf with a small positive value
    data = np.nan_to_num(data, nan=min_val, posinf=min_val, neginf=-min_val)
    
    # Replace zeros with a small positive value
    data = np.where(np.abs(data) < min_val, min_val * np.sign(data) if data.any() != 0 else min_val, data)
    
    return data

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

def calculate_dmi(df: pd.DataFrame, period=DMI_PERIOD) -> tuple:
    """Calculate Directional Movement Index (DMI) for trend strength."""
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
    adx, plus_di, minus_di = calculate_dmi(df)
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
# Wavelet Analysis
# ---------------------------
def wavelet_analysis(candles, timeframe, forecast_periods=1):
    """
    Perform wavelet analysis to identify cycles and forecast future prices.
    
    Parameters:
    - candles: List of candle data
    - timeframe: Timeframe string (e.g., "1m", "5m")
    - forecast_periods: Number of periods to forecast (default: 1)
    
    Returns:
    - dominant_cycle: Most dominant cycle period (in candles)
    - cycle_direction: "Up" or "Down" based on wavelet phase
    - forecast_price: Forecasted price for next period
    - wavelet_cycles: List of detected cycle periods
    - reconstruction: Reconstructed signal from dominant wavelet component
    """
    try:
        # Import PyWavelets if available
        import pywt
    except ImportError:
        logger.warning(f"PyWavelets not available for {timeframe} wavelet analysis.")
        return 0, "N/A", Decimal('0.0'), [], np.zeros(len(candles))
    
    if len(candles) < 10:
        logger.warning(f"Insufficient data ({len(candles)}) for wavelet analysis in {timeframe}.")
        return 0, "N/A", Decimal('0.0'), [], np.zeros(len(candles))
    
    try:
        # Extract close prices
        closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
        closes = clean_data(closes)
        n = len(closes)
        
        # Calculate mean for later conversion back to price scale
        mean_close = np.mean(closes)
        
        # Choose wavelet and determine decomposition levels
        wavelet = 'db4'  # Daubechies 4 wavelet
        max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        levels = min(4, max_level)  # Use up to 4 levels or as many as possible
        
        if levels < 1:
            logger.warning(f"Insufficient data for wavelet decomposition in {timeframe}.")
            return 0, "N/A", Decimal('0.0'), [], np.zeros(n)
        
        # Perform wavelet decomposition on centered data
        centered_closes = closes - mean_close
        coeffs = pywt.wavedec(centered_closes, wavelet, level=levels)
        
        # Calculate energy for each level
        energies = []
        for i in range(1, len(coeffs)):
            energy = np.sum(coeffs[i]**2)
            energies.append(energy)
        
        # Find dominant level (highest energy)
        if energies:
            dominant_level_idx = np.argmax(energies) + 1  # Skip approximation (index 0)
        else:
            dominant_level_idx = 1
        
        # Calculate dominant cycle period (in candles)
        dominant_cycle = 2 ** dominant_level_idx
        
        # Reconstruct the dominant component
        coeffs_dominant = [np.zeros_like(c) for c in coeffs]
        coeffs_dominant[dominant_level_idx] = coeffs[dominant_level_idx]
        reconstruction = pywt.waverec(coeffs_dominant, wavelet)
        
        # Ensure reconstruction has same length as original
        if len(reconstruction) > n:
            reconstruction = reconstruction[:n]
        elif len(reconstruction) < n:
            reconstruction = np.pad(reconstruction, (0, n - len(reconstruction)), 'constant')
        
        # Determine cycle direction based on phase of dominant component
        if len(reconstruction) >= 3:
            # Calculate slope of last 3 points
            x = np.array([0, 1, 2])
            y = reconstruction[-3:]
            slope = np.polyfit(x, y, 1)[0]
            cycle_direction = "Up" if slope > 0 else "Down"
        else:
            cycle_direction = "N/A"
        
        # Forecast next value using linear extrapolation of dominant component
        if len(reconstruction) >= 3:
            x = np.array([len(reconstruction)-3, len(reconstruction)-2, len(reconstruction)-1])
            y = reconstruction[-3:]
            coeffs_fit = np.polyfit(x, y, 1)
            next_value = np.polyval(coeffs_fit, len(reconstruction))
            # Convert back to original price scale by adding the mean
            forecast_price = Decimal(str(next_value + mean_close))
            
            # Apply bounds based on recent price range to prevent unrealistic forecasts
            recent_min = np.min(closes[-20:])
            recent_max = np.max(closes[-20:])
            forecast_price = max(forecast_price, Decimal(str(recent_min * 0.95)))
            forecast_price = min(forecast_price, Decimal(str(recent_max * 1.05)))
        else:
            forecast_price = Decimal(str(closes[-1]))
        
        # Collect all detected cycle periods
        wavelet_cycles = [2 ** i for i in range(1, len(coeffs))]
        
        logger.info(
            f"{timeframe} - Wavelet Analysis: Dominant Cycle: {dominant_cycle}, "
            f"Direction: {cycle_direction}, Forecast: {forecast_price:.25f}"
        )
        print(
            f"{timeframe} - Wavelet Analysis: Dominant Cycle: {dominant_cycle}, "
            f"Direction: {cycle_direction}, Forecast: {forecast_price:.25f}"
        )
        
        return dominant_cycle, cycle_direction, forecast_price, wavelet_cycles, reconstruction
    
    except Exception as e:
        logger.error(f"Error in wavelet analysis for {timeframe}: {e}")
        print(f"Error in wavelet analysis for {timeframe}: {e}")
        return 0, "N/A", Decimal('0.0'), [], np.zeros(len(candles))

# ---------------------------
# ML Forecasting
# ---------------------------
def generate_ml_forecast(candles, timeframe, forecast_periods=5):
    """
    Advanced ML forecast using Random Forest and Random Walk concepts for cycle prediction.
    
    Parameters:
    - candles: List of candle data
    - timeframe: Timeframe string (e.g., "1m", "5m")
    - forecast_periods: Number of periods to forecast (default: 5)
    
    Returns:
    - forecast_price: Predicted price after forecast_periods
    """
    if len(candles) < ML_LOOKBACK:
        logger.warning(f"Insufficient data ({len(candles)}) for ML forecast in {timeframe}")
        print(f"Insufficient data ({len(candles)}) for ML forecast in {timeframe}")
        return Decimal('0.0')
    
    try:
        # Extract data
        closes = np.array([float(c['close']) for c in candles[-ML_LOOKBACK:]], dtype=np.float64)
        highs = np.array([float(c['high']) for c in candles[-ML_LOOKBACK:]], dtype=np.float64)
        lows = np.array([float(c['low']) for c in candles[-ML_LOOKBACK:]], dtype=np.float64)
        volumes = np.array([float(c['volume']) for c in candles[-ML_LOOKBACK:]], dtype=np.float64)
        
        # Clean the data
        closes = clean_data(closes)
        highs = clean_data(highs)
        lows = clean_data(lows)
        volumes = clean_data(volumes)
        
        # Create DataFrame for easier feature engineering
        df = pd.DataFrame({
            'close': closes,
            'high': highs,
            'low': lows,
            'volume': volumes
        })
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate lagged returns
        for lag in range(1, 6):
            df[f'lagged_return_{lag}'] = df['returns'].shift(lag)
        
        # Calculate rolling statistics
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['close'].rolling(window=window).max()
        
        # Calculate technical indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
        
        # Calculate Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Calculate ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Calculate OBV
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Calculate Hurst exponent
        df['hurst'] = 0.5  # Default value
        for i in range(20, len(df)):
            window_data = df['close'].iloc[i-20:i].values
            if len(window_data) > 10:
                try:
                    H, _, _ = compute_Hc(window_data, kind='price', simplified=True)
                    df.at[df.index[i], 'hurst'] = H
                except:
                    pass
        
        # Calculate FFT components
        df['fft_real'] = 0.0
        df['fft_imag'] = 0.0
        df['fft_power'] = 0.0
        
        for i in range(20, len(df)):
            window_data = df['close'].iloc[i-20:i].values
            if len(window_data) > 10:
                try:
                    fft_result = fft(window_data - np.mean(window_data))
                    freqs = np.fft.fftfreq(len(window_data))
                    
                    # Get the dominant frequency (excluding DC component)
                    power = np.abs(fft_result[1:]) ** 2
                    if len(power) > 0:
                        dominant_idx = np.argmax(power) + 1
                        df.at[df.index[i], 'fft_real'] = np.real(fft_result[dominant_idx])
                        df.at[df.index[i], 'fft_imag'] = np.imag(fft_result[dominant_idx])
                        df.at[df.index[i], 'fft_power'] = power[dominant_idx - 1]
                except:
                    pass
        
        # Create target variable (future returns)
        df['target'] = df['close'].shift(-forecast_periods) / df['close'] - 1
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < 50:
            logger.warning(f"Insufficient data after feature engineering in {timeframe}")
            print(f"Insufficient data after feature engineering in {timeframe}")
            return Decimal('0.0')
        
        # Define features and target
        feature_columns = [col for col in df.columns if col not in ['close', 'high', 'low', 'volume', 'target']]
        X = df[feature_columns].values
        y = df['target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        
        # Get the most recent data point for prediction
        last_data = df.iloc[-1:][feature_columns].values
        last_data_scaled = scaler.transform(last_data)
        
        # Predict future return
        predicted_return = model.predict(last_data_scaled)[0]
        
        # Calculate forecast price
        current_close = Decimal(str(df['close'].iloc[-1]))
        forecast_price = current_close * (Decimal('1') + Decimal(str(predicted_return)))
        
        # Adjust forecast based on cycle direction
        # Get current cycle direction from the last few price movements
        recent_returns = df['returns'].iloc[-5:].values
        if len(recent_returns) >= 3:
            # Simple trend detection
            if np.mean(recent_returns) > 0:
                # Uptrend - ensure forecast is above current price
                forecast_price = max(forecast_price, current_close * Decimal('1.001'))
            else:
                # Downtrend - ensure forecast is below current price
                forecast_price = min(forecast_price, current_close * Decimal('0.999'))
        
        # Apply bounds based on recent price range
        recent_min = Decimal(str(df['close'].iloc[-20:].min()))
        recent_max = Decimal(str(df['close'].iloc[-20:].max()))
        
        # Don't let forecast go beyond reasonable bounds
        forecast_price = max(forecast_price, recent_min * Decimal('0.95'))
        forecast_price = min(forecast_price, recent_max * Decimal('1.05'))
        
        logger.info(f"{timeframe} - ML Forecast (Random Forest): {forecast_price:.25f}")
        print(f"{timeframe} - ML Forecast (Random Forest): {forecast_price:.25f}")
        
        return forecast_price
    except Exception as e:
        logger.error(f"Error generating ML forecast for {timeframe}: {e}")
        print(f"Error generating ML forecast for {timeframe}: {e}")
        return Decimal('0.0')

# ---------------------------
# FFT Forecasting
# ---------------------------
def generate_fft_forecast(candles, timeframe, forecast_periods=5):
    if len(candles) < 10:
        logger.warning(f"Insufficient data ({len(candles)}) for FFT forecast in {timeframe}")
        print(f"Insufficient data ({len(candles)}) for FFT forecast in {timeframe}")
        return Decimal('0.0')
    
    try:
        closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
        if np.any(np.isnan(closes)) or np.any(closes <= 0):
            logger.warning(f"Invalid close prices in {timeframe} for FFT forecast.")
            print(f"Invalid close prices in {timeframe} for FFT forecast.")
            return Decimal('0.0')
        
        # Clean the data
        closes = clean_data(closes)
        
        current_close = Decimal(str(closes[-1])) if len(closes) > 0 else Decimal('0.0')
        
        mean_close = np.mean(closes)
        
        fft_result = fft(closes - mean_close)  # Center around zero
        freqs = np.fft.fftfreq(len(closes))
        
        magnitudes = np.abs(fft_result)
        
        # Skip DC component (index 0) when finding dominant frequencies
        if len(magnitudes) > 1:
            # Get both positive and negative frequencies
            positive_freq_indices = np.where(freqs > 0)[0]
            negative_freq_indices = np.where(freqs < 0)[0]
            
            # Initialize all variables to ensure they're defined
            pos_freq, pos_amp, pos_phase = 0, 0, 0
            neg_freq, neg_amp, neg_phase = 0, 0, 0
            
            # Find the most significant positive and negative frequencies
            if len(positive_freq_indices) > 0:
                pos_idx = positive_freq_indices[np.argmax(magnitudes[positive_freq_indices])]
                pos_freq = freqs[pos_idx]
                pos_amp = magnitudes[pos_idx] / len(closes) * 2
                pos_phase = np.angle(fft_result[pos_idx])
                
            if len(negative_freq_indices) > 0:
                neg_idx = negative_freq_indices[np.argmax(magnitudes[negative_freq_indices])]
                neg_freq = freqs[neg_idx]
                neg_amp = magnitudes[neg_idx] / len(closes) * 2
                neg_phase = np.angle(fft_result[neg_idx])
            
            # Determine which frequency is more dominant
            if pos_amp > neg_amp:
                dominant_freq = pos_freq
                dominant_phase = pos_phase
                cycle_direction = "Down"  # Positive frequency dominant suggests down cycle
                dominant_amp = pos_amp
            else:
                dominant_freq = neg_freq
                dominant_phase = neg_phase
                cycle_direction = "Up"    # Negative frequency dominant suggests up cycle
                dominant_amp = neg_amp
        else:
            dominant_freq, dominant_amp, dominant_phase, cycle_direction = 0, 0, 0, "N/A"
        
        # Generate forecast using the dominant frequency
        forecast = np.zeros(forecast_periods)
        future_t = np.arange(len(closes), len(closes) + forecast_periods)
        
        if dominant_amp > 0:
            forecast += dominant_amp * np.cos(2 * np.pi * dominant_freq * future_t + dominant_phase)
        
        forecast_price = Decimal(str(forecast[-1] + mean_close))
        
        # Use min and max thresholds to adjust/bound the forecast
        min_th, _, max_th, _ = calculate_thresholds(closes)
        amp_adjust = (max_th - min_th) / Decimal('2')
        mean_close_decimal = Decimal(str(mean_close))
        
        # Adjust forecast based on cycle direction
        if cycle_direction == "Up":
            # For up cycle, forecast should be higher than current price
            forecast_price = max(forecast_price, current_close)
            forecast_price = min(forecast_price, mean_close_decimal + amp_adjust)
        else:
            # For down cycle, forecast should be lower than current price
            forecast_price = min(forecast_price, current_close)
            forecast_price = max(forecast_price, mean_close_decimal - amp_adjust)
        
        logger.info(f"{timeframe} - FFT Forecast: {forecast_price:.25f} (Cycle Direction: {cycle_direction})")
        print(f"{timeframe} - FFT Forecast: {forecast_price:.25f} (Cycle Direction: {cycle_direction})")
        
        return forecast_price
    except Exception as e:
        logger.error(f"Error generating FFT forecast for {timeframe}: {e}")
        print(f"Error generating FFT forecast for {timeframe}: {e}")
        return Decimal('0.0')

# ---------------------------
# Advanced FFT Forecasting
# ---------------------------
def advanced_fft_forecast(candles, timeframe, min_threshold, max_threshold, current_close, forecast_periods=5):
    """
    Advanced FFT forecast using 360-degree unit circle, harmonic analysis, and cycle stage detection.
    
    Parameters:
    - candles: List of candle data
    - timeframe: Timeframe string (e.g., "1m", "5m")
    - min_threshold: Minimum price threshold
    - max_threshold: Maximum price threshold
    - current_close: Current closing price
    - forecast_periods: Number of periods to forecast (default: 5)
    
    Returns:
    - forecast_price: Predicted price after forecast_periods
    - cycle_stage: Current stage in the cycle ("Early", "Middle", "Late")
    - cycle_direction: Cycle direction ("Up" or "Down")
    - dominant_frequency: Most significant frequency component
    - phase_angle: Phase angle in degrees (0-360)
    - harmonic_strength: Strength of harmonic components
    """
    if len(candles) < 10:
        logger.warning(f"Insufficient data ({len(candles)}) for advanced FFT forecast in {timeframe}")
        print(f"Insufficient data ({len(candles)}) for advanced FFT forecast in {timeframe}")
        return Decimal('0.0'), "N/A", "N/A", 0.0, 0.0, 0.0
    
    try:
        # Extract close prices
        closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
        
        # Clean the data
        closes = clean_data(closes)
        
        # Get last 5 close values for recent trend analysis
        last_5_closes = closes[-5:] if len(closes) >= 5 else closes
        
        # Center the data around zero
        mean_close = np.mean(closes)
        centered_closes = closes - mean_close
        
        # Perform FFT
        fft_result = fft(centered_closes)
        freqs = np.fft.fftfreq(len(closes))
        
        # Calculate power spectrum
        power = np.abs(fft_result) ** 2
        
        # Skip DC component (index 0)
        indices = np.arange(1, len(closes))
        
        # Find most negative and most positive frequencies
        negative_freq_indices = indices[freqs[indices] < 0]
        positive_freq_indices = indices[freqs[indices] > 0]
        
        # Initialize all variables to ensure they're defined
        dominant_neg_freq, dominant_neg_power, neg_phase = 0, 0, 0
        dominant_pos_freq, dominant_pos_power, pos_phase = 0, 0, 0
        
        # Get dominant negative and positive frequencies
        if len(negative_freq_indices) > 0:
            neg_idx = negative_freq_indices[np.argmax(power[negative_freq_indices])]
            dominant_neg_freq = freqs[neg_idx]
            dominant_neg_power = power[neg_idx]
            neg_phase = np.angle(fft_result[neg_idx])
            
        if len(positive_freq_indices) > 0:
            pos_idx = positive_freq_indices[np.argmax(power[positive_freq_indices])]
            dominant_pos_freq = freqs[pos_idx]
            dominant_pos_power = power[pos_idx]
            pos_phase = np.angle(fft_result[pos_idx])
        
        # Determine which frequency is more dominant
        if dominant_neg_power > dominant_pos_power:
            dominant_frequency = dominant_neg_freq
            dominant_phase = neg_phase
            cycle_direction = "Up"  # Negative frequency dominant suggests up cycle
        else:
            dominant_frequency = dominant_pos_freq
            dominant_phase = pos_phase
            cycle_direction = "Down"  # Positive frequency dominant suggests down cycle
        
        # Convert phase to degrees (0-360)
        phase_angle = (np.degrees(dominant_phase) + 360) % 360
        
        # Determine cycle stage based on phase angle
        if 0 <= phase_angle < 120:
            cycle_stage = "Early"
        elif 120 <= phase_angle < 240:
            cycle_stage = "Middle"
        else:
            cycle_stage = "Late"
        
        # Calculate harmonic strength (sum of power of top 5 harmonics)
        sorted_indices = indices[np.argsort(power[indices])[-5:]]
        harmonic_strength = np.sum(power[sorted_indices]) / np.sum(power[indices]) if len(indices) > 0 else 0
        
        # Generate forecast using dominant frequency and harmonics
        forecast = np.zeros(forecast_periods)
        future_t = np.arange(len(closes), len(closes) + forecast_periods)
        
        # Add contribution from dominant frequency
        if dominant_neg_power > 0 or dominant_pos_power > 0:
            dominant_amp = np.sqrt(max(dominant_neg_power, dominant_pos_power)) / len(closes) * 2
            forecast += dominant_amp * np.cos(2 * np.pi * dominant_frequency * future_t + dominant_phase)
        
        # Add contributions from harmonics
        for idx in sorted_indices:
            if idx in [neg_idx if 'neg_idx' in locals() else -1, pos_idx if 'pos_idx' in locals() else -1]:
                continue  # Skip the dominant frequency we already added
            freq = freqs[idx]
            amp = np.sqrt(power[idx]) / len(closes) * 2
            phase = np.angle(fft_result[idx])
            forecast += amp * np.cos(2 * np.pi * freq * future_t + phase)
        
        # Convert forecast back to price scale
        forecast_price = Decimal(str(forecast[-1] + mean_close))
        
        # Apply a directional bias based on cycle direction
        # This ensures the forecast moves in the direction of the cycle
        if cycle_direction == "Up":
            # For up cycle, ensure forecast is at least slightly above current price
            min_forecast = current_close * (Decimal('1') + Decimal('0.001'))  # 0.1% above current
            forecast_price = max(forecast_price, min_forecast)
        else:
            # For down cycle, ensure forecast is at least slightly below current price
            max_forecast = current_close * (Decimal('1') - Decimal('0.001'))  # 0.1% below current
            forecast_price = min(forecast_price, max_forecast)
        
        # Apply threshold bounds to keep forecast within reasonable range
        price_range = max_threshold - min_threshold
        tolerance = price_range * SUPPORT_RESISTANCE_TOLERANCE
        
        # Bound the forecast by the thresholds
        if forecast_price > max_threshold - tolerance:
            forecast_price = max_threshold - tolerance
        if forecast_price < min_threshold + tolerance:
            forecast_price = min_threshold + tolerance
        
        logger.info(
            f"{timeframe} - Advanced FFT Forecast: {forecast_price:.25f}, "
            f"Cycle Stage: {cycle_stage}, Direction: {cycle_direction}, "
            f"Dominant Freq: {dominant_frequency:.25f}, Phase: {phase_angle:.2f}°, "
            f"Harmonic Strength: {harmonic_strength:.25f}"
        )
        print(
            f"{timeframe} - Advanced FFT Forecast: {forecast_price:.25f}, "
            f"Cycle Stage: {cycle_stage}, Direction: {cycle_direction}, "
            f"Dominant Freq: {dominant_frequency:.25f}, Phase: {phase_angle:.2f}°, "
            f"Harmonic Strength: {harmonic_strength:.25f}"
        )
        
        return forecast_price, cycle_stage, cycle_direction, dominant_frequency, phase_angle, harmonic_strength
    except Exception as e:
        logger.error(f"Error in advanced FFT forecast for {timeframe}: {e}")
        print(f"Error in advanced FFT forecast for {timeframe}: {e}")
        return Decimal('0.0'), "N/A", "N/A", 0.0, 0.0, 0.0

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

    # Adjust pattern detection based on market regime
    if market_regime.startswith("trending"):
        # In trending markets, we want to be more selective with reversals
        stoch_threshold = 15 if market_regime.endswith("bullish") else 85
    else:
        # In ranging markets, standard thresholds
        stoch_threshold = 20 if market_regime.endswith("bullish") else 80

    # Check for arctanh reversal signal
    arctanh_signal = update_arctanh_reversals(current_price)
    
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

    # DBD
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] < df['open'].iloc[i-3] and
            df['close'].iloc[i-1] < df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] > stoch_threshold and
            (arctanh_signal == "short" or fft_confirms_short or lrc_signal == "short")):
        intensity = pattern_intensity(df, i)
        signals.append(('sell', float(df['close'].iloc[i]), int(intensity * 100), 'DBD'))

    # RBD
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] > df['open'].iloc[i-3] and
            df['close'].iloc[i-1] < df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] > stoch_threshold and
            (arctanh_signal == "short" or fft_confirms_short or lrc_signal == "short")):
        intensity = pattern_intensity(df, i)
        signals.append(('sell', float(df['close'].iloc[i]), int(intensity * 100), 'RBD'))

    # DBR
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] < df['open'].iloc[i-3] and
            df['close'].iloc[i-1] > df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] < (100 - stoch_threshold) and
            (arctanh_signal == "long" or fft_confirms_long or lrc_signal == "long")):
        intensity = pattern_intensity(df, i)
        signals.append(('buy', float(df['close'].iloc[i]), int(intensity * 100), 'DBR'))

    # RBR
    if (time_valid and spike_ok and base_ok and confirm_candle_ok and
            df['close'].iloc[i-3] > df['open'].iloc[i-3] and
            df['close'].iloc[i-1] > df['open'].iloc[i-1] and
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] < (100 - stoch_threshold) and
            (arctanh_signal == "long" or fft_confirms_long or lrc_signal == "long")):
        intensity = pattern_intensity(df, i)
        signals.append(('buy', float(df['close'].iloc[i]), int(intensity * 100), 'RBR'))

    # Spike Buy (downward spike rejection)
    if (df['spike'].iloc[i-1] == -1 and
            df['open'].iloc[i] > df['low'].iloc[i-1] and
            df['volume'].iloc[i-1] > df['volume_mean'].iloc[i-1] * VOLUME_THRESHOLD and
            abs(df['momentum'].iloc[i-1]) > MOMENTUM_THRESHOLD and
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            df['stoch_k'].iloc[i] < (100 - stoch_threshold) and
            (arctanh_signal == "long" or fft_confirms_long or lrc_signal == "long")):
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
            df['stoch_k'].iloc[i] > stoch_threshold and
            (arctanh_signal == "short" or fft_confirms_short or lrc_signal == "short")):
        intensity = pattern_intensity(df, i)
        intensity = min(1.0, intensity + 0.15)
        signals.append(('sell', float(df['close'].iloc[i]), int(intensity * 100), 'SpikeSell'))

    # Extra quick spike predictor: large sudden momentum on last tick
    last_mom = abs(df['momentum'].iloc[-1])
    if last_mom > MOMENTUM_THRESHOLD * 3:
        direction = 'buy' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'sell'
        intensity = min(100, int(score_between(last_mom, MOMENTUM_THRESHOLD*3, MOMENTUM_THRESHOLD*8) * 100 + 20))
        
        # Only add if confirmed by arctanh, FFT, or LRC
        if (direction == 'buy' and (arctanh_signal == "long" or fft_confirms_long or lrc_signal == "long")) or \
           (direction == 'sell' and (arctanh_signal == "short" or fft_confirms_short or lrc_signal == "short")):
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
    Uses ATR for dynamic stop-loss and take-profit levels, adjusted for fees.
    Fixed to properly utilize entire balance with optimal leverage.
    """
    global virtual_balance, real_balance, current_leverage

    # Use real balance if available (>0), otherwise use virtual balance
    balance = real_balance if real_balance > 0 else virtual_balance
    
    # Check if balance is too low to trade
    if balance < MIN_BALANCE_THRESHOLD:
        return 0.0, 0.0, 0.0, False
    
    # Get symbol info for quantity constraints
    stepSize = DEFAULT_QUANTITY_STEP
    minQty = MIN_ORDER_SIZE
    try:
        info = await fetch_symbol_info(async_client, SYMBOL)
        stepSize = info.get('stepSize', DEFAULT_QUANTITY_STEP)
        minQty = info.get('minQty', MIN_ORDER_SIZE)
    except Exception:
        pass
    
    # Calculate ATR for dynamic position sizing
    atr = calculate_atr(df) if not df.empty else 0.0
    if atr == 0.0:
        atr = BASE_SPIKE_THRESHOLD / 2.0  # Fallback to fixed value
    
    # Use ATR for stop distance
    stop_distance = atr * STOP_LOSS_ATR_MULTIPLIER
    take_profit_distance = atr * TAKE_PROFIT_ATR_MULTIPLIER
    
    # Adjust for fees
    fee_adjusted_stop_distance = stop_distance * (1 + FEE_RATE)
    fee_adjusted_take_profit_distance = take_profit_distance * (1 + FEE_RATE)
    
    # For very small balances, we need to calculate the minimum required quantity
    # and then find the optimal leverage to make it work
    if balance < 5.0:
        # Calculate minimum notional value required (5 USDC for BTCUSDC)
        min_notional = 5.0
        
        # Calculate minimum quantity needed to meet notional requirement
        min_quantity = min_notional / entry_price
        
        # Ensure we meet the minimum quantity requirement
        min_quantity = max(min_quantity, minQty)
        
        # Calculate the minimum leverage needed to trade this quantity
        min_leverage_needed = (min_quantity * entry_price) / balance
        
        # If we need more leverage than allowed, cap it at maximum
        if min_leverage_needed > max_allowed_leverage:
            min_leverage_needed = max_allowed_leverage
            # Recalculate quantity with maximum leverage
            min_quantity = (balance * max_allowed_leverage) / entry_price
        
        # Set quantity to the minimum required
        quantity = floor_to_step(min_quantity, stepSize)
        
        # Update leverage if needed
        if min_leverage_needed > current_leverage:
            logger.info(f"Adjusting leverage from {current_leverage}x to {min_leverage_needed:.1f}x for small balance")
            await set_leverage(async_client, int(min_leverage_needed))
    else:
        # For normal balances, calculate quantity based on risk percentage
        risk_percent = PNL_PERCENT
        risk_amount = balance * risk_percent
        
        # Calculate raw quantity based on risk
        raw_quantity = risk_amount / (fee_adjusted_stop_distance * PIP_VALUE) if fee_adjusted_stop_distance > 0 else 0.0
        
        # Calculate maximum quantity by balance (including fees) with a 1% buffer
        cost_per_unit = entry_price * (1/current_leverage + FEE_RATE) * 1.01  # 1% buffer
        max_q_by_balance = balance / cost_per_unit if cost_per_unit > 0 else 0.0
        
        # Also consider leverage limit with 1% buffer
        max_q_by_margin = (balance * current_leverage * 0.99) / entry_price if entry_price > 0 else 0.0  # 1% buffer
        
        # Use the minimum of these values
        quantity = min(raw_quantity, max_q_by_balance, max_q_by_margin)
        quantity = floor_to_step(quantity, stepSize)
    
    # Check notional value (minimum 5 USDC for BTCUSDC)
    notional_value = quantity * entry_price
    min_notional = 5.0
    if notional_value < min_notional:
        # We need to increase the quantity to meet the minimum notional
        min_quantity = min_notional / entry_price
        quantity = ceil_to_step(min_quantity, stepSize)
        # Recalculate costs with the new quantity
        margin_required = (quantity * entry_price) / current_leverage if entry_price > 0 else float('inf')
        opening_fee = quantity * entry_price * FEE_RATE
        total_cost = margin_required + opening_fee
        # Check if we can afford this
        if total_cost > balance:
            logger.info(f"Adjusted quantity to meet notional requirement, but now exceeds balance. Notional: ${notional_value:.4f}, Min Notional: ${min_notional}, New Quantity: {quantity}, Total Cost: ${total_cost:.4f}, Balance: ${balance:.4f}")
            return 0.0, 0.0, 0.0, False
    
    # Calculate margin required for this trade
    margin_required = (quantity * entry_price) / current_leverage if entry_price > 0 else float('inf')
    
    # Calculate opening fee
    opening_fee = quantity * entry_price * FEE_RATE
    
    # Total cost to open the position
    total_cost = margin_required + opening_fee
    
    # Check if we can afford this trade
    can_trade = total_cost <= balance
    
    # For very small balances, we need to ensure we can meet the minimum notional value
    if balance < 5.0:
        notional_value = quantity * entry_price
        if notional_value < 5.0:
            # If we can't meet the minimum notional value, we can't trade
            can_trade = False
            logger.info(f"Notional value ${notional_value:.4f} below minimum $5.00, cannot trade")
    
    # Set dynamic stop-loss and take-profit based on ATR, adjusted for fees
    if action == 'buy':
        stop_loss = entry_price - fee_adjusted_stop_distance
        take_profit = entry_price + fee_adjusted_take_profit_distance
    else:
        stop_loss = entry_price + fee_adjusted_stop_distance
        take_profit = entry_price - fee_adjusted_take_profit_distance

    quantity = float(round(quantity, 8))
    return quantity, stop_loss, take_profit, can_trade

async def update_real_balance(async_client):
    """Update the real balance from the Binance API and convert all assets to USDC."""
    global real_balance, available_balances, last_balance_update_time
    
    if not AUTH or async_client is None:
        return
    
    try:
        # Get futures account balance
        account = await async_client.futures_account_balance()
        
        # Reset available balances
        available_balances = {}
        
        # Find USDC balance
        usdc_balance = next((b for b in account if b['asset'] == 'USDC'), None)
        
        if usdc_balance:
            # Use crossWalletBalance (available for trading in cross margin account)
            if 'crossWalletBalance' in usdc_balance and usdc_balance['crossWalletBalance'] is not None:
                available_balances['USDC'] = float(usdc_balance['crossWalletBalance'])
                # Also set availableBalance to match crossWalletBalance for display purposes
                usdc_balance['availableBalance'] = usdc_balance['crossWalletBalance']
            elif 'marginBalance' in usdc_balance and usdc_balance['marginBalance'] is not None:
                available_balances['USDC'] = float(usdc_balance['marginBalance'])
                # Also set availableBalance to match marginBalance for display purposes
                usdc_balance['availableBalance'] = usdc_balance['marginBalance']
            else:
                logger.warning("USDC balance found but no crossWalletBalance or marginBalance available")
                available_balances['USDC'] = 0.0
        else:
            logger.warning("USDC balance not found in account")
            available_balances['USDC'] = 0.0
        
        # Check for other assets (like BNFCR) and convert them to USDC
        for asset in account:
            asset_symbol = asset['asset']
            if asset_symbol == 'USDC':
                continue  # Already processed
            
            # Get the available balance for this asset
            if 'crossWalletBalance' in asset and asset['crossWalletBalance'] is not None:
                asset_balance = float(asset['crossWalletBalance'])
            elif 'marginBalance' in asset and asset['marginBalance'] is not None:
                asset_balance = float(asset['marginBalance'])
            else:
                continue
            
            # Skip if balance is 0 or very small
            if asset_balance <= 0.00000001:
                continue
            
            # Store the balance
            available_balances[asset_symbol] = asset_balance
            
            # Try to convert to USDC if the asset is not USDC
            try:
                # Get the price of the asset in USDC
                if asset_symbol == 'BTC':
                    ticker = await async_client.futures_ticker(symbol='BTCUSDC')
                    price = float(ticker['lastPrice'])
                    usdc_value = asset_balance * price
                    logger.info(f"Converting {asset_balance} {asset_symbol} to {usdc_value} USDC at price {price}")
                elif asset_symbol == 'ETH':
                    ticker = await async_client.futures_ticker(symbol='ETHUSDC')
                    price = float(ticker['lastPrice'])
                    usdc_value = asset_balance * price
                    logger.info(f"Converting {asset_balance} {asset_symbol} to {usdc_value} USDC at price {price}")
                elif asset_symbol == 'BNB':
                    ticker = await async_client.futures_ticker(symbol='BNBUSDC')
                    price = float(ticker['lastPrice'])
                    usdc_value = asset_balance * price
                    logger.info(f"Converting {asset_balance} {asset_symbol} to {usdc_value} USDC at price {price}")
                else:
                    # For other assets, try to get the price
                    try:
                        ticker = await async_client.futures_ticker(symbol=f'{asset_symbol}USDC')
                        price = float(ticker['lastPrice'])
                        usdc_value = asset_balance * price
                        logger.info(f"Converting {asset_balance} {asset_symbol} to {usdc_value} USDC at price {price}")
                    except:
                        logger.warning(f"Could not get price for {asset_symbol}, skipping conversion")
                        continue
                
                # Add the USDC value to our total
                available_balances['USDC'] += usdc_value
                
            except Exception as e:
                logger.error(f"Error converting {asset_symbol} to USDC: {e}")
        
        # Update the real balance with the total USDC value
        real_balance = available_balances.get('USDC', 0.0)
        logger.info(f"Updated real USDC balance: ${real_balance:.4f}")
        
        # Update last balance update time
        last_balance_update_time = time.time()
        
    except Exception as e:
        logger.error(f"Failed to update real balance: {e}")

async def set_leverage(async_client, leverage=LEVERAGE):
    """Set leverage for the symbol (async)."""
    global current_leverage
    
    if not AUTH or async_client is None:
        logger.info("Skipping leverage setting (not authenticated or client missing).")
        return
    
    try:
        await async_client.futures_change_leverage(symbol=SYMBOL, leverage=leverage)
        current_leverage = leverage
        logger.info(f"Leverage set to {leverage}x for {SYMBOL}")
    except Exception as e:
        logger.error(f"Failed to set leverage: {e}")

async def execute_trade(async_client, action, entry_price, quantity, stop_loss, take_profit, can_trade, df: pd.DataFrame, pattern_name='', intensity=0):
    """
    Execute a real trade (async) or simulate it. Adds trade to open_trades.
    Implements trailing stop for trending markets.
    Fixed to only deduct balance after order is successfully placed.
    """
    global virtual_balance, real_balance, open_trades, trade_count, last_pattern_time, current_leverage
    
    # Check if quantity is 0 and skip if so
    if quantity <= 0:
        logger.info(f"Skipping trade with 0 quantity: {action.upper()} {SYMBOL} at ${entry_price:.2f}")
        # Add more detailed logging to understand why quantity is 0
        balance = real_balance if real_balance > 0 else virtual_balance
        logger.info(f"Debug info - Balance: ${balance:.4f}, Leverage: {current_leverage}x, Entry Price: ${entry_price:.2f}")
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
    trailing_distance = calculate_atr(df) * TRAILING_STOP_ATR_MULTIPLIER if not df.empty else 0.0
    
    # Calculate margin required for this trade
    margin_required = (quantity * entry_price) / current_leverage if entry_price > 0 else 0.0
    
    # Calculate opening fee
    opening_fee = quantity * entry_price * FEE_RATE
    
    # Total cost to open the position
    total_cost = margin_required + opening_fee
    
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
        'opening_fee': opening_fee,
        'leverage_used': current_leverage
    }

    if can_trade and async_client and AUTH:
        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                # Get symbol info for step size
                info = await fetch_symbol_info(async_client, SYMBOL)
                stepSize = info.get('stepSize', DEFAULT_QUANTITY_STEP)
                minQty = info.get('minQty', MIN_ORDER_SIZE)
                
                # Round quantity to proper step size to avoid precision errors
                rounded_quantity = floor_to_step(quantity, stepSize)
                
                # Ensure quantity meets minimum requirement
                if rounded_quantity < minQty:
                    rounded_quantity = minQty
                
                # Calculate notional value
                notional_value = rounded_quantity * entry_price
                min_notional = 5.0  # Minimum notional for BTCUSDC
                
                # Check if notional value meets minimum requirement
                if notional_value < min_notional:
                    # Calculate minimum quantity needed to meet notional requirement
                    min_quantity = min_notional / entry_price
                    rounded_quantity = ceil_to_step(min_quantity, stepSize)
                    
                    # Recalculate costs with new quantity
                    margin_required = (rounded_quantity * entry_price) / current_leverage if entry_price > 0 else 0.0
                    opening_fee = rounded_quantity * entry_price * FEE_RATE
                    new_total_cost = margin_required + opening_fee
                    
                    # Check if we can afford this
                    balance = real_balance if real_balance > 0 else virtual_balance
                    if new_total_cost > balance:
                        logger.info(f"Adjusted quantity to meet notional requirement, but now exceeds balance. Notional: ${notional_value:.4f}, Min Notional: ${min_notional}, New Quantity: {rounded_quantity}, Total Cost: ${new_total_cost:.4f}, Balance: ${balance:.4f}")
                        # Mark as simulated and break
                        trade_info['is_simulated'] = True
                        break
                
                # Format quantity to avoid precision errors - ensure it has no more than 6 decimal places
                formatted_quantity = float(f"{rounded_quantity:.6f}")
                
                side_open = 'BUY' if action == 'buy' else 'SELL'
                
                # Place the order
                order_result = await async_client.futures_create_order(
                    symbol=SYMBOL,
                    side=side_open,
                    type='MARKET',
                    quantity=str(formatted_quantity)
                )
                
                # Only deduct balance after order is successfully placed
                real_balance -= total_cost
                logger.info(f"Order placed successfully. Deducted ${total_cost:.4f} from balance. Remaining balance: ${real_balance:.4f}")
                
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
                        quantity=str(formatted_quantity)
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
                    quantity=str(formatted_quantity)
                )
                
                logger.info(f"Real trade placed: {action.upper()} {formatted_quantity} {SYMBOL} at {entry_price:.2f} | SL {stop_loss:.2f} TP {take_profit:.2f} | margin used: ${margin_required:.4f} | fee: ${opening_fee:.4f} | Real Balance: ${real_balance:.4f}")
                success = True
                break
            except Exception as e:
                logger.error(f"Real trade execution failed (attempt {attempt+1}/{max_retries}): {e}")
                
                # Handle specific errors
                if "Margin is insufficient" in str(e) and attempt < max_retries - 1:
                    # Calculate maximum affordable quantity based on available balance
                    balance = real_balance if real_balance > 0 else virtual_balance
                    
                    # Calculate maximum quantity by balance (including fees)
                    cost_per_unit = entry_price * (1/current_leverage + FEE_RATE)
                    max_q_by_balance = balance / cost_per_unit if cost_per_unit > 0 else 0.0
                    
                    # Also consider leverage limit
                    max_q_by_margin = (balance * current_leverage) / entry_price if entry_price > 0 else 0.0
                    
                    # Use the minimum of these values
                    new_quantity = min(max_q_by_balance, max_q_by_margin)
                    
                    # Round to step size
                    info = await fetch_symbol_info(async_client, SYMBOL)
                    stepSize = info.get('stepSize', DEFAULT_QUANTITY_STEP)
                    new_quantity = floor_to_step(new_quantity, stepSize)
                    
                    # Ensure minimum quantity
                    minQty = info.get('minQty', MIN_ORDER_SIZE)
                    if new_quantity < minQty:
                        new_quantity = minQty
                    
                    # Check notional value
                    notional_value = new_quantity * entry_price
                    min_notional = 5.0
                    if notional_value < min_notional:
                        # Calculate minimum quantity needed to meet notional requirement
                        min_quantity = min_notional / entry_price
                        new_quantity = ceil_to_step(min_quantity, stepSize)
                    
                    # Recalculate margin and fees
                    margin_required = (new_quantity * entry_price) / current_leverage if entry_price > 0 else 0.0
                    opening_fee = new_quantity * entry_price * FEE_RATE
                    new_total_cost = margin_required + opening_fee
                    
                    logger.info(f"Retrying with recalculated quantity: {new_quantity}, new total cost: {new_total_cost}")
                    quantity = new_quantity
                    total_cost = new_total_cost
                elif "Precision is over the maximum defined for this asset" in str(e) and attempt < max_retries - 1:
                    # Get symbol info for step size
                    info = await fetch_symbol_info(async_client, SYMBOL)
                    stepSize = info.get('stepSize', DEFAULT_QUANTITY_STEP)
                    
                    # Round quantity to proper step size
                    new_quantity = floor_to_step(quantity, stepSize)
                    
                    # Ensure minimum quantity
                    minQty = info.get('minQty', MIN_ORDER_SIZE)
                    if new_quantity < minQty:
                        new_quantity = minQty
                    
                    # Recalculate margin and fees
                    margin_required = (new_quantity * entry_price) / current_leverage if entry_price > 0 else 0.0
                    opening_fee = new_quantity * entry_price * FEE_RATE
                    new_total_cost = margin_required + opening_fee
                    
                    logger.info(f"Retrying with rounded quantity: {new_quantity}, new total cost: {new_total_cost}")
                    quantity = new_quantity
                    total_cost = new_total_cost
                else:
                    # Trade failed, mark as simulated
                    trade_info['is_simulated'] = True
                    break
        
        if success:
            # Update trade_info with the final quantity and costs
            trade_info['quantity'] = quantity
            trade_info['margin_used'] = margin_required
            trade_info['opening_fee'] = opening_fee
    else:
        reason = "not authorized" if not AUTH else "insufficient margin or size"
        logger.info(f"Simulating trade due to {reason}: {action.upper()} {quantity} {SYMBOL} at ${entry_price:.2f}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f} | fee: ${opening_fee:.4f} | Virtual Balance: ${virtual_balance:.4f}")
        
        # For simulated trades, deduct from virtual balance
        virtual_balance -= total_cost

    open_trades.append(trade_info)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ENTRY -> {action.upper():4} | Qty {quantity:.6f} | Price {entry_price:.2f} | SL {stop_loss:.2f} | TP {take_profit:.2f} | Simulated: {trade_info['is_simulated']} | Pattern: {pattern_name} | Real Balance: ${real_balance:.4f} | Virtual Balance: ${virtual_balance:.4f}")
    
    trade_count += 1

def send_webhook(action, entry_price, stop_loss, take_profit, quantity, is_simulated):
    """Send webhook (synchronous)."""
    if not WEBHOOK_URL:
        return
    payload = {
        'action': action,
        'symbol': SYMBOL,
        'leverage': current_leverage,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'quantity': quantity,
        'is_simulated': is_simulated,
        'timeframe': f"{TIMEFRAME_SECONDS}s",
        'timestamp': datetime.now().isoformat(),
        'market_regime': market_regime,
        'volatility_state': volatility_state,
        'balance': real_balance if real_balance > 0 else virtual_balance
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
    global virtual_balance, real_balance, open_trades, trade_count
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
        opening_fee = trade.get('opening_fee', 0.0)

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
                # Calculate gross PnL
                gross_pnl = (stop_loss - entry_price) * quantity * PIP_VALUE
                # Calculate closing fee
                closing_fee = quantity * stop_loss * FEE_RATE
                # Net PnL = gross PnL - opening_fee - closing_fee
                pnl = gross_pnl - opening_fee - closing_fee
                closed.append(trade)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT  -> BUY closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Real Balance: ${real_balance:.4f} | Virtual Balance: ${virtual_balance:.4f}")
                # Update balances
                if AUTH and real_balance > 0 and not is_simulated:
                    # Update real balance for real trades
                    real_balance += margin_used + pnl
                else:
                    # Update virtual balance for simulated trades
                    virtual_balance += margin_used + pnl
            elif current_price >= take_profit:
                exit_price = take_profit
                gross_pnl = (take_profit - entry_price) * quantity * PIP_VALUE
                closing_fee = quantity * take_profit * FEE_RATE
                pnl = gross_pnl - opening_fee - closing_fee
                closed.append(trade)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT  -> BUY closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Real Balance: ${real_balance:.4f} | Virtual Balance: ${virtual_balance:.4f}")
                if AUTH and real_balance > 0 and not is_simulated:
                    real_balance += margin_used + pnl
                else:
                    virtual_balance += margin_used + pnl
        else:  # sell
            if current_price >= stop_loss:
                exit_price = stop_loss
                gross_pnl = (entry_price - stop_loss) * quantity * PIP_VALUE
                closing_fee = quantity * stop_loss * FEE_RATE
                pnl = gross_pnl - opening_fee - closing_fee
                closed.append(trade)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT  -> SELL closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Real Balance: ${real_balance:.4f} | Virtual Balance: ${virtual_balance:.4f}")
                if AUTH and real_balance > 0 and not is_simulated:
                    real_balance += margin_used + pnl
                else:
                    virtual_balance += margin_used + pnl
            elif current_price <= take_profit:
                exit_price = take_profit
                gross_pnl = (entry_price - take_profit) * quantity * PIP_VALUE
                closing_fee = quantity * take_profit * FEE_RATE
                pnl = gross_pnl - opening_fee - closing_fee
                closed.append(trade)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT  -> SELL closed at ${exit_price:.2f} | PnL ${pnl:.4f} | Real Balance: ${real_balance:.4f} | Virtual Balance: ${virtual_balance:.4f}")
                if AUTH and real_balance > 0 and not is_simulated:
                    real_balance += margin_used + pnl
                else:
                    virtual_balance += margin_used + pnl
        
        # Record closed trade
        if exit_price is not None:
            trade['exit_price'] = exit_price
            trade['pnl'] = pnl
            trade_count += 1

    open_trades[:] = [t for t in open_trades if t not in closed]

# ---------------------------
# Backtesting Functionality
# ---------------------------
async def run_backtest(async_client):
    """Run backtest using historical data."""
    global virtual_balance, real_balance, backtest_results, candle_data, price_history, lrc_prices, current_leverage, trade_count
    
    logger.info(f"Starting backtest for the last {BACKTEST_DAYS} days...")
    
    # Reset state
    virtual_balance = INITIAL_BALANCE
    real_balance = 0.0
    open_trades.clear()
    backtest_results.clear()
    candle_data.clear()
    price_history.clear()
    lrc_prices.clear()
    current_leverage = LEVERAGE
    trade_count = 0
    
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
        current_price = float(trade['p'])
        signals = detect_patterns_with_intensity(df, current_price) if not df.empty else []
        
        # Execute signals
        for action, entry_price, intensity_pct, pname in signals:
            quantity, stop_loss, take_profit, can_trade = await calculate_quantity_and_levels(entry_price, action, async_client, df)
            await execute_trade(async_client, action, entry_price, quantity, stop_loss, take_profit, False, df, pname, intensity_pct)
        
        # Track trades on current price
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
        
        logger.info(f"Backtest completed. Total trades: {total_trades}, Win rate: {win_rate:.2f}%, Total PnL: ${total_pnl:.4f}, Avg PnL: ${avg_pnl:.4f}, Final Balance: ${virtual_balance:.4f}")
    else:
        logger.info("Backtest completed. No trades were executed.")

# ---------------------------
# Enhanced message processing
# ---------------------------
async def process_message(message, async_client):
    """Process a single trade message with error handling."""
    global processing_stats, candle_data, last_heartbeat_time, price_history
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
                    
                    # If quantity is 0, try with higher leverage up to the maximum allowed
                    if quantity == 0 and current_leverage < max_allowed_leverage:
                        logger.info(f"Quantity is 0 with {current_leverage}x leverage, trying with {max_allowed_leverage}x leverage")
                        await set_leverage(async_client, max_allowed_leverage)
                        # Recalculate with higher leverage
                        quantity, stop_loss, take_profit, can_trade = await calculate_quantity_and_levels(entry_price, action, async_client, df)
                    
                    # If quantity is still 0, provide detailed logging
                    if quantity == 0:
                        balance = real_balance if real_balance > 0 else virtual_balance
                        logger.info(f"Signal detected but quantity is 0. Debug info: Action={action}, Entry Price=${entry_price:.2f}, Balance=${balance:.4f}, Leverage={current_leverage}x")
                        
                        # Calculate the minimum balance needed for this trade
                        info = await fetch_symbol_info(async_client, SYMBOL)
                        min_qty = info.get('minQty', MIN_ORDER_SIZE)
                        min_notional = min_qty * entry_price
                        min_balance_needed = min_notional / max_allowed_leverage
                        logger.info(f"Minimum balance needed for this trade: ${min_balance_needed:.4f} (with {max_allowed_leverage}x leverage)")
                        continue
                    
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
    """Listen to WebSocket messages and handle connection issues."""
    global ws_connected, ws_manager, ws_socket, ws_active, ws_reconnect_attempts, last_heartbeat_time
    
    ws_symbol = SYMBOL.lower()
    
    while ws_reconnect_attempts < WS_MAX_RECONNECT_ATTEMPTS:
        try:
            # Create a new BinanceSocketManager
            ws_manager = BinanceSocketManager(async_client)
            
            # Create the trade socket
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
                        # Check if queue is getting full
                        queue_size = message_queue.qsize()
                        if queue_size > WS_QUEUE_SIZE * 0.8:
                            # Queue is getting full, log warning
                            logger.warning(f"Message queue is {queue_size}/{WS_QUEUE_SIZE} ({queue_size/WS_QUEUE_SIZE*100:.1f}%) full")
                            
                            # If queue is almost full, drop the message to prevent overflow
                            if queue_size > WS_QUEUE_SIZE * 0.95:
                                processing_stats['overflows'] += 1
                                logger.warning(f"Message queue nearly full, dropping message. Overflows: {processing_stats['overflows']}")
                                continue
                        
                        message_queue.put_nowait(trade)
                    except asyncio.QueueFull:
                        # Queue is full, log warning and drop message
                        processing_stats['overflows'] += 1
                        logger.warning(f"Message queue full, dropping message. Overflows: {processing_stats['overflows']}")
                    
                    # Periodically log processing statistics
                    if processing_stats['messages_received'] % 1000 == 0:
                        queue_size = message_queue.qsize()
                        logger.info(f"Processing statistics: Received={processing_stats['messages_received']}, "
                                  f"Processed={processing_stats['messages_processed']}, "
                                  f"Overflows={processing_stats['overflows']}, "
                                  f"QueueSize={queue_size}, "
                                  f"AvgProcessTime={processing_stats['processing_time'] / max(1, processing_stats['messages_processed']):.4f}s")
                
                except asyncio.TimeoutError:
                    # No message received within timeout, check if we need to reconnect
                    if time.time() - last_heartbeat_time > WS_HEARTBEAT_INTERVAL:
                        logger.warning(f"No messages received for {WS_HEARTBEAT_INTERVAL} seconds, reconnecting...")
                        break
                    continue
                
                except BinanceWebsocketQueueOverflow as e:
                    # Handle the specific overflow exception
                    processing_stats['overflows'] += 1
                    logger.error(f"Binance WebSocket queue overflow: {e}. Overflows: {processing_stats['overflows']}")
                    
                    # Sleep briefly to allow the queue to drain
                    await asyncio.sleep(0.1)
                    
                    # If we get too many overflows, reconnect
                    if processing_stats['overflows'] % 10 == 0:
                        logger.warning("Multiple WebSocket overflows detected, reconnecting...")
                        break
                
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break
            
            # Clean up the socket
            try:
                await ws_socket.__aexit__(None, None, None)
            except Exception:
                pass
            
            # Clean up the socket manager
            try:
                await ws_manager.close()
            except Exception:
                pass
            
            ws_connected = False
            
            # Wait before reconnecting
            if ws_active and ws_reconnect_attempts < WS_MAX_RECONNECT_ATTEMPTS:
                logger.info(f"Waiting {WS_RECONNECT_DELAY} seconds before reconnecting...")
                await asyncio.sleep(WS_RECONNECT_DELAY)
        
        except Exception as e:
            logger.error(f"Error creating WebSocket connection: {e}")
            ws_connected = False
            
            # Clean up the socket manager if it exists
            if ws_manager:
                try:
                    await ws_manager.close()
                except Exception:
                    pass
            
            # Wait before reconnecting
            if ws_reconnect_attempts < WS_MAX_RECONNECT_ATTEMPTS:
                logger.info(f"Waiting {WS_RECONNECT_DELAY} seconds before reconnecting...")
                await asyncio.sleep(WS_RECONNECT_DELAY)
        
        finally:
            ws_reconnect_attempts += 1
    
    # If we've exhausted all reconnection attempts
    if ws_reconnect_attempts >= WS_MAX_RECONNECT_ATTEMPTS:
        logger.error(f"Maximum WebSocket reconnection attempts ({WS_MAX_RECONNECT_ATTEMPTS}) reached. Giving up.")
        ws_connected = False
        ws_active = False

async def process_trade_stream(async_client):
    """
    Consume trade websocket, queue messages, and process them asynchronously.
    """
    global processing_task, ws_active
    
    # Run backtest if enabled
    if BACKTEST_MODE:
        await run_backtest(async_client)
        return
    
    # Start the message processor task
    processing_task = asyncio.create_task(message_processor(async_client))
    
    # Start the WebSocket listener
    ws_active = True
    websocket_task = asyncio.create_task(websocket_listener(async_client))
    
    try:
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [processing_task, websocket_task],
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
            # Show real balance if available, otherwise virtual balance
            active_balance = real_balance if real_balance > 0 else virtual_balance
            print(f"[{ts}] No candle yet | ForecastΔ: {forecast_delta:+.4f} | ForecastPrice: {forecast_price:.2f} | Signals: none | Active Balance: ${active_balance:.4f}")
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
        
        # Get FFT cycle info
        fft_period, fft_phase = fft_cycle_detection(price_history)
        fft_text = f"FFT Period: {fft_period:.1f}" if fft_period else "FFT: none"
        
        # Get LRC info
        lrc_text = f"LRC: {lrc_signal}" if lrc_signal else "LRC: neutral"
        if lrc_middle is not None and lrc_upper is not None and lrc_lower is not None:
            lrc_text += f" (M:{lrc_middle:.2f} U:{lrc_upper:.2f} L:{lrc_lower:.2f})"
        
        # Show real balance if available, otherwise virtual balance
        active_balance = real_balance if real_balance > 0 else virtual_balance
        
        print(f"[{ts}] Price: {price:.2f} | Vol: {vol:.6f} | Mom: {momentum:.4f} | EMAf: {ema_f:.2f} | EMAs: {ema_s:.2f} | StochK: {stoch_k:.1f} | Regime: {market_regime} | Volatility: {volatility_state} | {arctanh_text} | {fft_text} | {lrc_text} | ForecastΔ: {forecast_delta:+.4f} | Forecast: {forecast_price:.2f} | Signals: {sig_text} | Open: {open_count} | Active Balance: ${active_balance:.4f} | Leverage: {current_leverage}x")
    except Exception as e:
        logger.debug(f"print_main_status error: {e}")

# ---------------------------
# Main
# ---------------------------
async def main():
    global bot_start_time, virtual_balance, real_balance, current_leverage, max_allowed_leverage, trade_count
    bot_start_time = time.time()  # Record when the bot starts
    
    async_client = None
    try:
        if AUTH:
            async_client = await AsyncClient.create(API_KEY, API_SECRET)
            # Fetch real balance if authenticated
            await update_real_balance(async_client)
            
            # If real balance is available, use it as starting virtual balance
            if real_balance > 0:
                virtual_balance = real_balance
                logger.info(f"Using real balance for trading: ${real_balance:.4f} USDC")
            else:
                logger.warning("No real USDC balance found, using virtual balance for simulation")
        else:
            async_client = await AsyncClient.create()
            logger.info("No API credentials found, using virtual balance for simulation")

        # Get symbol info to determine maximum allowed leverage
        if AUTH and async_client:
            await fetch_symbol_info(async_client, SYMBOL)
            logger.info(f"Maximum allowed leverage for {SYMBOL}: {max_allowed_leverage}x")
        
        # Set initial leverage to maximum allowed
        await set_leverage(async_client, max_allowed_leverage)
        
        await process_trade_stream(async_client)

    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
    finally:
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

# Entrypoint
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")