"""
Configuration for Portfolio Optimization System
"""
import logging
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Logging
LOG_FILE = LOGS_DIR / "portfolio_system.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Portfolio parameters for India
DEFAULT_RISK_FREE_RATE = 0.06  # 6% for India (higher than US)
TRADING_DAYS_PER_YEAR = 252

# ML parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5

# Optimization constraints
DEFAULT_MAX_WEIGHT = 0.40
DEFAULT_MIN_WEIGHT = 0.0
ALLOW_SHORT_SELLING = False

# Monte Carlo
N_PORTFOLIOS = 5000

# Stress scenarios
STRESS_SCENARIOS = {
    "market_crash": -0.10,
    "volatility_spike": 1.5
}

# Feature engineering
LOOKBACK_PERIODS = [5, 10, 20, 50]
VOLATILITY_WINDOW = 20

# Indian stock universe (NSE)
INDIAN_STOCKS = {
    'Technology': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS'],
    'Auto': ['TATAMOTORS.NS', 'M&M.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS'],
    'FMCG': ['ITC.NS', 'HINDUNILVR.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'AUROPHARMA.NS'],
    'Metals': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'COALINDIA.NS'],
    'Cement': ['ULTRACEMCO.NS', 'AMBUJACEM.NS', 'SHREECEM.NS', 'ACC.NS', 'JKCEMENT.NS']
}
