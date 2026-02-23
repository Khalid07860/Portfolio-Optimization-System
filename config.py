"""
Configuration - EXPANDED with 100+ NSE stocks
"""
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Logging
LOG_FILE = LOGS_DIR / "portfolio_system.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Portfolio parameters
DEFAULT_RISK_FREE_RATE = 0.06  # 6% for India
TRADING_DAYS_PER_YEAR = 252

# ML parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5

# Optimization
DEFAULT_MAX_WEIGHT = 0.35  # 35% max
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

# EXPANDED Indian Stock Universe - 100+ stocks
INDIAN_STOCKS = {
    'Technology': [
        'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
        'LTI.NS', 'COFORGE.NS', 'MPHASIS.NS', 'PERSISTENT.NS', 'LTTS.NS'
    ],
    'Banking': [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
        'INDUSINDBK.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'AUBANK.NS',
        'PNB.NS', 'BANKBARODA.NS', 'CANBK.NS'
    ],
    'Financial Services': [
        'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'ICICIPRULI.NS',
        'HDFC.NS', 'ICICIGI.NS', 'SBICARD.NS', 'CHOLAFIN.NS', 'MUTHOOTFIN.NS',
        'PFC.NS', 'RECLTD.NS', 'LICHSGFIN.NS'
    ],
    'Energy & Power': [
        'RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS',
        'POWERGRID.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS', 'TATAPOWER.NS', 'GAIL.NS',
        'HINDPETRO.NS', 'ADANITRANS.NS', 'TORNTPOWER.NS'
    ],
    'Automobiles': [
        'TATAMOTORS.NS', 'M&M.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS',
        'EICHERMOT.NS', 'ASHOKLEY.NS', 'TVSMOTOR.NS', 'BALKRISIND.NS', 'MRF.NS',
        'APOLLOTYRE.NS', 'BHARATFORG.NS', 'MOTHERSON.NS', 'EXIDEIND.NS'
    ],
    'FMCG': [
        'ITC.NS', 'HINDUNILVR.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS',
        'GODREJCP.NS', 'MARICO.NS', 'TATACONSUM.NS', 'COLPAL.NS', 'EMAMILTD.NS',
        'UBL.NS', 'MCDOWELL-N.NS', 'PGHH.NS', 'VBL.NS'
    ],
    'Pharmaceuticals': [
        'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'AUROPHARMA.NS',
        'LUPIN.NS', 'ALKEM.NS', 'TORNTPHARM.NS', 'BIOCON.NS', 'ZYDUSLIFE.NS',
        'IPCALAB.NS', 'LALPATHLAB.NS', 'APOLLOHOSP.NS', 'MAXHEALTH.NS'
    ],
    'Metals & Mining': [
        'TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'COALINDIA.NS',
        'SAIL.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'HINDZINC.NS', 'NATIONALUM.NS',
        'MOIL.NS', 'RATNAMANI.NS'
    ],
    'Cement': [
        'ULTRACEMCO.NS', 'AMBUJACEM.NS', 'SHREECEM.NS', 'ACC.NS', 'JKCEMENT.NS',
        'DALMIACEM.NS', 'RAMCOCEM.NS', 'HEIDELBERG.NS', 'INDIACEM.NS'
    ],
    'Consumer Durables': [
        'TITAN.NS', 'HAVELLS.NS', 'VOLTAS.NS', 'WHIRLPOOL.NS', 'BLUESTARCO.NS',
        'CROMPTON.NS', 'DIXON.NS', 'AMBER.NS', 'BATAINDIA.NS', 'RELAXO.NS'
    ],
    'Real Estate': [
        'DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PHOENIXLTD.NS', 'PRESTIGE.NS',
        'BRIGADE.NS', 'SOBHA.NS', 'LODHA.NS'
    ],
    'Telecom': [
        'BHARTIARTL.NS', 'INDUSINDBK.NS', 'MTNL.NS'
    ],
    'Retail': [
        'DMART.NS', 'TRENT.NS', 'ABFRL.NS', 'SHOPERSTOP.NS', 'TEJASNET.NS'
    ],
    'Infrastructure': [
        'LT.NS', 'ADANIPORTS.NS', 'GMRINFRA.NS', 'IRCTC.NS', 'CONCOR.NS',
        'CUMMINSIND.NS', 'SIEMENS.NS', 'ABB.NS', 'THERMAX.NS'
    ],
    'Media & Entertainment': [
        'ZEEL.NS', 'SUNTV.NS', 'PVRINOX.NS', 'NETWORK18.NS'
    ],
    'Chemicals': [
        'UPL.NS', 'PIDILITIND.NS', 'SRF.NS', 'AARTI.NS', 'DEEPAKNTR.NS',
        'TATACHEM.NS', 'GNFC.NS', 'CHAMBLFERT.NS'
    ]
}

def get_all_stocks():
    """Get all stocks from the universe"""
    all_stocks = []
    for stocks in INDIAN_STOCKS.values():
        all_stocks.extend(stocks)
    return list(set(all_stocks))  # Remove duplicates

# Total stocks available
TOTAL_STOCKS = len(get_all_stocks())
