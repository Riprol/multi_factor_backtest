import os

TUSHARE_TOKEN = "your tushare token"

START_DATE = "start date in YYYYMMDD format"
END_DATE   = "end date in YYYYMMDD format"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DB_PATH    = os.path.join(DATA_DIR, "factor_db.sqlite")
CAL_FILE   = os.path.join(DATA_DIR, "trade_cal.csv")
CACHE_DIR  = os.path.join(DATA_DIR, "cache")

DROP_ST = True
MIN_PRICE = 1.0
DROP_LIMIT = True
MIN_LIST_DAYS = 60

DEFAULT_WINDOW = 20

LAYER_NUM    = 5
COMMISSION   = 0.0003
SLIPPAGE     = 0.001

TOP_N        = 30
REBALANCE_FREQ = "M"
WEIGHT_METHOD = "amount"

RISK_MIN_PRICE     = 1.0
RISK_MIN_VOLUME    = 1000
RISK_MIN_AMOUNT    = 100000
RISK_BLOCK_BOARDS  = ("688",)
RISK_STOP_LOSS     = -0.15
RISK_MAX_DRAWDOWN  = -0.25
RISK_MAX_DAILY_LOSS = -0.10

# Stock universe: CSI 300 (None = all A-shares)
INDEX_UNIVERSE   = "000300.SH"
INDEX_WEIGHT_MIN = 0.0

plt_style   = "seaborn-v0_8-darkgrid"
FONT_FAMILY = "Microsoft YaHei"
FIG_DPI     = 150

# ========== Feature Toggles ==========
ENABLE_DOWNLOAD          = True   # download market data
ENABLE_FACTOR_COMPUTE    = True   # compute factor values
ENABLE_IC_ANALYSIS       = True   # IC analysis
ENABLE_BACKTEST          = True   # layer + portfolio backtest
ENABLE_COMPOSITE         = True   # multi-factor composite
ENABLE_FACTOR_PREPROCESS = True   # winsorize + zscore + neutralize
ENABLE_RISK_CONTROL      = True   # pre-filter + stop-loss + circuit-breaker
ENABLE_CHARTS            = True   # save charts
ENABLE_REPORT            = True   # generate markdown report
