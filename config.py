import os

TUSHARE_TOKEN = "your tushare token"

START_DATE = "start date"
END_DATE   = "end date"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DB_PATH    = os.path.join(DATA_DIR, "factor_db.sqlite")
CAL_FILE   = os.path.join(DATA_DIR, "trade_cal.csv")
CACHE_DIR  = os.path.join(DATA_DIR, "cache")

DROP_ST = True
MIN_PRICE = 1.0
DROP_LIMIT = True

DEFAULT_WINDOW = 20

LAYER_NUM    = 5
COMMISSION   = 0.0003
SLIPPAGE     = 0.001

TOP_N        = 50
REBALANCE_FREQ = "M"
WEIGHT_METHOD = "equal"

plt_style   = "seaborn-v0_8-darkgrid"
FONT_FAMILY = "Microsoft YaHei"
FIG_DPI     = 150
