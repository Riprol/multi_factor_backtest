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

# 鈹€鈹€ 鑲＄エ姹狅細娌繁 300 鎴愬垎鑲¤繃婊?鈹€鈹€
INDEX_UNIVERSE   = "000300.SH"   # 鎸囨暟浠ｇ爜锛堟勃娣?00锛夛紝None 琛ㄧず鍏ㄥ競鍦?INDEX_WEIGHT_MIN = 0.0           # 鏈€灏忔潈閲嶉槇鍊硷紙0=鍏ㄩ儴绾冲叆锛?
plt_style   = "seaborn-v0_8-darkgrid"
FONT_FAMILY = "Microsoft YaHei"
FIG_DPI     = 150

# ======================== 鍔熻兘寮€鍏?========================
ENABLE_DOWNLOAD          = True   # 涓嬭浇琛屾儏鏁版嵁
ENABLE_FACTOR_COMPUTE    = True   # 璁＄畻鍥犲瓙鍊?ENABLE_IC_ANALYSIS       = True   # IC 鍒嗘瀽
ENABLE_BACKTEST          = True   # 鍒嗗眰鍥炴祴 + 缁勫悎鏋勫缓
ENABLE_COMPOSITE         = True   # 澶氬洜瀛愬悎鎴?ENABLE_FACTOR_PREPROCESS = True   # 缂╁熬 + 鏍囧噯鍖?+ 涓€у寲
ENABLE_RISK_CONTROL      = True   # 閫夎偂杩囨护 + 姝㈡崯 + 鐔旀柇
ENABLE_CHARTS            = True   # 淇濆瓨鍥捐〃
ENABLE_REPORT            = True   # 鐢熸垚 Markdown 鎶ュ憡
