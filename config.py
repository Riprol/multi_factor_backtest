import os

TUSHARE_TOKEN = "222f4303c39de592551042cecca697e20473505d9a046412c4305b8c"

START_DATE = "20230101"
END_DATE   = "20260101"

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

# ── 股票池：沪深 300 成分股过滤 ──
INDEX_UNIVERSE   = "000300.SH"   # 指数代码（沪深300），None 表示全市场
INDEX_WEIGHT_MIN = 0.0           # 最小权重阈值（0=全部纳入）

plt_style   = "seaborn-v0_8-darkgrid"
FONT_FAMILY = "Microsoft YaHei"
FIG_DPI     = 150

# ======================== 功能开关 ========================
ENABLE_DOWNLOAD          = True   # 下载行情数据
ENABLE_FACTOR_COMPUTE    = True   # 计算因子值
ENABLE_IC_ANALYSIS       = True   # IC 分析
ENABLE_BACKTEST          = True   # 分层回测 + 组合构建
ENABLE_COMPOSITE         = True   # 多因子合成
ENABLE_FACTOR_PREPROCESS = True   # 缩尾 + 标准化 + 中性化
ENABLE_RISK_CONTROL      = True   # 选股过滤 + 止损 + 熔断
ENABLE_CHARTS            = True   # 保存图表
ENABLE_REPORT            = True   # 生成 Markdown 报告
