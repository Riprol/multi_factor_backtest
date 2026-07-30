# 多因子回测框架 (Multi-Factor Backtest Framework)

基于 Tushare 免费接口的 A 股多因子回测系统，一键从数据下载到报告输出。

## 特性

- **零门槛数据获取**：仅需 Tushare 免费 Token，增量下载 + SQLite 持久化，同日期永不重复下载
- **完整数据清洗**：剔除停牌、涨跌停、重复值，创业板/科创板自适应涨跌幅，沪深 300 成分股过滤
- **因子即插即用**：继承 `BaseFactor` 实现 `compute()`，注册后自动缩尾→标准化→行业中性化，接入全流程
- **因子预处理三合一**：截面 Winsorize(1%/99%) → Z-score → 行业+市值中性化(多元OLS)
- **统计检验**：IC 均值 95% 置信区间、单侧 t 检验(p值)、偏度/峰度正态性核验、IC QQ图
- **风控系统**：选股前流动性过滤 + 持仓个股止损 + 组合回撤熔断
- **Numba 加速**：复杂滚动指标（量价相关、滚动斜率）的纯 numpy 编译加速，比纯 Python 快 20 倍
- **SQLite 因子仓库**：窄表结构 `(factor_name, ts_code, trade_date, value)`，复合索引，支持截面查询
- **专业可视化**：分层净值、IC 时序/QQ图、分层收益箱线图、多空回撤、基准对比
- **Markdown 报告**：IC 汇总(含 CI+t 检验)、分年 IC、因子相关性矩阵、风控统计、分层概要

## 项目结构

```
multi_factor_backtest/
├── config.py              ← 全局配置（Token、日期、功能开关、风控参数）
├── main.py                ← 主入口，一键运行
│
├── data/                  ← 数据层
│   ├── calendar.py        # 本地交易日历
│   ├── downloader.py      # Tushare 下载器（增量+SQLite优先）
│   └── cleaner.py         # 数据清洗 + 指数成分股过滤
│
├── factors/               ← 因子层（核心扩展点）
│   ├── base.py            # 因子基类 + 三合一预处理
│   ├── registry.py        # 因子注册表 + 批量计算
│   ├── momentum.py        # 动量因子
│   ├── reversal.py        # 反转因子
│   ├── volatility.py      # 低波动因子
│   ├── turnover.py        # 换手率因子
│   ├── liquidity.py       # 流动性因子(Amihud)
│   ├── price_volume_corr.py  # 量价协同因子(numba示例)
│   └── composite.py       # 多因子合成器(等权/IC加权)
│
├── backtest/              ← 回测层
│   ├── ic_analysis.py     # IC 分析（含 CI + t 检验）
│   ├── layer_backtest.py  # 分层回测（含多空波动率+回撤）
│   ├── portfolio.py       # 组合构建（金额加权+风控）
│   ├── performance.py     # 绩效评估（夏普/回撤/胜率）
│   └── risk_control.py    # 风控（选股过滤+止损+熔断）
│
├── visualization/         ← 可视化层
│   ├── charts.py          # 图表（净值/IC/QQ/箱线/回撤）
│   └── report.py          # Markdown 报告生成
│
├── utils/                 ← 工具层
│   ├── database.py        # SQLite（增量存储+下载日志）
│   ├── stats_tools.py     # 统计（缩尾/Z/IC/t检验/CI/正态性）
│   ├── regression.py      # 回归（OLS/哑变量/行业中性化）
│   ├── rolling_kernels.py # Numba加速（滚动相关/斜率/排名）
│   └── helpers.py         # 计时器/目录工具
│
└── output/                ← 输出（图表 + 报告）
```

## 快速开始

### 1. 获取 Tushare Token

前往 [Tushare Pro](https://tushare.pro/) 注册账号 → 个人主页 → 复制接口 TOKEN。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置参数

编辑 `config.py`：

```python
TUSHARE_TOKEN = "你的token"
START_DATE = "20230101"
END_DATE   = "20260101"
INDEX_UNIVERSE = "000300.SH"   # 沪深300，None=全市场
```

### 4. 运行

```bash
python main.py
```

全自动流程：**下载 → 清洗 → 因子计算 → 缩尾+标准化+中性化 → IC分析 → 分层回测 → 组合构建 → 多因子合成 → 报告输出**。

## 功能开关

`config.py` 底部可单独关闭模块：

```python
ENABLE_DOWNLOAD          = True   # 下载行情
ENABLE_FACTOR_PREPROCESS = True   # 缩尾+标准化+中性化
ENABLE_RISK_CONTROL      = True   # 选股过滤+止损+熔断
ENABLE_CHARTS            = True   # 保存图表
ENABLE_REPORT            = True   # 生成报告
```

## 输出说明

所有结果在 `output/` 目录：

| 文件 | 说明 |
|------|------|
| `layer_net_*.png` | 分层累计净值曲线 |
| `ic_*.png` | IC 时序 + 分布直方图 |
| `ic_dist_*.png` | IC QQ 图（正态性检验） |
| `boxplot_*.png` | 分层收益箱线图 |
| `nav_*.png` | 策略净值 + 回撤曲线（含基准） |
| `ic_comparison.png` | 多因子 IC 箱线对比 |
| `factor_corr.png` | 因子截面相关性热力图 |
| `backtest_report.md` | Markdown 完整报告 |

### 如何判断因子好坏？

- **Rank IC 均值**：> 0.02 不错，> 0.05 优秀
- **ICIR**：> 0.3 可用，> 0.5 良好
- **t 检验 p 值**：< 0.05 统计显著，IC 非随机
- **95% CI 不含 0**：因子溢价可靠
- **分层净值单调性**：5 条线从低到高排列清晰
- **多空最大回撤**：< 20% 可接受
- **多空净值**：稳健向上

## 自定义因子（三步）

### 示例：添加 RSI 因子

**第一步** —— 在 `factors/` 下新建 `rsi.py`：

```python
import pandas as pd
from factors.base import BaseFactor

class RSIFactor(BaseFactor):
    name = "rsi"
    label = "RSI因子"
    window = 14
    category = "technical"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ts_code", "trade_date"])
        delta = df.groupby("ts_code")["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = df.groupby("ts_code")["gain_tmp"].transform(
            lambda x: x.rolling(self.window).mean())
        # ... 完整实现见注释
        return df[["ts_code", "trade_date", "value"]].dropna()
```

**第二步** —— 在 `main.py` 中注册：

```python
from factors.rsi import RSIFactor

def register_all_factors():
    # ...已有注册...
    FactorRegistry.register(RSIFactor())  # ← 加这一行
```

**第三步** —— 运行：

```bash
python main.py
```

新因子自动参与 IC 分析、分层回测、组合构建、多因子合成。

## 配置参数

编辑 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TUSHARE_TOKEN` | — | Tushare 接口密钥 |
| `START_DATE` / `END_DATE` | — | 回测区间 |
| `INDEX_UNIVERSE` | `"000300.SH"` | 股票池（None=全市场） |
| `LAYER_NUM` | 5 | 分层数 |
| `TOP_N` | 30 | 组合持仓数 |
| `REBALANCE_FREQ` | `"M"` | 调仓频率：D/W/M |
| `COMMISSION` | 0.0003 | 手续费（万三） |
| `SLIPPAGE` | 0.001 | 滑点（千一） |
| `RISK_STOP_LOSS` | -0.15 | 个股止损线 |
| `RISK_MAX_DRAWDOWN` | -0.25 | 组合熔断线 |

## 依赖

```
tushare>=1.4.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
numba>=0.58.0
statsmodels>=0.14.0
```

## 许可

MIT License — 详见 [LICENSE](./LICENSE)
