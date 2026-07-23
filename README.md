# 多因子回测框架 (Multi-Factor Backtest Framework)

基于 Tushare 免费接口的 A 股多因子回测系统，一键从数据下载到报告输出。

## 特性

- **零门槛数据获取**：仅需 Tushare 免费 Token，自动下载 + 缓存日线
- **完整数据清洗**：剔除停牌、涨跌停、重复值、创业板/科创板自适应涨跌幅
- **因子即插即用**：继承 `BaseFactor` 实现 `compute()`，注册后自动接入全流程
- **六维分析输出**：IC 分析 → 分层回测 → 多空组合 → 策略净值 → 绩效指标 → Markdown 报告
- **SQLite 因子仓库**：因子值统一存储，支持多因子截面查询与合成
- **专业可视化**：`fig/ax` 多子图风格，分层净值、IC 时序/分布、策略回撤一应俱全

## 项目结构

```
multi_factor_backtest/
├── config.py              ← 全局配置（Token、日期、参数）
├── main.py                ← 主入口，一键运行
│
├── data/                  ← 数据层
│   ├── calendar.py        # 本地交易日历
│   ├── downloader.py      # Tushare 下载器
│   └── cleaner.py         # 数据清洗流水线
│
├── factors/               ← 因子层（核心扩展点）
│   ├── base.py            # 因子基类
│   ├── registry.py        # 因子注册表
│   ├── momentum.py        # 动量因子
│   ├── reversal.py        # 反转因子
│   ├── volatility.py      # 低波动因子
│   ├── turnover.py        # 换手率因子
│   ├── liquidity.py       # 流动性因子(Amihud)
│   └── composite.py       # 多因子合成器
│
├── backtest/              ← 回测层
│   ├── ic_analysis.py     # IC 分析
│   ├── layer_backtest.py  # 分层回测
│   ├── portfolio.py       # 组合构建
│   └── performance.py     # 绩效评估
│
├── visualization/         ← 可视化层
│   ├── charts.py          # 图表绘制
│   └── report.py          # 报告生成
│
├── utils/                 ← 工具层
│   ├── database.py        # SQLite 因子数据库
│   └── helpers.py         # 辅助函数
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
START_DATE = "起始日期"   
END_DATE   = "终止日期"   
```

### 4. 运行

```bash
python main.py
```

全自动流程：**下载 → 清洗 → 因子计算 → IC 分析 → 分层回测 → 组合构建 → 多因子合成 → 报告输出**。

## 输出说明

所有结果在 `output/` 目录：

| 文件 | 说明 |
|------|------|
| `layer_net_*.png` | 分层累计净值曲线（看因子区分度） |
| `ic_*.png` | IC 时序 + 分布直方图（看预测稳定性） |
| `nav_*.png` | 策略净值 + 回撤曲线 |
| `ic_comparison.png` | 多因子 IC 箱线对比 |
| `ic_summary.csv` | IC 明细数据 |
| `backtest_report.md` | Markdown 完整报告 |

### 如何判断因子好坏？

- **Rank IC 均值**：> 0.02 不错，> 0.05 优秀
- **ICIR**（IC 均值 / IC 标准差）：> 0.3 可用，> 0.5 良好
- **分层净值单调性**：5 条线从低到高排列清晰
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
| `START_DATE` / `END_DATE` | 2025上半年 | 回测区间 |
| `LAYER_NUM` | 5 | 分层数 |
| `TOP_N` | 50 | 组合持仓数 |
| `REBALANCE_FREQ` | `"M"` | 调仓频率：D/W/M |
| `COMMISSION` | 0.0003 | 手续费（万三） |
| `SLIPPAGE` | 0.001 | 滑点（千一） |

## 依赖

- Python ≥ 3.9
- tushare ≥ 1.4.0
- pandas ≥ 2.0
- numpy ≥ 1.24
- matplotlib ≥ 3.7

##  许可

MIT License — 详见 [LICENSE](./LICENSE)
