"""
复杂自定义滚动指标计算模块
===========================
两层结构：
  1. numba @njit 加速核心函数（纯 numpy 数组，禁用 pandas）
  2. wrapper 封装函数（接收 DataFrame，类型转换 + 对齐 + 回传）

适用场景：因子逻辑复杂，无法用 pandas rolling 向量化表达时使用。
示例：滚动窗口内的价格-成交量相关系数（量价协同因子）。

numba 避坑要点：
  1. @njit 编译在首次调用时发生，有 0.5~2s 的编译预热开销。
  2. 函数内禁止使用 pandas、scipy 对象，只能用 numpy + python 原生类型。
  3. 避免在循环内创建大数组对象（用预分配 + 切片）。
  4. np.nan 在 numba 中类型推断为 float64。
  5. np.mean/std 在 numba 中可用，但 axis 参数受限，建议手动实现。
  6. 不支持 try/except，需要用 if/else 处理边界条件。
"""
import numpy as np
import pandas as pd
from typing import Optional
from numba import njit


# ═══════════════════════════════════════════════════════════════
# 一、底层 numba 加速核心函数
# ═══════════════════════════════════════════════════════════════

@njit
def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """numba 兼容的 Pearson 相关系数（手工实现，不依赖 scipy）。

    避坑：numba 不支持 scipy.stats.pearsonr，必须自行计算。
    """
    n = len(x)
    # 剔除 nan
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = valid.sum()
    if n_valid < 3:
        return np.nan

    xv = x[valid]
    yv = y[valid]
    mx = np.mean(xv)
    my = np.mean(yv)
    sx = np.std(xv)
    sy = np.std(yv)

    if sx == 0.0 or sy == 0.0:
        return np.nan   # 常数序列，相关系数无定义

    cov = np.mean((xv - mx) * (yv - my))
    return cov / (sx * sy)


@njit
def rolling_corr_numba(prices: np.ndarray, volumes: np.ndarray,
                       window: int) -> np.ndarray:
    """量价滚动相关系数 —— 纯 numba 加速内核。

    参数
    ----
    prices  : (N,)   float64  股票价格序列（日频，按时间升序排列）
    volumes : (N,)   float64  成交量序列（与 prices 同长度、同顺序）
    window  : int             滚动窗口大小（交易日数量）

    返回
    ----
    result  : (N,)   float64  每日滚动相关系数
              result[i] = corr(prices[i-window+1:i+1], volumes[i-window+1:i+1])
              窗口未满位置（i < window-1）填充 np.nan

    避坑
    ----
    1. 仅使用 numpy 数组，不涉及任何 pandas 对象。
    2. 预分配 result 数组，在循环外一次性创建，避免循环内反复分配内存。
    3. Pearson 相关的手工实现放在 _pearson_r 中，确保 numba 可编译。
    """
    n = prices.shape[0]
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        # 切片创建 view（O(1)），然后 _pearson_r 内部会 copy 有效元素
        start = i - window + 1
        result[i] = _pearson_r(prices[start:i + 1], volumes[start:i + 1])

    return result


@njit
def rolling_slope_numba(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """滚动 OLS 斜率（pandas rolling 无法直接实现的典型场景）。

    每窗口内做一元线性回归 y = a + b*x，返回斜率 b。
    """
    n = x.shape[0]
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        start = i - window + 1
        xw = x[start:i + 1]
        yw = y[start:i + 1]
        valid = np.isfinite(xw) & np.isfinite(yw)
        if valid.sum() < 3:
            continue
        xv = xw[valid]
        yv = yw[valid]
        mx = np.mean(xv)
        my = np.mean(yv)
        # 斜率 = cov(x,y) / var(x)
        var_x = np.mean((xv - mx) ** 2)
        if var_x == 0.0:
            continue
        cov = np.mean((xv - mx) * (yv - my))
        result[i] = cov / var_x
    return result


@njit
def rolling_rank_numba(values: np.ndarray, window: int) -> np.ndarray:
    """滚动百分位排名（pandas rolling + rank 的纯 numba 替代）。

    返回：当日值在窗口内的分位数 (0~1)。
    """
    n = values.shape[0]
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        start = i - window + 1
        w = values[start:i + 1].copy()          # numba 需要显式 copy
        valid = np.isfinite(w)
        if valid.sum() < 3:
            continue
        wv = w[valid]
        nv = wv.shape[0]
        # 排序后找当前值的位置
        sorted_idx = np.argsort(wv)
        # 当天的值 = wv 最后一个有效元素
        cur_val = values[i]
        # 二分找 rank（简单实现：遍历计数）
        rank_count = 0
        for j in range(nv):
            if wv[j] <= cur_val:
                rank_count += 1
        result[i] = rank_count / nv
    return result


# ═══════════════════════════════════════════════════════════════
# 二、上层 wrapper 封装函数
# ═══════════════════════════════════════════════════════════════

def rolling_corr(df: pd.DataFrame,
                 price_col: str = "close",
                 volume_col: str = "vol",
                 group_col: str = "ts_code",
                 date_col: str = "trade_date",
                 window: int = 60) -> pd.Series:
    """量价滚动相关系数 —— 上层封装。

    对每只股票独立计算，结果与原始 DataFrame 完全对齐索引。

    参数
    ----
    df         : 已排序的行情 DataFrame（含 ts_code, trade_date）
    price_col  : 价格列名
    volume_col : 成交量列名
    group_col  : 分组列（股票代码）
    window     : 滚动窗口（交易日数，如 60 ≈ 3 个月）

    返回
    ----
    pd.Series  : 与 df 同长度、同索引的相关系数序列
    """
    df = df.sort_values([group_col, date_col]).copy()
    results = []

    for _, grp in df.groupby(group_col):
        prices  = grp[price_col].to_numpy(dtype=np.float64)
        volumes = grp[volume_col].to_numpy(dtype=np.float64)

        # 调用 numba 内核
        corr_arr = rolling_corr_numba(prices, volumes, window)

        results.append(pd.Series(corr_arr, index=grp.index, name="rolling_corr"))

    out = pd.concat(results).sort_index()
    return out


def rolling_slope(df: pd.DataFrame,
                  x_col: str = "amount",
                  y_col: str = "close",
                  group_col: str = "ts_code",
                  window: int = 60) -> pd.Series:
    """滚动回归斜率封装。"""
    df = df.sort_values([group_col, "trade_date"]).copy()
    results = []
    for _, grp in df.groupby(group_col):
        x = grp[x_col].to_numpy(dtype=np.float64)
        y = grp[y_col].to_numpy(dtype=np.float64)
        arr = rolling_slope_numba(x, y, window)
        results.append(pd.Series(arr, index=grp.index, name="rolling_slope"))
    return pd.concat(results).sort_index()


def rolling_rank(df: pd.DataFrame,
                 value_col: str = "close",
                 group_col: str = "ts_code",
                 window: int = 60) -> pd.Series:
    """滚动百分位排名封装。"""
    df = df.sort_values([group_col, "trade_date"]).copy()
    results = []
    for _, grp in df.groupby(group_col):
        v = grp[value_col].to_numpy(dtype=np.float64)
        arr = rolling_rank_numba(v, window)
        results.append(pd.Series(arr, index=grp.index, name="rolling_rank"))
    return pd.concat(results).sort_index()


# ═══════════════════════════════════════════════════════════════
# 三、性能对比说明
# ═══════════════════════════════════════════════════════════════
"""
三种实现方式的性能差异（以 5000 只股票 × 250 天 × window=60 为例）：

┌────────────────────┬──────────┬─────────────────────────────────┐
│ 方案               │ 耗时     │ 说明                            │
├────────────────────┼──────────┼─────────────────────────────────┤
│ 纯 pandas rolling  │ ~2s      │ 仅适用于内置聚合(mean/std/sum)  │
│                    │          │ 无法直接实现 corr 或自定义逻辑   │
├────────────────────┼──────────┼─────────────────────────────────┤
│ groupby + apply +  │ ~120s    │ 每只股票 apply(lambda)           │
│ 无 numba 循环      │          │ Python 循环解释执行，极慢        │
├────────────────────┼──────────┼─────────────────────────────────┤
│ numba @njit        │ ~8s      │ 首次含 ~2s 编译预热，            │
│ （本模块方案）     │          │ 后续 ~6s，比无 numba 快 20 倍    │
└────────────────────┴──────────┴─────────────────────────────────┘

适用场景：
  ✅ 自定义逻辑复杂，pandas rolling 的 str 参数无法表达
  ✅ 需要 Pearson/Spearman 相关、回归斜率、自定义排序等
  ✅ 数据量大（数千只股票 × 数年），性能敏感

不适用场景：
  ❌ pandas 已有内置函数可直接完成（如 rolling.mean）
  ❌ 数据量极小（几十行），numba 编译开销反而占主导
  ❌ 逻辑极度简单（一行表达式），用 numba 属于过度优化

numba 调试技巧：
  1. 先用 @njit(debug=True) 编译，确保无 JIT 错误
  2. 内核函数单独测试：创建一个小 numpy 数组调用验证
  3. 编译缓存：@njit(cache=True) 可避免重复编译（仅首次慢）
  4. 类型签名：@njit("f8[:](f8[:],f8[:],i4)") 显式声明加速编译
  5. 禁用 numba 对比测试：注释掉 @njit 对比结果一致性
"""
