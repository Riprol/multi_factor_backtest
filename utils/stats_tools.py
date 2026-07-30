import pandas as pd
import numpy as np


def winsorize_cross_section(df: pd.DataFrame, value_col: str = "value",
                             date_col: str = "trade_date",
                             lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """逐日截面缩尾：把每只股票的因子值夹到当天 [lower, upper] 分位数之间"""
    def _winsor(grp):
        lo = grp.quantile(lower)
        hi = grp.quantile(upper)
        return grp.clip(lower=lo, upper=hi)

    return df.groupby(date_col)[value_col].transform(_winsor)


def zscore_cross_section(df: pd.DataFrame, value_col: str = "value",
                          date_col: str = "trade_date") -> pd.Series:
    """逐日截面 Z-score：减均值除以标准差"""
    g = df.groupby(date_col)[value_col]
    return (df[value_col] - g.transform("mean")) / g.transform("std")


def calc_pearson_ic(factor_values: pd.Series, future_ret: pd.Series) -> float:
    """因子值 vs 下一期收益的 Pearson 相关系数。future_ret 须已 shift(-1)。"""
    return factor_values.corr(future_ret, method="pearson")


def calc_rank_ic(factor_values: pd.Series, future_ret: pd.Series) -> float:
    """因子值 vs 下一期收益的 Spearman 秩相关系数。future_ret 须已 shift(-1)。"""
    return factor_values.corr(future_ret, method="spearman")


def calc_ic_daily(factor_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.Series:
    """输入原始因子值和行情，自动 shift 得到未来收益，逐日计算 Rank IC。

    factor_df: 含 ts_code, trade_date, value
    market_df: 含 ts_code, trade_date, ret
    返回: index=trade_date 的 Rank IC 序列
    """
    m = market_df.sort_values(["ts_code", "trade_date"]).copy()
    m["future_ret"] = m.groupby("ts_code")["ret"].shift(-1)

    merged = factor_df.merge(
        m[["ts_code", "trade_date", "future_ret"]],
        on=["ts_code", "trade_date"], how="inner"
    ).dropna(subset=["value", "future_ret"])

    records = {}
    for date, grp in merged.groupby("trade_date"):
        if len(grp) < 30:
            continue
        records[date] = calc_rank_ic(grp["value"], grp["future_ret"])
    return pd.Series(records, name="rank_ic")


def ic_stats(ic_series: pd.Series) -> dict:
    """IC 序列统计：均值、标准差、ICIR"""
    m = ic_series.mean()
    s = ic_series.std()
    return {
        "ic_mean": m,
        "ic_std": s,
        "icir": m / s if s > 0 else 0.0,
        "ic_positive_ratio": (ic_series > 0).mean(),
    }


def calc_mean_conf_interval(ic_series: pd.Series, alpha: float = 0.05) -> dict:
    """IC 均值 95% 置信区间。

    公式：mean ± z_{α/2} × std/√n
    返回 {"mean", "lower", "upper", "n", "zero_inside"}
    """
    n = len(ic_series)
    if n < 2:
        return {"mean": float("nan"), "lower": float("nan"),
                "upper": float("nan"), "n": n, "zero_inside": None}
    from scipy.stats import norm
    m = ic_series.mean()
    se = ic_series.std() / np.sqrt(n)
    z = norm.ppf(1 - alpha / 2)   # 95% → 1.96
    lower = m - z * se
    upper = m + z * se
    return {
        "mean": m,
        "lower": lower,
        "upper": upper,
        "n": n,
        "zero_inside": lower <= 0 <= upper,
    }


def factor_corr_matrix(factor_wide: pd.DataFrame) -> pd.DataFrame:
    """批量因子截面相关系数矩阵。

    factor_wide: index=ts_code, columns=各因子名（数据库 load_all_factors 返回格式）
    返回: 因子 × 因子 Pearson 相关方阵
    """
    z = factor_wide.apply(lambda col: (col - col.mean()) / col.std())
    return z.corr(method="pearson")


def factor_cov_matrix(factor_wide: pd.DataFrame) -> pd.DataFrame:
    """批量因子截面协方差矩阵。用于组合优化、风险预算。"""
    z = factor_wide.apply(lambda col: (col - col.mean()) / col.std())
    return z.cov()


def calc_standard_error(sample: np.ndarray) -> float:
    """均值标准误 SE = std / √n"""
    n = len(sample)
    if n < 2:
        return float("nan")
    return float(np.std(sample, ddof=1) / np.sqrt(n))


def ttest_one_side(sample: np.ndarray, mu0: float = 0.0) -> dict:
    """单样本单侧 t 检验 H0: 均值 <= mu0  vs  H1: 均值 > mu0

    返回 {"t_value", "p_value", "df", "mean_val", "se", "ci_lower", "ci_upper"}
    95% CI 基于双侧 t 分布；df < 30 时打印低自由度风险提示。
    """
    from scipy.stats import t as t_dist
    n = len(sample)
    if n < 2:
        return {"error": "样本量不足"}
    m = float(np.mean(sample))
    se = float(np.std(sample, ddof=1) / np.sqrt(n))
    t_val = (m - mu0) / se if se > 0 else 0.0
    df = n - 1
    p_val = float(1 - t_dist.cdf(t_val, df))  # 单侧右尾
    t_crit = float(t_dist.ppf(0.975, df))
    ci_lower = m - t_crit * se
    ci_upper = m + t_crit * se

    if df < 30:
        print(f"  [警告] IC 自由度仅 {df}，t 检验结果仅供参考，建议样本量 >= 30")

    return {
        "t_value": t_val,
        "p_value": p_val,
        "df": df,
        "mean_val": m,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def normality_approx_check(data: np.ndarray) -> dict:
    """偏度 & 峰度（粗略核验 CLT 正态性近似是否成立）

    偏度 ≈ 0 表示对称，峰度 ≈ 3 表示正态（超额峰度 ≈ 0）
    """
    from scipy.stats import skew, kurtosis
    n = len(data)
    if n < 4:
        return {"skewness": float("nan"), "kurtosis": float("nan"), "n": n}
    return {
        "skewness": float(skew(data)),
        "kurtosis": float(kurtosis(data, fisher=False)),  # fisher=False → 正态=3
        "n": n,
    }
