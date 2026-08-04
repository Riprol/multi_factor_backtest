"""
回归分析模块
============
一元 / 多元 OLS、虚拟变量生成、因子行业-市值中性化。
依赖 statsmodels（已安装）提供完整统计推断。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def create_dummy_variables(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """分类变量生成 k-1 组虚拟变量（自动删第一类规避完全共线性）。

    入参
    ----
    df     : DataFrame，含 cat_col 列
    cat_col: 分类列名（如 "industry"）

    返回
    ----
    DataFrame : index 与 df 对齐，列为 cat_col_值1, cat_col_值2, ...
    """
    dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)
    dummies.index = df.index
    return dummies.astype(float)


def ols_simple_regression(y: np.ndarray, x: np.ndarray) -> dict:
    """一元线性回归 y = a + b*x。

    返回
    ----
    {"params": {"const": a, "slope": b},
     "pvalues": {"const": p_a, "slope": p_b},
     "r_squared", "adj_r2", "df_residual"}
    """
    valid = np.isfinite(y) & np.isfinite(x)
    yv, xv = y[valid], x[valid]
    if len(yv) < 3:
        return {"error": "有效样本不足"}

    X = sm.add_constant(xv)
    model = sm.OLS(yv, X).fit()
    return {
        "params": {"const": model.params[0], "slope": model.params[1]},
        "pvalues": {"const": model.pvalues[0], "slope": model.pvalues[1]},
        "tvalues": {"const": model.tvalues[0], "slope": model.tvalues[1]},
        "r_squared": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "f_stat": model.fvalue,
        "p_f": model.f_pvalue,
        "df_residual": int(model.df_resid),
    }


def ols_multi_regression(y: np.ndarray, X: np.ndarray) -> dict:
    """多元线性回归 y = X·β + ε。

    入参
    ----
    y : (N,)   因变量
    X : (N, K) 自变量矩阵（不含截距，函数自动添加）

    返回
    ----
    {"params"      : array (K+1,)  系数 [const, β₁, β₂, ...]
     "pvalues"     : array (K+1,)  各系数 p 值
     "adj_r2"      : float         调整 R²
     "f_stat"      : float         F 统计量
     "p_f"         : float         F 检验 p 值
     "df_residual" : int           残差自由度
     "residuals"   : array (N,)   残差序列
    }
    """
    valid = np.isfinite(y)
    for k in range(X.shape[1]):
        valid = valid & np.isfinite(X[:, k])
    yv, Xv = y[valid], X[valid]
    if len(yv) < X.shape[1] + 2:
        return {"error": "有效样本不足"}

    Xc = sm.add_constant(Xv)
    model = sm.OLS(yv, Xc).fit()
    return {
        "params": model.params,
        "pvalues": model.pvalues,
        "adj_r2": model.rsquared_adj,
        "f_stat": model.fvalue,
        "p_f": model.f_pvalue,
        "df_residual": int(model.df_resid),
        "residuals": model.resid,
    }


def ols_cross_sectional(factor_df: pd.DataFrame,
                         market_df: pd.DataFrame,
                         factor_col: str = "value",
                         ret_col: str = "future_ret",
                         date_col: str = "trade_date",
                         code_col: str = "ts_code",
                         min_stocks: int = 30) -> pd.DataFrame:
    """单变量截面 OLS 回归 —— 每日/每月独立执行。

    对每个截面日期，跑一元线性回归：
        future_ret = α + β × factor_value + ε

    输出每个截面的系数、R²、t 统计量、p 值，用于评估因子
    在截面上的定价能力（与 IC 分析互补，OLS 给出线性关系的强度和显著性）。

    入参
    ----
    factor_df  : DataFrame，含 code_col, date_col, factor_col 三列
    market_df  : DataFrame，含 code_col, date_col, ret_col 三列
    factor_col : 因子值列名（默认 "value"）
    ret_col    : 下期收益列名（默认 "future_ret"，需提前 shift 好）
    date_col   : 日期列名
    code_col   : 股票代码列名
    min_stocks : 单截面最少股票数，不足则跳过

    返回
    ----
    DataFrame，列：
        trade_date     — 截面日期
        alpha          — 截距项系数
        beta           — 因子斜率系数（反映因子对收益的边际影响）
        r_squared      — 拟合优度 R²
        adj_r2         — 调整 R²
        t_alpha        — 截距项 t 统计量
        t_beta         — 斜率 t 统计量
        p_alpha        — 截距项 p 值
        p_beta         — 斜率 p 值
        f_stat         — 整体 F 统计量
        p_f            — F 检验 p 值
        n_obs          — 该截面有效样本数
    """
    # 合并因子与未来收益
    merged = factor_df.merge(
        market_df[[code_col, date_col, ret_col]],
        on=[code_col, date_col], how="inner"
    )
    merged = merged.dropna(subset=[factor_col, ret_col])

    records = []
    for date, grp in merged.groupby(date_col):
        n = len(grp)
        if n < min_stocks:
            continue

        y = grp[ret_col].to_numpy(dtype=float)
        x = grp[factor_col].to_numpy(dtype=float)

        # 剔除 inf / NaN（dropna 已处理 NaN，再拦一手 inf）
        valid = np.isfinite(y) & np.isfinite(x)
        yv, xv = y[valid], x[valid]
        if len(yv) < min_stocks:
            continue

        # 调用统一的一元回归函数（消除重复 OLS 代码）
        result = ols_simple_regression(yv, xv)
        if "error" in result:
            continue

        records.append({
            "trade_date": date,
            "alpha":      result["params"]["const"],
            "beta":       result["params"]["slope"],
            "r_squared":  result["r_squared"],
            "adj_r2":     result["adj_r2"],
            "t_alpha":    result["tvalues"]["const"],
            "t_beta":     result["tvalues"]["slope"],
            "p_alpha":    result["pvalues"]["const"],
            "p_beta":     result["pvalues"]["slope"],
            "f_stat":     result["f_stat"],
            "p_f":        result["p_f"],
            "n_obs":      len(yv),
        })

    return pd.DataFrame(records)


def neutralize_industry_style(df: pd.DataFrame,
                               factor_name: str,
                               industry_col: str = "industry",
                               size_col: str = "ln_cap") -> pd.Series:
    """因子行业 + 市值中性化。

    流程：
        因子值 = y，行业哑变量 + 市值 = X
        跑多元 OLS，返回残差 = 剔除了行业和市值影响后的纯因子暴露。

    入参
    ----
    df          : DataFrame，含 factor_name, industry_col, size_col 列
    factor_name : 因子值列名
    industry_col: 行业分类列名
    size_col    : 市值列名（对数市值，如 ln_cap）

    返回
    ----
    pd.Series : 中性化后的因子值（长度与 df 对齐，缺失值填充 NaN）
    """
    if factor_name not in df.columns:
        raise KeyError(f"因子列 '{factor_name}' 不存在")
    if industry_col not in df.columns:
        raise KeyError(f"行业列 '{industry_col}' 不存在")

    y = df[factor_name].to_numpy(dtype=float)

    # 行业哑变量
    dummies = create_dummy_variables(df, industry_col)

    # 拼接 X 矩阵
    X_cols = dummies.columns.tolist()
    if size_col in df.columns:
        X_cols.append(size_col)
        X_mat = np.column_stack([dummies.values, df[size_col].to_numpy(dtype=float)])
    else:
        X_mat = dummies.values

    result = ols_multi_regression(y, X_mat)
    if "error" in result:
        print(f"  [中性化] OLS 失败: {result['error']}")
        return df[factor_name]

    # 残差回填到原 index
    valid_idx = np.isfinite(y)
    for k in range(X_mat.shape[1]):
        valid_idx = valid_idx & np.isfinite(X_mat[:, k])

    residuals = np.full(len(df), np.nan)
    residuals[valid_idx] = result["residuals"]

    return pd.Series(residuals, index=df.index, name=f"{factor_name}_neutral")
