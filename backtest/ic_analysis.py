import os
import pandas as pd
import numpy as np
from utils.database import FactorDatabase
from utils.stats_tools import calc_mean_conf_interval, ttest_one_side, normality_approx_check
from config import CACHE_DIR


class ICAnalyzer:

    def __init__(self, db: FactorDatabase):
        self.db = db
        self.market_df = None

    def _load_market(self, start: str, end: str):
        if self.market_df is None:
            self.market_df = self.db.load_market_with_future(start, end)
        return self.market_df

    def compute_ic_series(self, factor_name: str, start: str, end: str) -> pd.DataFrame:
        """逐日计算 Rank IC & Pearson IC。

        时间对齐：factor_value(signal_date=t) 对应 future_ret(return_date=t+1)
        future_ret 由 load_market_with_future 通过 shift(-1) 生成，无前瞻偏差。
        """
        factor_df = self.db.load_factor(factor_name, start, end)
        market = self._load_market(start, end)

        merged = factor_df.merge(
            market[["ts_code", "trade_date", "future_ret"]],
            on=["ts_code", "trade_date"], how="inner"
        )
        merged = merged.dropna(subset=["value", "future_ret"])

        records = []
        for date, grp in merged.groupby("trade_date"):
            if len(grp) < 30:
                continue
            rank_ic = grp["value"].corr(grp["future_ret"], method="spearman")
            pearson = grp["value"].corr(grp["future_ret"], method="pearson")
            records.append({
                "trade_date": date,
                "factor_name": factor_name,
                "rank_ic": rank_ic,
                "pearson_ic": pearson
            })
        return pd.DataFrame(records)

    def compute_all_factors_ic(self, factor_names: list,
                                start: str, end: str) -> pd.DataFrame:
        all_ic = []
        for name in factor_names:
            ic_series = self.compute_ic_series(name, start, end)
            all_ic.append(ic_series)
        return pd.concat(all_ic, ignore_index=True)

    def ic_summary(self, ic_df: pd.DataFrame) -> pd.DataFrame:
        summary = ic_df.groupby("factor_name").apply(
            lambda g: pd.Series({
                "ic_mean": g["rank_ic"].mean(),
                "ic_std": g["rank_ic"].std(),
                "icir": g["rank_ic"].mean() / g["rank_ic"].std() if g["rank_ic"].std() > 0 else 0,
                "ic_positive_ratio": (g["rank_ic"] > 0).mean(),
                "ic_abs_mean": g["rank_ic"].abs().mean(),
                "obs_count": len(g)
            })
        ).reset_index()
        return summary

    def ic_summary_with_ci(self, ic_df: pd.DataFrame) -> pd.DataFrame:
        """IC 汇总 + 95% 置信区间 + t 检验"""
        rows = []
        for name, grp in ic_df.groupby("factor_name"):
            ci = calc_mean_conf_interval(grp["rank_ic"])
            tt = ttest_one_side(grp["rank_ic"].values)
            rows.append({
                "factor_name": name,
                "ic_mean": ci["mean"],
                "ic_std": grp["rank_ic"].std(),
                "icir": ci["mean"] / grp["rank_ic"].std() if grp["rank_ic"].std() > 0 else 0,
                "ic_positive_ratio": (grp["rank_ic"] > 0).mean(),
                "ic_abs_mean": grp["rank_ic"].abs().mean(),
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
                "zero_inside": ci["zero_inside"],
                "t_value": tt.get("t_value", float("nan")),
                "p_value": tt.get("p_value", float("nan")),
                "df": tt.get("df", 0),
                "obs_count": len(grp),
            })
        return pd.DataFrame(rows)

    def _load_market_with_industry_size(self, start: str, end: str) -> pd.DataFrame:
        """加载行情 + future_ret + industry + ln_cap（对数成交额作为市值代理）。"""
        market = self._load_market(start, end)
        if market.empty:
            return market

        # ln_cap: 对数成交额作为市值规模代理变量
        market["ln_cap"] = np.log(market["amount"].clip(lower=1))

        # 合并行业信息（来自 stock_basic.csv 缓存）
        csv_path = os.path.join(CACHE_DIR, "stock_basic.csv")
        if os.path.exists(csv_path):
            stock_info = pd.read_csv(csv_path, dtype={"ts_code": str})
            if "industry" in stock_info.columns:
                market = market.merge(
                    stock_info[["ts_code", "industry"]],
                    on="ts_code", how="left"
                )
        return market

    def compute_neutral_ic_series(self, factor_name: str,
                                   start: str, end: str) -> pd.DataFrame:
        """逐日计算行业+市值中性化后的 Rank IC & Pearson IC。

        流程：
        1. 加载因子值 & 行情（含 industry, ln_cap, future_ret）
        2. 逐日回归：因子值 ~ 行业哑变量 + ln_cap，取残差 = 纯因子暴露
        3. 用残差计算 IC
        """
        from utils.regression import neutralize_industry_style

        factor_df = self.db.load_factor(factor_name, start, end)
        if factor_df.empty:
            return pd.DataFrame()

        market = self._load_market_with_industry_size(start, end)
        if market.empty or "industry" not in market.columns:
            print(f"  [中性化] 缺少行业数据，跳过 {factor_name}")
            return pd.DataFrame()

        merged = factor_df.merge(
            market[["ts_code", "trade_date", "future_ret", "industry", "ln_cap"]],
            on=["ts_code", "trade_date"], how="inner"
        )
        merged = merged.dropna(subset=["value", "future_ret"])

        records = []
        for date, grp in merged.groupby("trade_date"):
            if len(grp) < 30:
                continue
            # 逐日中性化
            neutral = neutralize_industry_style(
                grp, factor_name="value",
                industry_col="industry", size_col="ln_cap"
            )
            valid_mask = grp["future_ret"].notna() & neutral.notna()
            n_valid = valid_mask.sum()
            if n_valid < 30:
                continue
            rank_ic = neutral[valid_mask].corr(
                grp.loc[valid_mask, "future_ret"], method="spearman"
            )
            pearson = neutral[valid_mask].corr(
                grp.loc[valid_mask, "future_ret"], method="pearson"
            )
            records.append({
                "trade_date": date,
                "factor_name": factor_name,
                "rank_ic": rank_ic,
                "pearson_ic": pearson
            })
        return pd.DataFrame(records)

    def compute_all_factors_neutral_ic(self, factor_names: list,
                                        start: str, end: str) -> pd.DataFrame:
        """批量计算所有因子的中性化 IC。"""
        all_ic = []
        for name in factor_names:
            print(f"  [中性化IC] 计算 {name} ...")
            ic_series = self.compute_neutral_ic_series(name, start, end)
            if not ic_series.empty:
                all_ic.append(ic_series)
        return pd.concat(all_ic, ignore_index=True) if all_ic else pd.DataFrame()

    def yearly_summary(self, ic_df: pd.DataFrame) -> pd.DataFrame:
        ic = ic_df.copy()
        ic["trade_date"] = pd.to_datetime(ic["trade_date"])
        ic["year"] = ic["trade_date"].dt.year
        result = ic.groupby(["factor_name", "year"]).apply(
            lambda g: pd.Series({
                "ic_mean": g["rank_ic"].mean(),
                "ic_std": g["rank_ic"].std(),
                "icir": g["rank_ic"].mean() / g["rank_ic"].std() if g["rank_ic"].std() > 0 else 0,
                "ic_positive_ratio": (g["rank_ic"] > 0).mean(),
                "obs_count": len(g)
            })
        ).reset_index()
        return result
