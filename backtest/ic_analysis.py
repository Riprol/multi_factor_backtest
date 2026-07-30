import pandas as pd
import numpy as np
from utils.database import FactorDatabase
from utils.stats_tools import calc_mean_conf_interval, ttest_one_side, normality_approx_check


class ICAnalyzer:

    def __init__(self, db: FactorDatabase):
        self.db = db
        self.market_df = None

    def _load_market(self, start: str, end: str):
        if self.market_df is None:
            self.market_df = self.db.load_market_with_future(start, end)
        return self.market_df

    def compute_ic_series(self, factor_name: str, start: str, end: str) -> pd.DataFrame:
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
