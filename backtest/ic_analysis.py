import pandas as pd
import numpy as np
from utils.database import FactorDatabase


class ICAnalyzer:

    def __init__(self, db: FactorDatabase):
        self.db = db
        self.market_df = None

    def _load_market(self, start: str, end: str):
        if self.market_df is None:
            df = self.db.load_market(start, end)
            df = df.sort_values(["ts_code", "trade_date"])
            df["future_ret"] = df.groupby("ts_code")["ret"].shift(-1)
            self.market_df = df
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
