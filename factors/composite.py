import pandas as pd
import numpy as np
from utils.database import FactorDatabase


class CompositeFactor:

    def __init__(self, db: FactorDatabase):
        self.db = db

    def equal_weight(self, factor_names: list, date: str) -> pd.DataFrame:
        wide = self.db.load_all_factors(factor_names, date)
        if wide.empty:
            return pd.DataFrame(columns=["value"])
        z_scored = wide.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
        composite = z_scored.mean(axis=1)
        result = composite.reset_index()
        result.columns = ["ts_code", "value"]
        result["trade_date"] = date
        return result.dropna()

    def ic_weighted(self, factor_names: list, ic_df: pd.DataFrame) -> dict:
        subset = ic_df[ic_df["factor_name"].isin(factor_names)]
        if subset.empty:
            w = 1.0 / len(factor_names)
            return {n: w for n in factor_names}
        subset = subset.set_index("factor_name")
        total = subset["ic_abs_mean"].sum()
        if total == 0:
            w = 1.0 / len(factor_names)
            return {n: w for n in factor_names}
        return (subset["ic_abs_mean"] / total).to_dict()

    def compute_daily(self, factor_names: list, date: str,
                      method: str = "equal",
                      weights: dict = None) -> pd.DataFrame:
        wide = self.db.load_all_factors(factor_names, date)
        if wide.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date", "value"])
        z_scored = wide.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
        if method == "weighted" and weights:
            composite = pd.Series(0.0, index=z_scored.index)
            for name, w in weights.items():
                if name in z_scored.columns:
                    composite += w * z_scored[name]
        else:
            composite = z_scored.mean(axis=1)
        result = composite.reset_index()
        result.columns = ["ts_code", "value"]
        result["trade_date"] = date
        return result.dropna()
