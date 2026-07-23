import pandas as pd
from factors.base import BaseFactor


class ReversalFactor(BaseFactor):
    name = "reversal"
    label = "反转因子"
    window = 5
    category = "reversal"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ts_code", "trade_date"])
        df["short_ret"] = df.groupby("ts_code")["ret"].transform(
            lambda x: x.rolling(self.window, min_periods=3).sum()
        )
        df["value"] = -df["short_ret"]
        return df[["ts_code", "trade_date", "value"]].dropna(subset=["value"]).reset_index(drop=True)
