import pandas as pd
from factors.base import BaseFactor


class MomentumFactor(BaseFactor):
    name = "momentum"
    label = "动量因子"
    window = 20
    category = "momentum"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ts_code", "trade_date"])
        df["value"] = df.groupby("ts_code")["ret"].transform(
            lambda x: x.rolling(self.window, min_periods=int(self.window * 0.6)).sum()
        )
        return df[["ts_code", "trade_date", "value"]].dropna(subset=["value"]).reset_index(drop=True)
