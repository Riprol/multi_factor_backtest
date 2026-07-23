import numpy as np
import pandas as pd
from factors.base import BaseFactor


class VolatilityFactor(BaseFactor):
    name = "volatility"
    label = "波动率因子"
    window = 20
    category = "volatility"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ts_code", "trade_date"])
        df["std_ret"] = df.groupby("ts_code")["ret"].transform(
            lambda x: x.rolling(self.window, min_periods=int(self.window * 0.6)).std()
        )
        df["value"] = -df["std_ret"]
        return df[["ts_code", "trade_date", "value"]].dropna(subset=["value"]).reset_index(drop=True)
