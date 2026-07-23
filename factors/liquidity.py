import pandas as pd
import numpy as np
from factors.base import BaseFactor


class LiquidityFactor(BaseFactor):
    name = "liquidity"
    label = "流动性因子"
    window = 20
    category = "volume"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ts_code", "trade_date"])
        df["illiq"] = df["ret"].abs() / (df["amount"] / 10000 + 1)
        df["value"] = -df.groupby("ts_code")["illiq"].transform(
            lambda x: x.rolling(self.window, min_periods=int(self.window * 0.6)).mean()
        )
        return df[["ts_code", "trade_date", "value"]].dropna(subset=["value"]).reset_index(drop=True)
