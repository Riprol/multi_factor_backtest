import pandas as pd
from factors.base import BaseFactor


class TurnoverFactor(BaseFactor):
    name = "turnover"
    label = "换手率因子"
    window = 20
    category = "volume"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ts_code", "trade_date"])
        df["daily_turn"] = df["vol"] / df.groupby("ts_code")["vol"].transform("mean")
        df["value"] = df.groupby("ts_code")["daily_turn"].transform(
            lambda x: x.rolling(self.window, min_periods=int(self.window * 0.6)).mean()
        )
        return df[["ts_code", "trade_date", "value"]].dropna(subset=["value"]).reset_index(drop=True)
