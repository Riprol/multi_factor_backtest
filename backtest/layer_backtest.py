import pandas as pd
import numpy as np
from config import LAYER_NUM
from utils.database import FactorDatabase


class LayerBacktester:

    def __init__(self, db: FactorDatabase, layer_num: int = LAYER_NUM):
        self.db = db
        self.layer_num = layer_num
        self.market_df = None

    def _load_market(self, start: str, end: str):
        if self.market_df is None:
            df = self.db.load_market(start, end)
            df = df.sort_values(["ts_code", "trade_date"])
            df["future_ret"] = df.groupby("ts_code")["ret"].shift(-1)
            self.market_df = df
        return self.market_df

    def run(self, factor_name: str, start: str, end: str) -> dict:
        factor_df = self.db.load_factor(factor_name, start, end)
        market = self._load_market(start, end)
        merged = factor_df.merge(
            market[["ts_code", "trade_date", "future_ret"]],
            on=["ts_code", "trade_date"], how="inner"
        )
        merged = merged.dropna(subset=["value", "future_ret"])

        daily_records = []
        long_short_records = []
        for date, grp in merged.groupby("trade_date"):
            if len(grp) < self.layer_num * 10:
                continue
            grp = grp.copy()
            grp["layer"] = pd.qcut(
                grp["value"], self.layer_num,
                labels=list(range(1, self.layer_num + 1)),
                duplicates="drop"
            )
            layer_ret = grp.groupby("layer")["future_ret"].mean()
            for lv, ret in layer_ret.items():
                daily_records.append({"trade_date": date, "layer": lv, "future_ret": ret})
            if 1 in layer_ret.index and self.layer_num in layer_ret.index:
                long_short_records.append({
                    "trade_date": date,
                    "long_short_ret": layer_ret[self.layer_num] - layer_ret[1]
                })

        daily_df = pd.DataFrame(daily_records)
        if daily_df.empty:
            print(f"  [分层回测] {factor_name}: 无足够数据")
            return {}
        pivot = pd.pivot_table(daily_df, index="trade_date", columns="layer", values="future_ret")
        pivot.index = pd.to_datetime(pivot.index)
        cum_net = (1 + pivot).cumprod()

        ls_df = pd.DataFrame(long_short_records)
        if not ls_df.empty:
            ls_df["trade_date"] = pd.to_datetime(ls_df["trade_date"])
            ls_df = ls_df.set_index("trade_date")
            ls_df["cum_long_short"] = (1 + ls_df["long_short_ret"]).cumprod()

        return {"daily_layer_ret": pivot, "cum_net_value": cum_net, "long_short": ls_df}
