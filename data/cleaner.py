import pandas as pd
import numpy as np
from config import MIN_PRICE, DROP_LIMIT


class DataCleaner:

    @staticmethod
    def clean(raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        df["ts_code"]    = df["ts_code"].astype(str)
        df["trade_date"] = df["trade_date"].astype(str)
        df["ret"] = df["close"] / df["pre_close"] - 1

        before = df.shape[0]
        df = df[(df["vol"] > 0) & (df["amount"] > 0)]

        if DROP_LIMIT:
            is_cyb_kcb = df["ts_code"].str.startswith(("300", "688"))
            limit = np.where(is_cyb_kcb, 0.195, 0.098)
            df["ret_abs"] = df["ret"].abs()
            df = df[df["ret_abs"] <= limit * 1.01]
            df = df.drop(columns=["ret_abs"])

        df = df[df["close"] >= MIN_PRICE]
        df = df.drop_duplicates(subset=["ts_code", "trade_date"])

        after = df.shape[0]
        print(f"[清洗器] {before} -> {after} 条 (剔除 {before-after} 条脏数据)")
        return df.reset_index(drop=True)
