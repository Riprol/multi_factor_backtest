import pandas as pd
import numpy as np
from config import MIN_PRICE, DROP_LIMIT, MIN_LIST_DAYS


class DataCleaner:

    @staticmethod
    def clean(raw_df: pd.DataFrame,
              stock_basic: pd.DataFrame = None) -> pd.DataFrame:
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

        # ST / 次新股过滤
        if stock_basic is not None and not stock_basic.empty:
            df = DataCleaner._filter_st_and_new(df, stock_basic)

        df = df.drop_duplicates(subset=["ts_code", "trade_date"])

        after = df.shape[0]
        print(f"[清洗器] {before} -> {after} 条 (剔除 {before-after} 条脏数据)")
        return df.reset_index(drop=True)

    @staticmethod
    def _filter_st_and_new(df: pd.DataFrame,
                            stock_basic: pd.DataFrame) -> pd.DataFrame:
        """剔除 ST/*ST 和上市不足 MIN_LIST_DAYS 交易日的次新股。

        stock_basic 需含 ts_code, name, list_date 列。
        """
        info = stock_basic[["ts_code", "name", "list_date"]].copy()
        info["list_date"] = pd.to_datetime(info["list_date"], errors="coerce")

        # ST 标记
        st_codes = set(info[info["name"].str.contains("ST", na=False)]["ts_code"])

        before = df.shape[0]
        df = df[~df["ts_code"].isin(st_codes)]

        # 次新股：trade_date - list_date < MIN_LIST_DAYS → 剔除
        if "list_date" in info.columns and MIN_LIST_DAYS > 0:
            info_dict = dict(zip(info["ts_code"], info["list_date"]))
            df["_list_date"] = df["ts_code"].map(info_dict)
            df["trade_dt"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df["_listed_days"] = (df["trade_dt"] - df["_list_date"]).dt.days
            df = df[df["_listed_days"] >= MIN_LIST_DAYS]
            df = df.drop(columns=["_list_date", "trade_dt", "_listed_days"])

        after = df.shape[0]
        if before != after:
            print(f"  [ST/次新] {before} -> {after} 只 (剔除 {before-after} 只)")
        return df

    @staticmethod
    def filter_by_index_db(df: pd.DataFrame, db, index_code: str) -> pd.DataFrame:
        """
        基于 SQLite 中 index_weights 表做月度截面过滤。
        逻辑：对每一行，取 trade_date 的前 6 位（YYYYMM），
        匹配 index_weights 中同月的成分股。不匹配的剔除。
        这是动态成分——每月使用当月真实成分名单，杜绝未来函数。
        """
        if index_code is None:
            return df

        iw = db.get_index_stocks(index_code)
        if iw.empty:
            print(f"[指数过滤] {index_code} 无成分数据，跳过过滤")
            return df

        # 构建 "YYYYMM" -> set(con_code) 映射
        iw["ym"] = iw["trade_date"].str[:6]
        monthly_set = iw.groupby("ym")["con_code"].apply(set).to_dict()

        before = df.shape[0]
        df = df.copy()
        df["_ym"] = df["trade_date"].str[:6]

        # 向量化：对每个月份批量过滤
        keep = pd.Series(False, index=df.index)
        for ym, stocks in monthly_set.items():
            idx = df[df["_ym"] == ym].index
            keep[idx] = df.loc[idx, "ts_code"].isin(stocks)

        # 没有成分数据的月份默认保留（避免因数据缺失误删）
        no_data_mask = ~df["_ym"].isin(monthly_set.keys())
        keep = keep | no_data_mask

        df = df[keep].drop(columns=["_ym"])
        after = df.shape[0]
        print(f"[指数过滤] {index_code} (动态成分): {before} -> {after} 条"
              f" ({len(monthly_set)} 个月份, 剔除 {before-after} 条非成分股)")
        return df.reset_index(drop=True)
