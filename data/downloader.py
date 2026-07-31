import os
import time
import pandas as pd
import tushare as ts
from config import TUSHARE_TOKEN, CACHE_DIR
from data.calendar import TradeCalendar
from utils.helpers import ensure_dir, timer


class DataDownloader:

    def __init__(self, token: str = TUSHARE_TOKEN):
        self.pro = ts.pro_api(token)
        self.cal = TradeCalendar(pro=self.pro)
        ensure_dir(CACHE_DIR)

    @timer
    def download_range(self, start: str, end: str, db=None) -> pd.DataFrame:
        """下载 [start, end] 日线。优先从 SQLite 读已有数据，只下载缺失日期。"""
        trade_dates = self.cal.get_range(start, end)

        # 从 SQLite 查已有日期
        existing_df = pd.DataFrame()
        missing_dates = trade_dates
        if db is not None:
            missing_dates = db.get_missing_dates(trade_dates)
            have_dates = [d for d in trade_dates if d not in missing_dates]
            if have_dates:
                existing_df = db.load_market_by_dates(have_dates)
                print(f"[下载器] SQLite 已有 {len(have_dates)} 天, 缺 {len(missing_dates)} 天")

        if not missing_dates:
            print("[下载器] 所有日期已入库，跳过下载")
            return existing_df

        print(f"[下载器] 需下载 {len(missing_dates)} 个交易日 ...")
        all_frames = []
        for i, day in enumerate(missing_dates):
            try:
                df = self.pro.daily(trade_date=day)
                if df is not None and not df.empty:
                    all_frames.append(df)
            except Exception as e:
                print(f"  [警告] {day} 下载失败: {e}")
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(missing_dates)}")
            time.sleep(1.2)

        new_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

        if db is not None and not new_df.empty:
            cols = ["ts_code", "trade_date", "open", "high", "low", "close",
                    "pre_close", "vol", "amount"]
            sub = new_df[cols].copy()
            sub["ret"] = sub["close"] / sub["pre_close"] - 1
            db.save_market_raw(sub)
            print(f"  [下载器] 新数据已实时入库: {sub.shape[0]} 行")

        # 标记所有尝试过的日期，即使 Tushare 返回空（假期等）也不反复下载
        if db is not None and missing_dates:
            db.mark_dates_downloaded(missing_dates)

        cache_path = os.path.join(CACHE_DIR, f"daily_{start}_{end}.csv")
        full = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
        full.to_csv(cache_path, index=False)

        return full

    @timer
    def download_index_weights(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """Tushare index_weight API 拉取指数成分股权重。"""
        if index_code is None:
            return pd.DataFrame()

        csv_path = os.path.join(CACHE_DIR, f"index_weights_{index_code}_{start}_{end}.csv")
        if os.path.exists(csv_path):
            print(f"[指数成分] 命中缓存: {csv_path}")
            return pd.read_csv(csv_path, dtype={"con_code": str, "trade_date": str})

        result = pd.DataFrame()
        cal = TradeCalendar()
        all_dates = cal.get_range(start, end)
        s = pd.Series(pd.to_datetime(all_dates))
        monthly_dates = [all_dates[i] for i in s.groupby(s.dt.to_period("M")).idxmax()]
        print(f"[指数成分] Tushare API，共 {len(monthly_dates)} 个月 ...")
        frames = []
        for i, mdate in enumerate(monthly_dates):
            try:
                df = self.pro.index_weight(index_code=index_code, trade_date=mdate)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                break
            time.sleep(0.6)
        if frames:
            result = pd.concat(frames, ignore_index=True)

        # ── 方式3：本地 CSV ──
        if result.empty:
            local_csv = os.path.join(CACHE_DIR, f"hs300_constituents_{start}_{end}.csv")
            if os.path.exists(local_csv):
                print(f"[指数成分] 使用本地文件: {local_csv}")
                result = pd.read_csv(local_csv, dtype={"con_code": str, "trade_date": str})

        if not result.empty:
            result.to_csv(csv_path, index=False)
            print(f"[指数成分] 已缓存 {result.shape[0]} 条")
        else:
            print("[指数成分] 无数据源，回退全市场模式")

        return result

    def download_stock_basic(self, db=None) -> pd.DataFrame:
        """下载全市场股票基础信息（list_date, name, industry），缓存到 SQLite。"""
        csv_path = os.path.join(CACHE_DIR, "stock_basic.csv")
        if os.path.exists(csv_path):
            print("[股票信息] 命中缓存")
            return pd.read_csv(csv_path, dtype={"ts_code": str})

        print("[股票信息] 从 Tushare 下载 stock_basic ...")
        fields = "ts_code,name,industry,list_date,list_status"
        df = self.pro.stock_basic(exchange="", list_status="L", fields=fields)
        if df is not None and not df.empty:
            df["list_date"] = df["list_date"].astype(str)
            df.to_csv(csv_path, index=False)
            print(f"[股票信息] 已缓存 {len(df)} 只")
            return df
        return pd.DataFrame()
