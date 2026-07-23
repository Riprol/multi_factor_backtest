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
        self.cal = TradeCalendar()
        ensure_dir(CACHE_DIR)

    @timer
    def download_range(self, start: str, end: str) -> pd.DataFrame:
        cache_path = os.path.join(CACHE_DIR, f"daily_{start}_{end}.csv")
        if os.path.exists(cache_path):
            print(f"[下载器] 命中缓存: {cache_path}")
            return pd.read_csv(cache_path, dtype={"ts_code": str, "trade_date": str})

        trade_dates = self.cal.get_range(start, end)
        print(f"[下载器] 共 {len(trade_dates)} 个交易日，开始下载...")
        all_frames = []
        for i, day in enumerate(trade_dates):
            try:
                df = self.pro.daily(trade_date=day)
                if df is not None and not df.empty:
                    all_frames.append(df)
            except Exception as e:
                print(f"  [警告] {day} 下载失败: {e}")
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(trade_dates)}")
            time.sleep(1.2)

        if not all_frames:
            raise RuntimeError("未下载到任何数据，请检查 Token 或网络！")

        result = pd.concat(all_frames, ignore_index=True)
        result.to_csv(cache_path, index=False)
        print(f"[下载器] 数据已缓存至 {cache_path}")
        return result
