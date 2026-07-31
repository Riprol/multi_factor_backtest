import datetime
import os
import time
import pandas as pd
from config import CAL_FILE


class TradeCalendar:

    def __init__(self, cal_file: str = CAL_FILE, pro=None):
        self.cal_file = cal_file
        if not os.path.exists(cal_file):
            self._generate(pro)

    def _generate(self, pro=None):
        """优先生成 Tushare 真实交易日历（排除节假日），降级为工作日生成。"""
        df = pd.DataFrame()

        if pro is not None:
            try:
                print("[交易日历] 从 Tushare trade_cal 下载 ...")
                raw = pro.trade_cal(exchange="SSE", start_date="20100101",
                                    end_date="20301231")
                if raw is not None and not raw.empty:
                    df = raw[raw["is_open"] == 1][["cal_date"]].copy()
                    df.columns = ["trade_date"]
                    df["trade_date"] = df["trade_date"].astype(str)
                    print(f"[交易日历] Tushare 成功: {len(df)} 个交易日")
            except Exception as e:
                print(f"[交易日历] Tushare 失败: {e}")

        if df.empty:
            print("[交易日历] 降级为工作日生成（不含节假日）")
            start = datetime.date(2010, 1, 1)
            end   = datetime.date(2030, 12, 31)
            dates = []
            cur = start
            while cur <= end:
                if cur.weekday() < 5:
                    dates.append(cur.strftime("%Y%m%d"))
                cur += datetime.timedelta(days=1)
            df = pd.DataFrame({"trade_date": dates})

        df.to_csv(self.cal_file, index=False)
        print(f"[交易日历] 已保存 {self.cal_file}")

    def get_range(self, start: str, end: str) -> list:
        cal = pd.read_csv(self.cal_file, dtype={"trade_date": str})
        mask = (cal["trade_date"] >= start) & (cal["trade_date"] <= end)
        return cal.loc[mask, "trade_date"].tolist()

    def get_rebalance_dates(self, start: str, end: str, freq: str = "M") -> list:
        all_dates = self.get_range(start, end)
        if not all_dates:
            return []
        s = pd.Series(pd.to_datetime(all_dates))
        if freq == "M":
            group = s.dt.to_period("M")
            idx = s.groupby(group).idxmax()
            return [all_dates[i] for i in idx]
        elif freq == "W":
            group = s.dt.to_period("W")
            idx = s.groupby(group).idxmax()
            return [all_dates[i] for i in idx]
        else:
            return all_dates
