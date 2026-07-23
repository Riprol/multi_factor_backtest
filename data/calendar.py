import datetime
import os
import pandas as pd
from config import CAL_FILE


class TradeCalendar:

    def __init__(self, cal_file: str = CAL_FILE):
        self.cal_file = cal_file
        if not os.path.exists(cal_file):
            self._generate()

    def _generate(self):
        start = datetime.date(2010, 1, 1)
        end   = datetime.date(2030, 12, 31)
        dates = []
        cur = start
        while cur <= end:
            if cur.weekday() < 5:
                dates.append(cur.strftime("%Y%m%d"))
            cur += datetime.timedelta(days=1)
        pd.DataFrame({"trade_date": dates}).to_csv(self.cal_file, index=False)
        print(f"[交易日历] 已生成 {self.cal_file}")

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
