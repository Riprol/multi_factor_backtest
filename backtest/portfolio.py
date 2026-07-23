import pandas as pd
import numpy as np
from config import TOP_N, REBALANCE_FREQ, COMMISSION, SLIPPAGE
from data.calendar import TradeCalendar
from utils.database import FactorDatabase


class PortfolioBuilder:

    def __init__(self, db: FactorDatabase):
        self.db = db
        self.cal = TradeCalendar()
        self.market_df = None

    def _load_market(self, start: str, end: str):
        if self.market_df is None:
            df = self.db.load_market(start, end)
            df = df.sort_values(["ts_code", "trade_date"])
            self.market_df = df
        return self.market_df

    def run(self, factor_name: str, start: str, end: str,
            top_n: int = TOP_N, rebalance: str = REBALANCE_FREQ) -> pd.DataFrame:
        factor_df = self.db.load_factor(factor_name, start, end)
        market = self._load_market(start, end)

        rebalance_dates = self.cal.get_rebalance_dates(start, end, rebalance)
        all_dates = self.cal.get_range(start, end)

        holdings_map = {}
        for rb_date in rebalance_dates:
            day_factor = factor_df[factor_df["trade_date"] == rb_date]
            if day_factor.empty:
                continue
            top_stocks = day_factor.nlargest(top_n, "value")["ts_code"].tolist()
            holdings_map[rb_date] = top_stocks

        if not holdings_map:
            print("  [组合构建] 无有效调仓日")
            return pd.DataFrame()

        records = []
        current_holdings = []
        for i, date in enumerate(all_dates):
            if date in holdings_map:
                current_holdings = holdings_map[date]
                turnover_cost = COMMISSION * 2 if i > 0 and current_holdings else COMMISSION

            if not current_holdings:
                records.append({"trade_date": date, "portfolio_ret": 0.0})
                continue

            day_market = market[market["trade_date"] == date]
            day_hold = day_market[day_market["ts_code"].isin(current_holdings)]
            if day_hold.empty:
                records.append({"trade_date": date, "portfolio_ret": 0.0})
                continue

            avg_ret = day_hold["ret"].mean()
            if date in holdings_map:
                avg_ret = avg_ret - SLIPPAGE - turnover_cost
            records.append({"trade_date": date, "portfolio_ret": avg_ret})

        portfolio_df = pd.DataFrame(records)
        portfolio_df["trade_date"] = pd.to_datetime(portfolio_df["trade_date"])
        portfolio_df["cum_nav"] = (1 + portfolio_df["portfolio_ret"]).cumprod()
        return portfolio_df
