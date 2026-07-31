import pandas as pd
import numpy as np
from config import (
    TOP_N, REBALANCE_FREQ, COMMISSION, SLIPPAGE, WEIGHT_METHOD,
    RISK_MIN_PRICE, RISK_MIN_VOLUME, RISK_MIN_AMOUNT,
    RISK_BLOCK_BOARDS, RISK_STOP_LOSS,
    RISK_MAX_DRAWDOWN, RISK_MAX_DAILY_LOSS
)
from data.calendar import TradeCalendar
from utils.database import FactorDatabase
from backtest.risk_control import RiskController


class PortfolioBuilder:

    def __init__(self, db: FactorDatabase):
        self.db = db
        self.cal = TradeCalendar()
        self.market_df = None
        self.risk = RiskController(
            min_price=RISK_MIN_PRICE,
            min_volume=RISK_MIN_VOLUME,
            min_amount=RISK_MIN_AMOUNT,
            block_boards=RISK_BLOCK_BOARDS,
            stop_loss=RISK_STOP_LOSS,
            max_drawdown=RISK_MAX_DRAWDOWN,
            max_daily_loss=RISK_MAX_DAILY_LOSS,
        )

    def _load_market(self, start: str, end: str):
        if self.market_df is None:
            df = self.db.load_market(start, end)
            df = df.sort_values(["ts_code", "trade_date"])
            self.market_df = df
            self._market_by_date = {}
            for date, grp in df.groupby("trade_date"):
                self._market_by_date[date] = grp
        return self.market_df

    def get_benchmark_returns(self, start: str, end: str) -> pd.Series:
        """全市场等权日收益，作为基准"""
        market = self._load_market(start, end)
        bench = market.groupby("trade_date")["ret"].mean()
        bench.index = pd.to_datetime(bench.index)
        return bench

    def run(self, factor_name: str, start: str, end: str,
            top_n: int = TOP_N, rebalance: str = REBALANCE_FREQ) -> dict:
        factor_df = self.db.load_factor(factor_name, start, end)
        market = self._load_market(start, end)

        rebalance_dates = self.cal.get_rebalance_dates(start, end, rebalance)
        all_dates = self.cal.get_range(start, end)

        holdings_map = {}
        for rb_date in rebalance_dates:
            day_factor = factor_df[factor_df["trade_date"] == rb_date]
            # 时间对齐：signal_date=rb_date，用当日盘后因子值选股
            # return_date 从 rb_date+1 起，由 daily_ret 逐日累加
            if day_factor.empty:
                continue

            day_market = self._market_by_date.get(rb_date)
            if day_market is not None:
                day_factor = day_factor.merge(
                    day_market[["ts_code", "close", "vol", "amount", "ret"]],
                    on="ts_code", how="left"
                )
                day_factor = self.risk.pre_filter(day_factor)

            if day_factor.empty:
                continue

            selected = day_factor.nlargest(top_n, "value")
            top_stocks = selected["ts_code"].tolist()

            if WEIGHT_METHOD == "amount" and "amount" in selected.columns:
                raw_w = selected["amount"].fillna(selected["amount"].median())
                raw_w = raw_w.clip(lower=raw_w.quantile(0.05))
                weights = (raw_w / raw_w.sum()).to_dict()
            else:
                weights = {s: 1.0 / len(top_stocks) for s in top_stocks}

            holdings_map[rb_date] = {"stocks": top_stocks, "weights": weights}

        if not holdings_map:
            print("  [组合构建] 无有效调仓日")
            return {"portfolio_df": pd.DataFrame(), "risk_stats": self.risk.stats()}

        records = []
        current_holdings = []
        current_weights = {}
        entry_prices = {}
        peak_nav = 1.0
        cum_nav = 1.0
        liquidated = False

        for i, date in enumerate(all_dates):
            if liquidated:
                records.append({"trade_date": date, "portfolio_ret": 0.0})
                continue

            day_market = self._market_by_date.get(date)
            if day_market is None:
                records.append({"trade_date": date, "portfolio_ret": 0.0})
                continue

            today_prices = dict(zip(day_market["ts_code"], day_market["close"]))

            if date in holdings_map:
                h = holdings_map[date]
                current_holdings = h["stocks"]
                current_weights = h["weights"]
                for stock in current_holdings:
                    if stock in today_prices and stock not in entry_prices:
                        entry_prices[stock] = today_prices[stock]
                turnover_cost = COMMISSION * 2 if i > 0 and current_holdings else COMMISSION

            to_sell = self.risk.check_stop_loss(current_holdings, entry_prices, today_prices)
            for stock in to_sell:
                if stock in entry_prices:
                    del entry_prices[stock]
                if stock in current_weights:
                    del current_weights[stock]
            current_holdings = [s for s in current_holdings if s not in to_sell]

            if not current_holdings:
                records.append({"trade_date": date, "portfolio_ret": 0.0})
                continue

            day_hold = day_market[day_market["ts_code"].isin(current_holdings)]
            if day_hold.empty:
                records.append({"trade_date": date, "portfolio_ret": 0.0})
                continue

            ret_map = dict(zip(day_hold["ts_code"], day_hold["ret"]))
            total_w = sum(current_weights.get(s, 0) for s in current_holdings)
            if total_w > 0:
                portfolio_ret = sum(
                    ret_map.get(s, 0) * current_weights.get(s, 0) / total_w
                    for s in current_holdings
                )
            else:
                portfolio_ret = np.mean([ret_map.get(s, 0) for s in current_holdings])

            if date in holdings_map:
                portfolio_ret = portfolio_ret - SLIPPAGE - turnover_cost

            cum_nav *= (1 + portfolio_ret)
            peak_nav = max(peak_nav, cum_nav)
            risk_check = self.risk.check_portfolio_risk(cum_nav, peak_nav, portfolio_ret)
            if risk_check["liquidate"]:
                print(f"  [风控-熔断] {date} {risk_check['reason']}，清仓")
                current_holdings = []
                current_weights = {}
                entry_prices = {}
                liquidated = True

            records.append({"trade_date": date, "portfolio_ret": portfolio_ret})

        portfolio_df = pd.DataFrame(records)
        portfolio_df["trade_date"] = pd.to_datetime(portfolio_df["trade_date"])
        portfolio_df["cum_nav"] = (1 + portfolio_df["portfolio_ret"]).cumprod()
        return {"portfolio_df": portfolio_df, "risk_stats": self.risk.stats()}
