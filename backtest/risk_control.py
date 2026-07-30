"""
选股前过滤 + 持仓中止损/止盈 + 组合回撤熔断
"""
import pandas as pd
import numpy as np


class RiskController:

    def __init__(self,
                 min_price: float = 5.0,
                 min_volume: float = 10000,
                 min_amount: float = 500000,
                 block_boards: tuple = ("688",),
                 stop_loss: float = -0.08,
                 max_drawdown: float = -0.15,
                 max_daily_loss: float = -0.05):
        self.min_price = min_price
        self.min_volume = min_volume
        self.min_amount = min_amount
        self.block_boards = block_boards
        self.stop_loss = stop_loss
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.stop_loss_count = 0
        self.liquidate_count = 0
        self.total_holdings_checked = 0

    def pre_filter(self, candidates: pd.DataFrame) -> pd.DataFrame:
        before = candidates.shape[0]

        candidates = candidates[candidates["close"] >= self.min_price]

        if "vol" in candidates.columns:
            candidates = candidates[candidates["vol"] >= self.min_volume]

        if "amount" in candidates.columns:
            candidates = candidates[candidates["amount"] >= self.min_amount]

        candidates = candidates[
            ~candidates["ts_code"].str.startswith(self.block_boards)
        ]

        if "ret" in candidates.columns:
            candidates = candidates[candidates["ret"].abs() < 0.095]

        after = candidates.shape[0]
        print(f"  [风控-筛选] {before} -> {after} 只 (过滤 {before - after} 只)")
        return candidates

    def check_stop_loss(self, holdings: list, entry_prices: dict,
                         today_prices: dict) -> list:
        to_sell = []
        for stock in list(holdings):
            self.total_holdings_checked += 1
            if stock not in entry_prices or stock not in today_prices:
                continue
            cost = entry_prices[stock]
            today = today_prices[stock]
            pnl = (today - cost) / cost
            if pnl <= self.stop_loss:
                to_sell.append(stock)
                self.stop_loss_count += 1
        if to_sell:
            print(f"  [风控-止损] 触发 {len(to_sell)} 只: {to_sell[:5]}...")
        return to_sell

    def check_portfolio_risk(self, cum_nav: float, peak_nav: float,
                              daily_ret: float) -> dict:
        drawdown = (cum_nav - peak_nav) / peak_nav

        if drawdown <= self.max_drawdown:
            self.liquidate_count += 1
            return {"liquidate": True, "reason": f"回撤熔断({drawdown:.1%})"}

        if daily_ret <= self.max_daily_loss:
            self.liquidate_count += 1
            return {"liquidate": True, "reason": f"单日亏损熔断({daily_ret:.1%})"}

        return {"liquidate": False, "reason": ""}

    def stats(self) -> dict:
        return {
            "stop_loss_triggered": self.stop_loss_count,
            "total_holdings_checked": self.total_holdings_checked,
            "stop_loss_ratio": self.stop_loss_count / max(self.total_holdings_checked, 1),
            "liquidate_triggered": self.liquidate_count,
        }
