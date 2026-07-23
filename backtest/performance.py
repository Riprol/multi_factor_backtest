import pandas as pd
import numpy as np


class PerformanceEvaluator:

    @staticmethod
    def evaluate(daily_ret: pd.Series, rf: float = 0.02) -> dict:
        if daily_ret.empty or len(daily_ret) < 10:
            return {"error": "数据不足"}

        n = len(daily_ret)
        years = n / 252
        cum_nav = (1 + daily_ret).cumprod()
        total_ret = cum_nav.iloc[-1] - 1
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        annual_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (annual_ret - rf) / annual_vol if annual_vol > 0 else 0

        cum_max = cum_nav.cummax()
        drawdown = (cum_nav - cum_max) / cum_max
        max_dd = drawdown.min()
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

        win_rate = (daily_ret > 0).mean()
        avg_win = daily_ret[daily_ret > 0].mean() if (daily_ret > 0).any() else 0
        avg_loss = abs(daily_ret[daily_ret < 0].mean()) if (daily_ret < 0).any() else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        return {
            "累计收益率": f"{total_ret:.2%}",
            "年化收益率": f"{annual_ret:.2%}",
            "年化波动率": f"{annual_vol:.2%}",
            "夏普比率": f"{sharpe:.2f}",
            "最大回撤": f"{max_dd:.2%}",
            "Calmar比率": f"{calmar:.2f}",
            "胜率": f"{win_rate:.2%}",
            "盈亏比": f"{profit_loss_ratio:.2f}",
            "年数": f"{years:.1f}",
            "交易天数": n
        }

    @staticmethod
    def compare_factors(perf_dicts: dict) -> pd.DataFrame:
        return pd.DataFrame(perf_dicts).T
