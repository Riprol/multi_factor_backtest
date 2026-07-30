"""
量价协同因子 —— 滚动窗口内价格与成交量的 Pearson 相关系数。

调用 rolling_kernels 模块的 numba 加速内核实现。
"""
import pandas as pd
from factors.base import BaseFactor
from utils.rolling_kernels import rolling_corr


class PriceVolumeCorrFactor(BaseFactor):
    name = "pv_corr"
    label = "量价协同因子"
    window = 60          # 约 3 个月交易日
    category = "volume"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        直接调用 rolling_kernels.rolling_corr，自动走 numba 加速。
        """
        corr_series = rolling_corr(
            df,
            price_col="close",
            volume_col="vol",
            group_col="ts_code",
            date_col="trade_date",
            window=self.window,
        )
        result = df[["ts_code", "trade_date"]].copy()
        result["value"] = corr_series.values
        return result.dropna(subset=["value"]).reset_index(drop=True)
