import numpy as np
import pandas as pd
from factors.base import BaseFactor


class VolatilityFactor(BaseFactor):
    name = "volatility"
    label = "波动率因子"
    window = 20
    category = "volatility"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """已统一迁移至 FactorRegistry.compute_all，此处保留供独立调用。"""
        from factors.registry import FactorRegistry
        FactorRegistry._windows["volatility"] = self.window
        return df
