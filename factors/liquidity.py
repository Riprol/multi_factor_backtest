import pandas as pd
import numpy as np
from factors.base import BaseFactor


class LiquidityFactor(BaseFactor):
    name = "liquidity"
    label = "流动性因子"
    window = 20
    category = "volume"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """已统一迁移至 FactorRegistry.compute_all，此处保留供独立调用。"""
        from factors.registry import FactorRegistry
        FactorRegistry._windows["liquidity"] = self.window
        return df
