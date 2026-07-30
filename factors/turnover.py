import pandas as pd
from factors.base import BaseFactor


class TurnoverFactor(BaseFactor):
    name = "turnover"
    label = "换手率因子"
    window = 20
    category = "volume"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """已统一迁移至 FactorRegistry.compute_all，此处保留供独立调用。"""
        from factors.registry import FactorRegistry
        FactorRegistry._windows["turnover"] = self.window
        return df
