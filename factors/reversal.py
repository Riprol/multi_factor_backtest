import pandas as pd
from factors.base import BaseFactor


class ReversalFactor(BaseFactor):
    name = "reversal"
    label = "反转因子"
    window = 5
    category = "reversal"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """已统一迁移至 FactorRegistry.compute_all，此处保留供独立调用。"""
        from factors.registry import FactorRegistry
        FactorRegistry._windows["reversal"] = self.window
        return df
