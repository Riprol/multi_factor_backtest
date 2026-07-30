import pandas as pd
from factors.base import BaseFactor


class MomentumFactor(BaseFactor):
    name = "momentum"
    label = "动量因子"
    window = 20
    category = "momentum"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """已统一迁移至 FactorRegistry.compute_all，此处保留供独立调用。"""
        from factors.registry import FactorRegistry
        from utils.database import FactorDatabase
        import sqlite3, os
        FactorRegistry._windows["momentum"] = self.window
        return df
