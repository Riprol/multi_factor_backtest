import pandas as pd
from typing import List, Type
from factors.base import BaseFactor
from utils.database import FactorDatabase


class FactorRegistry:

    _factors: dict = {}

    @classmethod
    def register(cls, factor: BaseFactor):
        cls._factors[factor.name] = factor
        print(f"[注册表] 已注册因子: {factor.name} ({factor.label})")

    @classmethod
    def get(cls, name: str) -> BaseFactor:
        if name not in cls._factors:
            raise KeyError(f"因子 '{name}' 未注册，可用: {list(cls._factors.keys())}")
        return cls._factors[name]

    @classmethod
    def list_all(cls) -> list:
        return list(cls._factors.keys())

    @classmethod
    def compute_all(cls, market_df: pd.DataFrame, db: FactorDatabase,
                    factor_names: List[str] = None):
        if factor_names is None:
            factor_names = cls.list_all()
        for name in factor_names:
            factor = cls.get(name)
            print(f"\n[因子计算] {factor.name} ({factor.label}) ...")
            factor_df = factor.compute(market_df)
            db.save_factor(factor.name, factor_df)
            print(f"  -> 已保存 {factor_df.shape[0]} 条记录")
