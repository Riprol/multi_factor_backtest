from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseFactor(ABC):

    name: str = "base"
    label: str = "基础因子"
    window: int = 20
    category: str = "other"

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @staticmethod
    def process_cross_section(df: pd.DataFrame,
                               value_col: str = "value",
                               date_col: str = "trade_date",
                               lower: float = 0.01,
                               upper: float = 0.99,
                               industry_col: str = None,
                               size_col: str = None) -> pd.DataFrame:
        """三合一因子预处理：缩尾 → Z-score → 行业+市值中性化。

        若提供 industry_col / size_col，则在 Z-score 后追加中性化。
        残差 = 剔除行业、市值风格后的纯净因子暴露。
        """
        df = df.copy()

        # 1. 截面缩尾
        from utils.stats_tools import winsorize_cross_section
        df[value_col] = winsorize_cross_section(df, value_col, date_col, lower, upper)

        # 2. 截面 Z-score
        g = df.groupby(date_col)[value_col]
        df[value_col] = (df[value_col] - g.transform("mean")) / g.transform("std")
        df = df.dropna(subset=[value_col])

        # 3. 行业+市值中性化（可选）
        if industry_col and industry_col in df.columns:
            from utils.regression import neutralize_industry_style
            for date, grp in df.groupby(date_col):
                if len(grp) < 30:
                    continue
                neutral = neutralize_industry_style(
                    grp, factor_name=value_col,
                    industry_col=industry_col, size_col=size_col
                )
                df.loc[grp.index, value_col] = neutral.values

        return df.dropna(subset=[value_col])
