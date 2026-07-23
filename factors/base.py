from abc import ABC, abstractmethod
import pandas as pd


class BaseFactor(ABC):

    name: str = "base"
    label: str = "基础因子"
    window: int = 20
    category: str = "other"

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        ...
