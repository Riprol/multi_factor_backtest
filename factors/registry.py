import pandas as pd
from typing import List, Type
from factors.base import BaseFactor
from utils.database import FactorDatabase


class FactorRegistry:

    _factors: dict = {}
    _windows: dict = {}  # 缓存因子窗口参数

    @classmethod
    def register(cls, factor: BaseFactor):
        cls._factors[factor.name] = factor
        cls._windows[factor.name] = getattr(factor, "window", 20)
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
        """
        统一向量化计算所有因子。
        只排序一次 + 一次 groupby 内批量 rolling，禁用 Python lambda。
        """
        if factor_names is None:
            factor_names = cls.list_all()
        print(f"\n[因子计算] 统一向量化模式，批量计算 {len(factor_names)} 个因子 ...")

        # ── 1. 排序一次，预处理派生列（全向量化，无循环） ──
        df = market_df.sort_values(["ts_code", "trade_date"]).copy()
        df["vol_mean"] = df.groupby("ts_code")["vol"].transform("mean")
        df["daily_turn"] = df["vol"] / df["vol_mean"]          # 标准化换手率
        df["illiq"] = df["ret"].abs() / (df["amount"] / 10000 + 1)  # 非流动性

        # ── 2. 统一 groupby，用 C 级 rolling 内置聚合计算所有因子 ──
        g = df.groupby("ts_code")

        # 动量: ret 20 日滚动求和
        if "momentum" in factor_names:
            w = cls._windows.get("momentum", 20)
            df["mom_val"] = g["ret"].rolling(w, min_periods=max(1, int(w * 0.6))).sum().values

        # 反转: -ret 5 日滚动求和
        if "reversal" in factor_names:
            w = cls._windows.get("reversal", 5)
            df["rev_val"] = -g["ret"].rolling(w, min_periods=max(1, int(w * 0.6))).sum().values

        # 波动率: -ret 20 日滚动标准差
        if "volatility" in factor_names:
            w = cls._windows.get("volatility", 20)
            df["vol_val"] = -g["ret"].rolling(w, min_periods=max(1, int(w * 0.6))).std().values

        # 换手率: -daily_turn 20 日滚动均值
        if "turnover" in factor_names:
            w = cls._windows.get("turnover", 20)
            df["turn_val"] = -g["daily_turn"].rolling(w, min_periods=max(1, int(w * 0.6))).mean().values

        # 流动性: illiq 20 日滚动均值（非流动性越高越差，不取反）
        if "liquidity" in factor_names:
            w = cls._windows.get("liquidity", 20)
            df["liq_val"] = g["illiq"].rolling(w, min_periods=max(1, int(w * 0.6))).mean().values

        # ── 3. 截面清洗 + 保存 ──
        col_map = {
            "momentum":   "mom_val",
            "reversal":   "rev_val",
            "volatility": "vol_val",
            "turnover":   "turn_val",
            "liquidity":  "liq_val",
        }
        total_before = 0
        total_after = 0
        for name in factor_names:
            if name not in col_map:
                continue
            col = col_map[name]
            sub = df[["ts_code", "trade_date", col]].dropna(subset=[col])
            sub = sub.rename(columns={col: "value"})
            total_before += sub.shape[0]

            # 截面缩尾 + 标准化（BaseFactor 统一封装）
            sub = BaseFactor.process_cross_section(sub)
            total_after += sub.shape[0]

            db.save_factor(name, sub)
            print(f"  -> {name} 已保存 {sub.shape[0]} 条记录")

        if total_before > 0:
            print(f"  [截面清洗] 缩尾+标准化完成, {total_before} -> {total_after} 条")
