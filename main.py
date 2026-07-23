"""
多因子回测框架 v1.0
改 config.py 里的 token 和日期，然后 python main.py 一键跑通
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config import (
    TUSHARE_TOKEN, START_DATE, END_DATE,
    LAYER_NUM, TOP_N, REBALANCE_FREQ, OUTPUT_DIR
)

from data.calendar import TradeCalendar
from data.downloader import DataDownloader
from data.cleaner import DataCleaner

from factors.base import BaseFactor
from factors.registry import FactorRegistry
from factors.momentum import MomentumFactor
from factors.reversal import ReversalFactor
from factors.volatility import VolatilityFactor
from factors.turnover import TurnoverFactor
from factors.liquidity import LiquidityFactor
from factors.composite import CompositeFactor

from backtest.ic_analysis import ICAnalyzer
from backtest.layer_backtest import LayerBacktester
from backtest.portfolio import PortfolioBuilder
from backtest.performance import PerformanceEvaluator

from visualization.charts import ChartDrawer
from visualization.report import ReportGenerator

from utils.database import FactorDatabase
from utils.helpers import timer, ensure_dir


def register_all_factors():
    FactorRegistry.register(MomentumFactor())
    FactorRegistry.register(ReversalFactor())
    FactorRegistry.register(VolatilityFactor())
    FactorRegistry.register(TurnoverFactor())
    FactorRegistry.register(LiquidityFactor())
    print(f"\n已注册 {len(FactorRegistry.list_all())} 个因子: {FactorRegistry.list_all()}")


@timer
def prepare_data():
    downloader = DataDownloader(token=TUSHARE_TOKEN)
    raw_df = downloader.download_range(START_DATE, END_DATE)
    clean_df = DataCleaner.clean(raw_df)
    db = FactorDatabase()
    db.save_market(clean_df)
    print(f"[数据] 已入库 {clean_df.shape[0]} 条行情记录")
    return db, clean_df


@timer
def compute_factors(db: FactorDatabase, clean_df: pd.DataFrame):
    FactorRegistry.compute_all(clean_df, db)
    print(f"\n[因子] 数据库中已有因子: {db.list_factors()}")


def run_ic_analysis(db: FactorDatabase):
    print("\n" + "=" * 60)
    print("  IC 分析")
    print("=" * 60)

    analyzer = ICAnalyzer(db)
    factor_names = FactorRegistry.list_all()

    all_ic = analyzer.compute_all_factors_ic(factor_names, START_DATE, END_DATE)
    if all_ic.empty:
        print("[IC] 无有效数据，跳过")
        return None, None, None

    summary = analyzer.ic_summary(all_ic)
    print("\n[IC 汇总]")
    print(summary.to_string(index=False))

    ic_csv_path = os.path.join(OUTPUT_DIR, "ic_summary.csv")
    all_ic.to_csv(ic_csv_path, index=False)
    print(f"\n[IC] IC明细已保存至 {ic_csv_path}")

    return analyzer, all_ic, summary


def run_backtest(db: FactorDatabase):
    print("\n" + "=" * 60)
    print("  分层回测 & 组合构建")
    print("=" * 60)

    layer = LayerBacktester(db, layer_num=LAYER_NUM)
    portfolio = PortfolioBuilder(db)
    chart = ChartDrawer()

    factor_names = FactorRegistry.list_all()
    layer_results = {}
    perf_results = {}

    for name in factor_names:
        factor = FactorRegistry.get(name)
        print(f"\n--- {factor.label} ({factor.name}) ---")

        lr = layer.run(factor.name, START_DATE, END_DATE)
        layer_results[factor.label] = lr
        if "cum_net_value" in lr:
            chart.plot_layer_net_value(lr["cum_net_value"], factor.label)

        analyzer = ICAnalyzer(db)
        ic_series = analyzer.compute_ic_series(factor.name, START_DATE, END_DATE)
        if not ic_series.empty:
            chart.plot_ic_series(ic_series, factor.label)

        pf = portfolio.run(factor.name, START_DATE, END_DATE,
                          top_n=TOP_N, rebalance=REBALANCE_FREQ)
        if not pf.empty:
            chart.plot_portfolio_nav(pf, factor.label)
            perf = PerformanceEvaluator.evaluate(pf["portfolio_ret"])
            perf_results[factor.label] = perf
            print(f"  [绩效] {factor.label}: 年化={perf.get('年化收益率','N/A')}, "
                  f"夏普={perf.get('夏普比率','N/A')}, 最大回撤={perf.get('最大回撤','N/A')}")

    return layer_results, perf_results


def run_composite_factor(db: FactorDatabase, ic_summary: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  多因子合成")
    print("=" * 60)

    factor_names = FactorRegistry.list_all()
    if len(factor_names) < 2:
        print("[合成] 因子不足2个，跳过多因子合成")
        return

    composite = CompositeFactor(db)
    weights = composite.ic_weighted(factor_names, ic_summary)
    print(f"[合成] IC加权权重: {weights}")

    cal = TradeCalendar()
    all_dates = cal.get_range(START_DATE, END_DATE)
    comp_records = []
    for date in all_dates:
        day_comp = composite.compute_daily(factor_names, date, method="weighted", weights=weights)
        if not day_comp.empty:
            comp_records.append(day_comp)

    if comp_records:
        comp_df = pd.concat(comp_records, ignore_index=True)
        db.save_factor("composite", comp_df)
        print(f"[合成] 复合因子已保存，共 {comp_df.shape[0]} 条")

        layer = LayerBacktester(db)
        chart = ChartDrawer()
        lr = layer.run("composite", START_DATE, END_DATE)
        if "cum_net_value" in lr:
            chart.plot_layer_net_value(lr["cum_net_value"], "复合因子(IC加权)")

        portfolio = PortfolioBuilder(db)
        pf = portfolio.run("composite", START_DATE, END_DATE)
        if not pf.empty:
            chart.plot_portfolio_nav(pf, "复合因子(IC加权)")
            perf = PerformanceEvaluator.evaluate(pf["portfolio_ret"])
            print(f"\n[复合因子绩效] 年化={perf.get('年化收益率','N/A')}, "
                  f"夏普={perf.get('夏普比率','N/A')}, 最大回撤={perf.get('最大回撤','N/A')}")


def main():
    print("=" * 60)
    print("  多因子回测框架 v1.0")
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    ensure_dir(OUTPUT_DIR)
    register_all_factors()
    db, clean_df = prepare_data()
    compute_factors(db, clean_df)
    analyzer, all_ic, ic_summary = run_ic_analysis(db)
    layer_results, perf_results = run_backtest(db)

    if ic_summary is not None:
        ic_for_weight = ic_summary.rename(columns={
            "factor_name": "factor_name",
            "ic_abs_mean": "ic_abs_mean"
        })
        run_composite_factor(db, ic_for_weight)

    if all_ic is not None and not all_ic.empty:
        chart = ChartDrawer()
        chart.plot_ic_comparison(all_ic)

    if perf_results:
        perf_df = pd.DataFrame(perf_results).T
        report = ReportGenerator()
        report.generate(ic_summary if ic_summary is not None else pd.DataFrame(),
                       perf_df, layer_results)

    db.close()

    print("\n" + "=" * 60)
    print("  回测完成！请查看 output/ 目录下的图表和报告")
    print("=" * 60)


if __name__ == "__main__":
    main()
