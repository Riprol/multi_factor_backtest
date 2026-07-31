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
    LAYER_NUM, TOP_N, REBALANCE_FREQ, OUTPUT_DIR, INDEX_UNIVERSE
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
# 复杂因子示例（需 numba）：
# from factors.price_volume_corr import PriceVolumeCorrFactor

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
    db = FactorDatabase()
    downloader = DataDownloader(token=TUSHARE_TOKEN)
    raw_df = downloader.download_range(START_DATE, END_DATE, db=db)

    stock_basic = downloader.download_stock_basic()
    clean_df = DataCleaner.clean(raw_df, stock_basic=stock_basic)
    db.save_market(clean_df)

    # ── 指数成分股：拉取 → 存 SQLite → 截面过滤 ──
    if INDEX_UNIVERSE:
        iw_df = downloader.download_index_weights(INDEX_UNIVERSE, START_DATE, END_DATE)
        if not iw_df.empty:
            db.save_index_weights(iw_df, INDEX_UNIVERSE)
            before = clean_df.shape[0]
            clean_df = DataCleaner.filter_by_index_db(clean_df, db, INDEX_UNIVERSE)
            # 过滤后重新保存行情（仅成分股）
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

    summary = analyzer.ic_summary_with_ci(all_ic)
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
    bench_ret = portfolio.get_benchmark_returns(START_DATE, END_DATE)

    factor_names = FactorRegistry.list_all()
    layer_results = {}
    perf_results = {}
    risk_stats_all = {}

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
            chart.plot_ic_distribution(ic_series, factor.label)

        if "daily_layer_ret" in lr:
            chart.plot_group_boxplot(lr["daily_layer_ret"], factor.label)

        pf_result = portfolio.run(factor.name, START_DATE, END_DATE,
                          top_n=TOP_N, rebalance=REBALANCE_FREQ)
        pf = pf_result["portfolio_df"]
        risk_stats_all[factor.label] = pf_result["risk_stats"]
        if not pf.empty:
            chart.plot_portfolio_nav(pf, factor.label, benchmark_ret=bench_ret)
            perf = PerformanceEvaluator.evaluate(pf["portfolio_ret"])
            perf_results[factor.label] = perf
            print(f"  [绩效] {factor.label}: 年化={perf.get('年化收益率','N/A')}, "
                  f"夏普={perf.get('夏普比率','N/A')}, 最大回撤={perf.get('最大回撤','N/A')}")

    return layer_results, perf_results, risk_stats_all


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
        pf_result = portfolio.run("composite", START_DATE, END_DATE)
        pf = pf_result["portfolio_df"]
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
    layer_results, perf_results, risk_stats_all = run_backtest(db)

    yearly_ic = None
    if analyzer is not None and all_ic is not None and not all_ic.empty:
        yearly_ic = analyzer.yearly_summary(all_ic)
        print("\n[IC 分年统计]")
        for fname in yearly_ic["factor_name"].unique():
            sub = yearly_ic[yearly_ic["factor_name"] == fname]
            print(f"\n  {fname}:")
            print(sub[["year", "ic_mean", "icir"]].to_string(index=False))

    if ic_summary is not None:
        ic_for_weight = ic_summary.rename(columns={
            "factor_name": "factor_name",
            "ic_abs_mean": "ic_abs_mean"
        })
        run_composite_factor(db, ic_for_weight)

    if all_ic is not None and not all_ic.empty:
        chart = ChartDrawer()
        chart.plot_ic_comparison(all_ic)

    # 因子相关性矩阵
    factor_corr = None
    try:
        mid_date = sorted(db.get_existing_dates())[len(db.get_existing_dates()) // 2]
        wide = db.load_all_factors(FactorRegistry.list_all(), mid_date)
        if not wide.empty:
            from utils.stats_tools import factor_corr_matrix
            factor_corr = factor_corr_matrix(wide)
            print("\n[因子相关性]")
            print(factor_corr.round(4).to_string())
    except Exception:
        pass

    if perf_results:
        perf_df = pd.DataFrame(perf_results).T
        report = ReportGenerator()
        report.generate(ic_summary if ic_summary is not None else pd.DataFrame(),
                       perf_df, layer_results,
                       yearly_ic=yearly_ic, risk_stats=risk_stats_all,
                       factor_corr=factor_corr)

    db.close()

    print("\n" + "=" * 60)
    print("  回测完成！请查看 output/ 目录下的图表和报告")
    print("=" * 60)


if __name__ == "__main__":
    main()
