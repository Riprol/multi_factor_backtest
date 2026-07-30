import os
import pandas as pd
from datetime import datetime
from config import OUTPUT_DIR
from utils.helpers import ensure_dir


class ReportGenerator:

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        ensure_dir(output_dir)

    def generate(self,
                 ic_summary: pd.DataFrame,
                 perf_comparison: pd.DataFrame,
                 layer_results: dict,
                 yearly_ic: pd.DataFrame = None,
                 risk_stats: dict = None,
                 factor_corr: pd.DataFrame = None,
                 factor_desc: dict = None) -> str:
        lines = []
        lines.append("# 多因子回测报告")
        lines.append(f"\n> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("## 一、因子 IC 分析\n")
        lines.append("| 因子 | IC均值 | ICIR | IC>0占比 | t值 | p值 | 95%CI下 | 95%CI上 | 0在区间内 | 样本 |")
        lines.append("|------|--------|------|----------|-----|-----|---------|---------|-----------|------|")
        for _, row in ic_summary.iterrows():
            ci_low = row.get("ci_lower", float("nan"))
            ci_high = row.get("ci_upper", float("nan"))
            zero_in = row.get("zero_inside", None)
            flag = "Yes" if zero_in else ("No" if zero_in is False else "-")
            tv = row.get("t_value", float("nan"))
            pv = row.get("p_value", float("nan"))
            lines.append(
                f"| {row['factor_name']} "
                f"| {row['ic_mean']:.4f} "
                f"| {row['icir']:.2f} "
                f"| {row['ic_positive_ratio']:.1%} "
                f"| {tv:.2f} "
                f"| {pv:.4f} "
                f"| {ci_low:.4f} "
                f"| {ci_high:.4f} "
                f"| {flag} "
                f"| {int(row['obs_count'])} |"
            )

        if yearly_ic is not None and not yearly_ic.empty:
            lines.append("\n### IC 分年统计\n")
            lines.append("| 因子 | 年份 | IC均值 | ICIR | IC>0占比 | 样本 |")
            lines.append("|------|------|--------|------|----------|------|")
            for _, row in yearly_ic.iterrows():
                lines.append(
                    f"| {row['factor_name']} "
                    f"| {int(row['year'])} "
                    f"| {row['ic_mean']:.4f} "
                    f"| {row['icir']:.2f} "
                    f"| {row['ic_positive_ratio']:.1%} "
                    f"| {int(row['obs_count'])} |"
                )

        if factor_corr is not None and not factor_corr.empty:
            lines.append("\n### 因子相关性矩阵\n")
            lines.append(factor_corr.round(4).to_markdown())

        lines.append("\n## 二、组合绩效对比\n")
        lines.append(perf_comparison.to_markdown())

        if risk_stats:
            lines.append("\n### 风控统计\n")
            lines.append("| 因子 | 止损触发 | 止损占比 | 熔断触发 |")
            lines.append("|------|----------|----------|----------|")
            for factor_name, stats in risk_stats.items():
                lines.append(
                    f"| {factor_name} "
                    f"| {stats.get('stop_loss_triggered', 0)} "
                    f"| {stats.get('stop_loss_ratio', 0):.1%} "
                    f"| {stats.get('liquidate_triggered', 0)} |"
                )

        lines.append("\n## 三、分层回测概要\n")
        for factor_name, result in layer_results.items():
            if "cum_net_value" not in result:
                continue
            cum = result["cum_net_value"]
            last_row = cum.iloc[-1]
            lines.append(f"\n### {factor_name}\n")
            lines.append("| 分层 | 最终累计净值 |")
            lines.append("|------|-------------|")
            for layer in sorted(last_row.index):
                lines.append(f"| 第{layer}层 | {last_row[layer]:.3f} |")
            if "long_short" in result and not result["long_short"].empty:
                ls = result["long_short"]
                ls_final = ls["cum_long_short"].iloc[-1] if "cum_long_short" in ls.columns else 0
                lines.append(f"\n多空累计净值：**{ls_final:.3f}**")
                lines.append(f"多空日胜率：**{(ls['long_short_ret'] > 0).mean():.1%}**")
            if "ls_stats" in result:
                s = result["ls_stats"]
                lines.append(f"多空日均收益：**{s.get('ls_mean', 0):.5f}**")
                lines.append(f"多空日波动率：**{s.get('ls_vol', 0):.5f}**")
                lines.append(f"多空最大回撤：**{s.get('ls_max_dd', 0):.2%}**")
        lines.append("\n## 四、输出图表\n")
        png_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith(".png")])
        for f in png_files:
            lines.append(f"- `{f}`")
        report = "\n".join(lines)
        report_path = os.path.join(self.output_dir, "backtest_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[报告] 已生成 {report_path}")
        return report
