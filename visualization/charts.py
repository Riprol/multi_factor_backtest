import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import OUTPUT_DIR, plt_style, FONT_FAMILY, FIG_DPI
from utils.helpers import ensure_dir


plt.rcParams["font.sans-serif"] = [FONT_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
try:
    plt.style.use(plt_style)
except Exception:
    plt.style.use("ggplot")


class ChartDrawer:

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        ensure_dir(output_dir)

    def plot_layer_net_value(self, cum_net: pd.DataFrame, factor_label: str,
                              save: bool = True):
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(cum_net.columns)))
        for i, col in enumerate(cum_net.columns):
            ax.plot(cum_net.index, cum_net[col],
                    color=colors[i], lw=1.2, label=f"第{col}层")
        ax.axhline(y=1, color="gray", ls="--", alpha=0.5)
        ax.set_title(f"{factor_label} —— {len(cum_net.columns)}分层累计净值", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期")
        ax.set_ylabel("累计净值")
        ax.legend(loc="upper left", ncol=2, fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save:
            path = os.path.join(self.output_dir, f"layer_net_{factor_label}.png")
            fig.savefig(path, dpi=FIG_DPI)
            print(f"  [图表] 已保存 {path}")
        plt.show()
        return fig

    def plot_ic_series(self, ic_df: pd.DataFrame, factor_label: str,
                        save: bool = True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        ic_df_plot = ic_df.copy()
        ic_df_plot["trade_date"] = pd.to_datetime(ic_df_plot["trade_date"])
        ax1.plot(ic_df_plot["trade_date"], ic_df_plot["rank_ic"],
                 color="#2E86AB", lw=1.0, label="Rank IC")
        ax1.axhline(y=0, color="gray", ls="--", alpha=0.5)
        ax1.set_title(f"{factor_label} Rank IC 时序", fontsize=12, fontweight="bold")
        ax1.set_ylabel("IC")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax2.hist(ic_df_plot["rank_ic"].dropna(), bins=30,
                 color="#A23B72", alpha=0.7, edgecolor="white")
        ax2.axvline(x=0, color="gray", ls="--")
        ax2.axvline(x=ic_df_plot["rank_ic"].mean(), color="red", ls="-",
                    label=f"均值={ic_df_plot['rank_ic'].mean():.4f}")
        ax2.set_title(f"{factor_label} IC 分布", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Rank IC")
        ax2.set_ylabel("频数")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        if save:
            path = os.path.join(self.output_dir, f"ic_{factor_label}.png")
            fig.savefig(path, dpi=FIG_DPI)
            print(f"  [图表] 已保存 {path}")
        plt.show()
        return fig

    def plot_ic_comparison(self, all_ic_df: pd.DataFrame, save: bool = True):
        fig, ax = plt.subplots(figsize=(12, 6))
        factor_names = all_ic_df["factor_name"].unique()
        ic_data = [all_ic_df[all_ic_df["factor_name"] == n]["rank_ic"].dropna().values
                   for n in factor_names]
        bp = ax.boxplot(ic_data, labels=factor_names, patch_artist=True,
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="red"))
        for patch in bp["boxes"]:
            patch.set_facecolor("#A8DADC")
        ax.axhline(y=0, color="gray", ls="--")
        ax.set_title("多因子 Rank IC 分布对比", fontsize=14, fontweight="bold")
        ax.set_ylabel("Rank IC")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "ic_comparison.png")
            fig.savefig(path, dpi=FIG_DPI)
            print(f"  [图表] 已保存 {path}")
        plt.show()
        return fig

    def plot_portfolio_nav(self, portfolio_df: pd.DataFrame, factor_label: str,
                            benchmark_ret: pd.Series = None, save: bool = True):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(portfolio_df["trade_date"], portfolio_df["cum_nav"],
                 color="#E63946", lw=1.5, label=f"{factor_label} 组合")
        if benchmark_ret is not None and not benchmark_ret.empty:
            bench_nav = (1 + benchmark_ret).cumprod()
            common_idx = portfolio_df["trade_date"][:len(bench_nav)]
            ax1.plot(common_idx, bench_nav.values[:len(common_idx)],
                     color="gray", lw=1.0, ls="--", label="等权基准")
        ax1.axhline(y=1, color="gray", ls=":", alpha=0.5)
        ax1.set_title(f"{factor_label} 策略净值曲线", fontsize=14, fontweight="bold")
        ax1.set_ylabel("累计净值")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)
        cum_max = portfolio_df["cum_nav"].cummax()
        dd = (portfolio_df["cum_nav"] - cum_max) / cum_max
        ax2.fill_between(portfolio_df["trade_date"], 0, dd, color="#E63946", alpha=0.3)
        ax2.plot(portfolio_df["trade_date"], dd, color="#E63946", lw=0.8)
        ax2.set_title("回撤曲线", fontsize=12)
        ax2.set_ylabel("回撤")
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        if save:
            path = os.path.join(self.output_dir, f"nav_{factor_label}.png")
            fig.savefig(path, dpi=FIG_DPI)
            print(f"  [图表] 已保存 {path}")
        plt.show()
        return fig

    def plot_factor_correlation(self, factor_names: list,
                                 daily_factor_df: pd.DataFrame,
                                 save: bool = True):
        corr = daily_factor_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
        ax.set_title("因子截面相关性", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "factor_corr.png")
            fig.savefig(path, dpi=FIG_DPI)
            print(f"  [图表] 已保存 {path}")
        plt.show()
        return fig
