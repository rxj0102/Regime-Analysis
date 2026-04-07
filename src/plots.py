"""Visualisation utilities for regime-dependent risk analysis."""

from itertools import groupby

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist

from .config import COLORS, TICKERS


def shade_regimes(ax: plt.Axes, regime_series: pd.Series) -> None:
    """Shade contiguous regime blocks on an axis — one span per run, not per day.

    Args:
        ax:             Matplotlib axes to draw on.
        regime_series:  Series of 'Low-Vol'/'High-Vol' labels.
    """
    dates = regime_series.index
    for val, grp in groupby(enumerate(regime_series), key=lambda x: x[1]):
        if pd.isna(val):
            continue
        idxs = list(grp)
        ax.axvspan(dates[idxs[0][0]], dates[idxs[-1][0]],
                   alpha=0.25, color=COLORS[val])


def plot_qq(returns: pd.DataFrame, diag: pd.DataFrame) -> plt.Figure:
    """QQ-plots comparing empirical returns to Normal and fitted Student-t.

    Lower fitted degrees-of-freedom indicates heavier tails than Normal,
    confirming that Gaussian VaR will systematically underestimate tail risk.

    Args:
        returns: DataFrame of daily log returns.
        diag:    Diagnostics DataFrame with 'kurt' column per ticker.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("QQ-Plots: Normal vs Student-t — Fat Tails Confirmed",
                 fontsize=14, fontweight="bold")

    for ax, t in zip(axes.flatten(), returns.columns):
        r = returns[t].values
        (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
        ax.scatter(osm, osr, s=8, alpha=0.4, color="steelblue", label="Empirical")
        ax.plot(osm, slope * np.array(osm) + intercept,
                "k--", lw=1.5, label="Normal")

        df_fit, loc_fit, scale_fit = t_dist.fit(r)
        t_q = np.sort(t_dist.ppf(
            np.linspace(0.001, 0.999, len(r)),
            df=df_fit, loc=loc_fit, scale=scale_fit,
        ))
        ax.plot(t_q, np.sort(r), "r-", lw=1.2, alpha=0.8,
                label=f"Student-t (df={df_fit:.1f})")

        ax.set_title(f"{t} — excess kurtosis: {diag.loc[t, 'kurt']:.2f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_regimes(
    prices: pd.DataFrame,
    regime: pd.Series,
    spy_rvol: pd.Series,
    threshold: float,
) -> plt.Figure:
    """Two-panel plot: SPY price with regime shading and rolling-vol panel.

    Args:
        prices:    Adjusted close price DataFrame (must contain 'SPY').
        regime:    Regime label series aligned to prices.
        spy_rvol:  SPY 30-day rolling volatility series.
        threshold: Vol threshold separating Low-Vol from High-Vol.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(
        "Market Regimes: Low-Vol vs High-Vol (SPY 30-day σ vs median)",
        fontsize=14, fontweight="bold",
    )

    ax1.plot(prices.index, prices["SPY"], color="black", lw=1.8)
    shade_regimes(ax1, regime)
    ax1.set_ylabel("SPY Price ($)", fontsize=11)
    ax1.grid(True, alpha=0.25)

    for label, date in [
        ("COVID\n(Mar 2020)", "2020-03-06"),
        ("2022 Rate Shock", "2022-04-01"),
    ]:
        ax1.axvline(pd.Timestamp(date), color="navy", ls=":", lw=1.2, alpha=0.7)
        ylo, yhi = prices["SPY"].min(), prices["SPY"].max()
        ax1.text(pd.Timestamp(date), ylo + 0.6 * (yhi - ylo), label,
                 fontsize=8, color="navy", rotation=90, va="top")

    legend_handles = [
        mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0",
                                facecolor=COLORS["Low-Vol"], alpha=0.5, label="Low-Vol"),
        mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0",
                                facecolor=COLORS["High-Vol"], alpha=0.5, label="High-Vol"),
        plt.Line2D([0], [0], color="black", lw=1.8, label="SPY"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=10)

    ax2.plot(spy_rvol.index, spy_rvol, color="darkorange", lw=1.8)
    ax2.axhline(threshold, color="black", ls="--", lw=1.4,
                label=f"Median ({threshold:.4f})")
    ax2.fill_between(spy_rvol.index, spy_rvol, threshold,
                     where=(spy_rvol > threshold),  color=COLORS["High-Vol"], alpha=0.25)
    ax2.fill_between(spy_rvol.index, spy_rvol, threshold,
                     where=(spy_rvol <= threshold), color=COLORS["Low-Vol"],  alpha=0.25)
    ax2.set_ylabel("Rolling σ (daily)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


def plot_beta_scatter(
    returns: pd.DataFrame,
    regime: pd.Series,
    beta_df: pd.DataFrame,
    benchmark: str = "SPY",
) -> plt.Figure:
    """Scatter plots of asset vs benchmark returns, coloured by regime.

    Each panel includes fitted CAPM lines for full sample, Low-Vol, and
    High-Vol regimes with beta and 95% CI annotations.

    Args:
        returns:   DataFrame of aligned daily log returns.
        regime:    Regime label series.
        beta_df:   DataFrame from capm.compute_regime_betas.
        benchmark: Benchmark column name.

    Returns:
        Matplotlib Figure.
    """
    assets = [t for t in returns.columns if t != benchmark]
    fig, axes = plt.subplots(1, len(assets), figsize=(16, 5))
    fig.suptitle("Beta Instability: Regime-Conditional CAPM",
                 fontsize=14, fontweight="bold")

    x_grid = np.linspace(returns[benchmark].min(), returns[benchmark].max(), 100)

    for ax, t in zip(axes, assets):
        row = beta_df[beta_df.Ticker == t].iloc[0]

        for regime_lbl, marker in [("Low-Vol", "o"), ("High-Vol", "s")]:
            mask = regime == regime_lbl
            ax.scatter(
                returns.loc[mask, benchmark], returns.loc[mask, t],
                alpha=0.2, s=12, color=COLORS[regime_lbl], marker=marker,
                label=regime_lbl,
            )

        for lbl, key, lc, ls in [
            ("Full",     "Full",    "black",            "-"),
            ("Low-Vol",  "LowVol",  COLORS["Low-Vol"],  "--"),
            ("High-Vol", "HighVol", COLORS["High-Vol"], "--"),
        ]:
            b, a = row[f"{key}_beta"], row[f"{key}_alpha"]
            if not np.isnan(b):
                ax.plot(x_grid, a + b * x_grid,
                        color=lc, ls=ls, lw=2, label=f"{lbl} β={b:.2f}")

        ax.set_title(f"{t}: Δβ={row.dBeta:+.3f} ({row.dBeta_pct:+.1f}%)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{benchmark} log return")
        ax.set_ylabel(f"{t} log return")
        ax.text(
            0.03, 0.97,
            (f"High-Vol β: {row.HighVol_beta:.3f}\n"
             f"95%CI [{row.HighVol_blo:.3f},{row.HighVol_bhi:.3f}]\n"
             f"R²={row.HighVol_r2:.3f}"),
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.75),
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


def plot_volatility(vol_models: dict, lookback: int = 252) -> plt.Figure:
    """EWMA vs GARCH conditional volatility for the last N trading days.

    Each panel also shows the 1-step-ahead forecast as a scatter point and
    the GARCH persistence parameter (alpha + beta).

    Args:
        vol_models: Dict from volatility.fit_all_vol_models.
        lookback:   Number of recent trading days to display.

    Returns:
        Matplotlib Figure.
    """
    tickers = list(vol_models.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Volatility Forecasting: EWMA vs GARCH(1,1) — Last 252 Trading Days",
        fontsize=14, fontweight="bold",
    )

    for ax, t in zip(axes.flatten(), tickers):
        m = vol_models[t]
        recent = m["r"].index[-lookback:]

        ax.fill_between(recent, 0, m["r"].loc[recent].abs(),
                        alpha=0.15, color="gray", label="|returns|")
        ax.plot(recent, m["ewma_vol"].loc[recent],
                color="steelblue", lw=2, label="EWMA")

        if m["garch_vol"] is not None:
            ax.plot(recent, m["garch_vol"].loc[recent],
                    color="crimson", lw=2, label=f"GARCH ({m['garch_dist']})")

        fdate = recent[-1] + pd.offsets.BDay(1)
        ax.scatter([fdate], [m["ewma_fcast"]], color="steelblue", s=90, zorder=5)
        if m["garch_fcast"]:
            ax.scatter([fdate], [m["garch_fcast"]],
                       color="crimson", marker="s", s=90, zorder=5)

        if m["garch_fit"] is not None:
            p = m["garch_fit"].params
            ax.text(0.02, 0.98,
                    f"α+β = {p['alpha[1]'] + p['beta[1]']:.4f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.set_ylabel("Daily σ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


def plot_dashboard(
    metrics_df: pd.DataFrame,
    regime: pd.Series,
    beta_df: pd.DataFrame,
    vol_models: dict,
    stress_summary: list[dict],
    tickers: list[str] = TICKERS,
    save_path: str | None = None,
) -> plt.Figure:
    """3×3 summary dashboard combining all analysis results.

    Panels:
    1. Regime distribution (pie).
    2. Sharpe ratio by regime (grouped bar).
    3. CAPM beta by regime (grouped bar).
    4. 95% CVaR by regime (grouped bar).
    5. Max drawdown by regime (grouped bar).
    6. Next-day vol forecast: EWMA vs GARCH (grouped bar).
    7. Stress test portfolio returns (bar).
    8. Key findings text box.

    Args:
        metrics_df:     Per-regime risk metrics DataFrame.
        regime:         Regime label series.
        beta_df:        CAPM beta DataFrame.
        vol_models:     Volatility model results dict.
        stress_summary: Portfolio stress test result dicts.
        tickers:        Full ticker list including benchmark.
        save_path:      If provided, save figure to this path at 150 dpi.

    Returns:
        Matplotlib Figure.
    """
    nb_tickers = [t for t in tickers if t != "SPY"]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Regime-Dependent Risk & Performance Attribution — Summary Dashboard",
        fontsize=16, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    W = 0.30
    x4 = np.arange(len(tickers))
    x3 = np.arange(len(nb_tickers))

    # 1. Regime distribution
    ax1 = fig.add_subplot(gs[0, 0])
    vc = regime.value_counts()
    ax1.pie(vc.values, labels=vc.index, colors=[COLORS[r] for r in vc.index],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax1.set_title("Regime distribution", fontsize=12, fontweight="bold")

    # 2. Sharpe by regime
    ax2 = fig.add_subplot(gs[0, 1])
    for i, lbl in enumerate(["Low-Vol", "High-Vol"]):
        sharpes = [
            metrics_df.loc[(metrics_df.Ticker == t) & (metrics_df.Label == lbl), "Sharpe"].values[0]
            for t in tickers
        ]
        ax2.bar(x4 + (i - 0.5) * W, sharpes, W,
                label=lbl, color=COLORS[lbl], alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_xticks(x4); ax2.set_xticklabels(tickers)
    ax2.set_title("Sharpe by regime (RF=2.5%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Sharpe"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)

    # 3. Beta instability
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (lbl, key) in enumerate([("Low-Vol", "LowVol"), ("High-Vol", "HighVol")]):
        betas = [
            beta_df.loc[beta_df.Ticker == t, f"{key}_beta"].values[0]
            for t in nb_tickers
        ]
        ax3.bar(x3 + (i - 0.5) * W, betas, W,
                label=lbl, color=COLORS[lbl], alpha=0.8)
    ax3.set_xticks(x3); ax3.set_xticklabels(nb_tickers)
    ax3.set_title("CAPM β by regime", fontsize=12, fontweight="bold")
    ax3.set_ylabel("β"); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25)

    # 4. CVaR by regime
    ax4 = fig.add_subplot(gs[1, 0])
    for i, lbl in enumerate(["Low-Vol", "High-Vol"]):
        cvars = [
            abs(metrics_df.loc[(metrics_df.Ticker == t) & (metrics_df.Label == lbl), "CVaR95"].values[0])
            for t in tickers
        ]
        bars = ax4.bar(x4 + (i - 0.5) * W, cvars, W,
                       label=lbl, color=COLORS[lbl], alpha=0.8)
        for bar, v in zip(bars, cvars):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax4.set_xticks(x4); ax4.set_xticklabels(tickers)
    ax4.set_title("95% CVaR by regime |%|", fontsize=12, fontweight="bold")
    ax4.set_ylabel("CVaR 95% |%|"); ax4.legend(fontsize=9); ax4.grid(True, alpha=0.25)

    # 5. Max drawdown by regime
    ax5 = fig.add_subplot(gs[1, 1])
    for i, lbl in enumerate(["Low-Vol", "High-Vol"]):
        mdd = [
            abs(metrics_df.loc[(metrics_df.Ticker == t) & (metrics_df.Label == lbl), "MaxDD"].values[0])
            for t in tickers
        ]
        ax5.bar(x4 + (i - 0.5) * W, mdd, W,
                label=lbl, color=COLORS[lbl], alpha=0.8)
    ax5.set_xticks(x4); ax5.set_xticklabels(tickers)
    ax5.set_title("Max Drawdown by regime (%)", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Max DD |%|"); ax5.legend(fontsize=9); ax5.grid(True, alpha=0.25)

    # 6. Vol forecasts
    ax6 = fig.add_subplot(gs[1, 2])
    ewma_f  = [vol_models[t]["ewma_fcast"] * np.sqrt(252) * 100 for t in tickers]
    garch_f = [(vol_models[t]["garch_fcast"] or 0) * np.sqrt(252) * 100 for t in tickers]
    ax6.bar(x4 - W / 2, ewma_f,  W, label="EWMA",         color="steelblue", alpha=0.8)
    ax6.bar(x4 + W / 2, garch_f, W, label="GARCH (best)", color="crimson",   alpha=0.8)
    ax6.set_xticks(x4); ax6.set_xticklabels(tickers)
    ax6.set_title("Next-day vol forecast (ann. %)", fontsize=12, fontweight="bold")
    ax6.set_ylabel("Ann. σ forecast (%)"); ax6.legend(fontsize=9); ax6.grid(True, alpha=0.25)

    # 7. Stress test returns
    ax7 = fig.add_subplot(gs[2, :2])
    if stress_summary:
        sc_labels = [s["name"] for s in stress_summary]
        sc_rets   = [s["p_cum"] * 100 for s in stress_summary]
        bars = ax7.bar(sc_labels, sc_rets, color=["#c0392b", "#e67e22"], alpha=0.8, width=0.4)
        for bar, ret, s in zip(bars, sc_rets, stress_summary):
            ax7.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.5 if ret < 0 else bar.get_height() + 0.3,
                f"{ret:.1f}%\n(${s['dollar_pnl']:+,.0f})",
                ha="center", va="top" if ret < 0 else "bottom",
                fontsize=9, fontweight="bold", color="white" if ret < 0 else "black",
            )
    ax7.axhline(0, color="black", lw=1)
    ax7.set_title("Stress test: equal-weight portfolio return",
                  fontsize=12, fontweight="bold")
    ax7.set_ylabel("Return (%)"); ax7.grid(True, alpha=0.25)

    # 8. Key findings text box
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    ax8.text(
        0.05, 0.98,
        ("Key Findings\n" + "─" * 38 + "\n"
         "1. Full-sample Sharpe masks negative\n"
         "   risk-adj returns in High-Vol.\n\n"
         "2. CAPM β is non-stationary: assets\n"
         "   become more correlated in crises.\n\n"
         "3. Full-sample VaR fails Christoffersen\n"
         "   IND test (violation clustering).\n\n"
         "4. GARCH(t/skewt) outperforms EWMA\n"
         "   by AIC across all assets.\n\n"
         "5. Equal-weight portfolio lost material\n"
         "   capital in both stress events."),
        transform=ax8.transAxes, fontsize=10, va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="#f0f4ff", alpha=0.85),
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
