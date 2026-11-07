"""Risk/performance metrics and VaR backtesting statistics."""

import math

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import RF_ANNUAL, RF_DAILY


def calc_metrics(returns: pd.Series, label: str) -> dict:
    """Compute annualised risk/return metrics for a daily log-return series.

    Metrics:
    - Annualised return and volatility (252 trading days).
    - Sharpe ratio against the configured risk-free rate.
    - Sortino ratio using semi-deviation (downside risk only).
    - Maximum drawdown and Calmar ratio.
    - Historical 95% VaR and CVaR (Expected Shortfall).

    Args:
        returns: Daily log return series.
        label:   Identifier stored in the returned dict (e.g. 'Low-Vol').

    Returns:
        Dict with keys: Label, N, AnnRet, AnnVol, Sharpe, Sortino,
        MaxDD, Calmar, VaR95, CVaR95.  Returns empty dict if len < 2.
    """
    n = len(returns)
    if n < 2:
        return {}

    cum = np.exp(returns.sum())
    ann_ret = cum ** (252 / n) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan

    # Sortino: semi-deviation = sqrt(E[min(r - MAR, 0)^2]) * sqrt(252).
    # This is stricter than std(negative returns), which understates downside.
    dd_dev = np.sqrt(np.mean(np.minimum(returns - RF_DAILY, 0) ** 2)) * np.sqrt(252)
    sortino = (ann_ret - RF_ANNUAL) / dd_dev if dd_dev > 0 else np.nan

    cum_curve = np.exp(returns.cumsum())
    max_dd = (
        (cum_curve - cum_curve.expanding().max()) / cum_curve.expanding().max()
    ).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    var95 = float(np.percentile(returns, 5))
    cvar95 = float(returns[returns <= var95].mean())

    return dict(
        Label=label, N=n,
        AnnRet=ann_ret * 100, AnnVol=ann_vol * 100,
        Sharpe=sharpe, Sortino=sortino,
        MaxDD=max_dd * 100, Calmar=calmar,
        VaR95=var95 * 100, CVaR95=cvar95 * 100,
    )


def christoffersen_test(violations: pd.Series, alpha: float = 0.05, adjacent=None) -> dict:
    """Christoffersen (1998) conditional coverage test for VaR backtesting.

    Tests two null hypotheses via likelihood-ratio statistics:

    - POF (Kupiec 1995): the unconditional violation rate equals ``alpha``.
    - IND: violations are serially independent (no clustering).

    Both tests must pass (p > 0.05) for a VaR model to be considered
    well-calibrated.

    Args:
        violations: Binary integer series (1 = VaR breach, 0 = no breach).
        alpha:      Target violation rate, e.g. 0.05 for 95% VaR.
        adjacent:   Optional boolean array of length ``len(violations) - 1``;
                    entry i is True when observations i and i+1 are
                    consecutive trading days.  Only such pairs enter the
                    independence transition counts (needed when the series
                    is a regime subset with gaps).  Default: all pairs.

    Returns:
        Dict with keys: T, N, pi_hat, lr_pof, p_pof, lr_ind, p_ind.
    """
    v = np.asarray(violations, dtype=int)
    T = len(v)
    N = int(v.sum())
    pi_hat = N / T if T > 0 else 0.0

    # --- Kupiec POF ---
    if pi_hat in (0.0, 1.0):
        lr_pof, p_pof = np.inf, 0.0
    else:
        lr_pof = -2.0 * (
            N * math.log(alpha / pi_hat)
            + (T - N) * math.log((1 - alpha) / (1 - pi_hat))
        )
        p_pof = float(1 - chi2.cdf(lr_pof, df=1))

    # --- Christoffersen Independence ---
    adj = np.ones(max(T - 1, 0), dtype=bool) if adjacent is None else np.asarray(adjacent, dtype=bool)
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0) & adj))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1) & adj))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0) & adj))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1) & adj))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    def _slog(x: float) -> float:
        return math.log(x) if x > 1e-12 else 0.0

    ll_free = (
        n00 * _slog(1 - pi01) + n01 * _slog(pi01)
        + n10 * _slog(1 - pi11) + n11 * _slog(pi11)
    )
    ll_restr = (
        (n00 + n10) * _slog(1 - pi_hat)
        + (n01 + n11) * _slog(pi_hat)
    )
    lr_ind = max(-2.0 * (ll_restr - ll_free), 0.0)
    p_ind = float(1 - chi2.cdf(lr_ind, df=1))

    return dict(
        T=T, N=N, pi_hat=pi_hat,
        lr_pof=lr_pof, p_pof=p_pof,
        lr_ind=lr_ind, p_ind=p_ind,
    )
