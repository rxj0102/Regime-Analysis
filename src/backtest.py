"""Rolling historical VaR backtesting via Kupiec POF and Christoffersen IND."""

import numpy as np
import pandas as pd

from .metrics import christoffersen_test


def backtest_var(
    returns: pd.DataFrame,
    regime: pd.Series,
    var_window: int = 252,
    min_periods: int = 60,
    alpha: float = 0.05,
    min_obs: int = 120,
) -> pd.DataFrame:
    """Backtest rolling historical VaR across all tickers and regime slices.

    For each (ticker, regime) pair, a rolling empirical quantile VaR is
    computed, violations are identified, and then tested for:
    - Correct unconditional coverage (Kupiec POF).
    - Serial independence of violations (Christoffersen IND).

    Args:
        returns:     DataFrame of aligned daily log returns.
        regime:      Series of 'Low-Vol'/'High-Vol' labels, same index as returns.
        var_window:  Rolling lookback window for VaR estimation (days).
        min_periods: Minimum observations to start computing rolling VaR.
        alpha:       VaR confidence tail, e.g. 0.05 for 95% VaR.
        min_obs:     Minimum backtesting observations required per segment.

    Returns:
        DataFrame with one row per (ticker, regime), columns:
        Ticker, Regime, T, N, pi_hat, lr_pof, p_pof, lr_ind, p_ind,
        pof_pass, ind_pass.
    """
    rows = []
    pos = pd.Series(np.arange(len(returns)), index=returns.index)

    for t in returns.columns:
        # VaR is computed on the full, contiguous return series and lagged one
        # day, so the VaR applied to day t uses only returns up to t-1.
        var_full = (
            returns[t].rolling(var_window, min_periods=min_periods).quantile(alpha).shift(1)
        )
        viol_full = (returns[t] < var_full).astype(float).where(var_full.notna())

        segments = [
            ("Full Sample", pd.Series(True, index=returns.index)),
            ("Low-Vol", regime == "Low-Vol"),
            ("High-Vol", regime == "High-Vol"),
        ]
        for lbl, mask in segments:
            v = viol_full[mask.reindex(returns.index, fill_value=False)].dropna()
            if mask.sum() < min_obs or len(v) < min_obs // 2:
                continue
            p_ = pos.loc[v.index].to_numpy()
            adjacent = np.diff(p_) == 1
            ct = christoffersen_test(v.astype(int), alpha=alpha, adjacent=adjacent)

            rows.append(dict(
                Ticker=t,
                Regime=lbl,
                **ct,
                pof_pass=ct["p_pof"] > alpha,
                ind_pass=ct["p_ind"] > alpha,
            ))

    return pd.DataFrame(rows)
