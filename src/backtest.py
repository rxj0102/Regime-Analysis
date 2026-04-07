"""Rolling historical VaR backtesting via Kupiec POF and Christoffersen IND."""

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

    for t in returns.columns:
        segments = [
            ("Full Sample", returns[t]),
            ("Low-Vol",     returns.loc[regime == "Low-Vol",  t]),
            ("High-Vol",    returns.loc[regime == "High-Vol", t]),
        ]
        for lbl, r in segments:
            if len(r) < min_obs:
                continue

            var_s = r.rolling(var_window, min_periods=min_periods).quantile(alpha)
            r_aligned, var_aligned = r.dropna().align(var_s.dropna(), join="inner")
            if len(r_aligned) < min_obs // 2:
                continue

            violations = (r_aligned < -var_aligned.abs()).astype(int)
            ct = christoffersen_test(violations, alpha=alpha)

            rows.append(dict(
                Ticker=t,
                Regime=lbl,
                **ct,
                pof_pass=ct["p_pof"] > alpha,
                ind_pass=ct["p_ind"] > alpha,
            ))

    return pd.DataFrame(rows)
