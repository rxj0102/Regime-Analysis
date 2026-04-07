"""Historical scenario stress testing for an equity portfolio."""

import numpy as np
import pandas as pd

from .capm import fit_capm
from .config import STRESS_EVENTS


def run_stress_tests(
    returns: pd.DataFrame,
    weights: dict[str, float] | None = None,
    portfolio_value: float = 1_000_000,
    events: dict[str, tuple[str, str, str]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run historical scenario stress tests on a long-only equity portfolio.

    For each stress event the function reports:
    - Per-asset cumulative return, max drawdown, and stress-period beta.
    - Portfolio-level cumulative return, dollar P&L, and daily 95% CVaR.

    Args:
        returns:          DataFrame of daily log returns.
        weights:          {ticker: weight} dict.  Defaults to equal weights.
        portfolio_value:  Notional portfolio size in USD (default: $1,000,000).
        events:           Stress scenario dict {name: (start, end, description)}.
                          Defaults to STRESS_EVENTS from config.

    Returns:
        Tuple of (asset_records, portfolio_summary):
        - asset_records:     List of per-asset dicts with keys
                             event, ticker, n_days, cum_ret_pct, max_dd_pct, stress_beta.
        - portfolio_summary: List of portfolio-level dicts with keys
                             name, p_cum, dollar_pnl, remaining, p_cvar.
    """
    if weights is None:
        w_uniform = 1.0 / len(returns.columns)
        weights = {t: w_uniform for t in returns.columns}
    if events is None:
        events = STRESS_EVENTS

    asset_records: list[dict] = []
    portfolio_summary: list[dict] = []

    for name, (start, end, _desc) in events.items():
        window = returns.loc[start:end]
        if window.empty:
            continue

        for t in returns.columns:
            r = window[t]
            cum_ret = np.exp(r.sum()) - 1
            cc = np.exp(r.cumsum())
            max_dd = float(
                ((cc - cc.expanding().max()) / cc.expanding().max()).min()
            )
            stress_beta = (
                fit_capm(r.values, window["SPY"].values)[1]
                if t != "SPY" else 1.0
            )
            asset_records.append(dict(
                event=name,
                ticker=t,
                n_days=len(r),
                cum_ret_pct=cum_ret * 100,
                max_dd_pct=max_dd * 100,
                stress_beta=stress_beta,
            ))

        tickers = list(returns.columns)
        w_arr = np.array([weights.get(t, 0.0) for t in tickers])
        port_ret = window[tickers] @ w_arr
        port_cum = float(np.exp(port_ret.sum()) - 1)
        port_cvar = float(port_ret[port_ret <= np.percentile(port_ret, 5)].mean())
        dollar_pnl = portfolio_value * port_cum

        portfolio_summary.append(dict(
            name=name,
            p_cum=port_cum,
            dollar_pnl=dollar_pnl,
            remaining=portfolio_value + dollar_pnl,
            p_cvar=port_cvar,
        ))

    return asset_records, portfolio_summary
