"""Regime detection via SPY rolling-volatility median split."""

import numpy as np
import pandas as pd

from .config import REGIME_WINDOW


def classify_regimes(
    returns: pd.DataFrame,
    benchmark: str = "SPY",
    window: int = REGIME_WINDOW,
    method: str = "full_sample",
    min_periods: int = 252,
    point_in_time: bool = False,
    history: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame, float | pd.Series]:
    """Classify each trading day as Low-Vol or High-Vol.

    Methods:

    * ``"full_sample"`` (default): threshold is the full-sample median of the
      benchmark's rolling vol. Balanced split, but it uses future data, so it
      suits retrospective attribution only.
    * ``"expanding"``: day t is labelled from information through t-1 only. The
      signal is the rolling vol at t-1 and the threshold is the expanding median
      of all rolling-vol readings up to t-1 (at least ``min_periods`` of them).
      ``history`` can supply a longer benchmark return series to warm the
      median up before the study window. This is what the notebook uses.

    ``point_in_time=True`` is kept as an alias for ``method="expanding"``.

    Args:
        returns:     DataFrame of daily log returns containing the benchmark column.
        benchmark:   Column to use as the market benchmark (default: 'SPY').
        window:      Rolling window size in trading days (default: 30).
        method:      ``"full_sample"`` or ``"expanding"``.
        min_periods: Rolling-vol readings required before the expanding median
                     is defined; earlier days get no label.
        point_in_time: Alias for ``method="expanding"``.
        history:     Optional longer benchmark return series (expanding method).

    Returns:
        Tuple of:
        - regime:          pd.Series of 'Low-Vol'/'High-Vol' labels for days with a valid label.
        - returns_aligned: returns restricted to those days.
        - threshold:       float for ``full_sample``; for ``expanding`` the
                           threshold series in force on each labelled day.
    """
    if point_in_time:
        method = "expanding"
    if method == "full_sample":
        rolling_vol = returns[benchmark].rolling(window).std()
        threshold = float(rolling_vol.median())
        regime = pd.Series(
            np.where(rolling_vol > threshold, "High-Vol", "Low-Vol"),
            index=rolling_vol.index,
        ).where(rolling_vol.notna()).reindex(returns.index)
        valid_idx = regime.dropna().index
        return regime.loc[valid_idx], returns.loc[valid_idx], threshold
    if method != "expanding":
        raise ValueError(f"unknown method {method!r}")

    base = history if history is not None else returns[benchmark]
    rolling_vol = base.rolling(window).std()
    signal = rolling_vol.shift(1)
    thr = rolling_vol.expanding(min_periods=min_periods).median().shift(1)
    ok = signal.notna() & thr.notna()
    regime = pd.Series(np.where(signal > thr, "High-Vol", "Low-Vol"), index=base.index).where(ok)
    regime = regime.reindex(returns.index)
    valid_idx = regime.dropna().index
    return regime.loc[valid_idx], returns.loc[valid_idx], thr.reindex(valid_idx)


