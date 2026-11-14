"""Regime detection via SPY rolling-volatility median split."""

import numpy as np
import pandas as pd

from .config import REGIME_WINDOW


def classify_regimes(
    returns: pd.DataFrame,
    benchmark: str = "SPY",
    window: int = REGIME_WINDOW,
    point_in_time: bool = False,
    history: pd.Series | None = None,
    min_history: int = 252,
) -> tuple[pd.Series, pd.DataFrame, float]:
    """Classify each trading day as Low-Vol or High-Vol.

    Two modes:

    * ``point_in_time=False`` (retrospective): threshold is the full-sample
      median of the benchmark's rolling vol. Balanced split, but it uses
      future data.
    * ``point_in_time=True``: day t is labelled from information through t-1
      only. The signal is the rolling vol at t-1 and the threshold is the
      expanding median of all rolling-vol readings up to t-1 (at least
      ``min_history`` of them). ``history`` can supply a longer benchmark
      return series to warm the median up before the study window.

    Args:
        returns:   DataFrame of daily log returns containing the benchmark column.
        benchmark: Ticker to use as the market benchmark (default: 'SPY').
        window:    Rolling window size in trading days (default: 30).
        point_in_time: Use the look-ahead-free expanding threshold.
        history:   Optional longer benchmark return series (point-in-time mode).
        min_history: Minimum rolling-vol readings before a threshold is defined.

    Returns:
        Tuple of:
        - regime:          pd.Series of 'Low-Vol'/'High-Vol' labels for days with a valid label.
        - returns_aligned: returns restricted to those days.
        - threshold:       The threshold in force on the last labelled day.
    """
    if not point_in_time:
        rolling_vol = returns[benchmark].rolling(window).std()
        threshold = float(rolling_vol.median())
        regime = pd.Series(
            np.where(rolling_vol > threshold, "High-Vol", "Low-Vol"),
            index=rolling_vol.index,
        ).where(rolling_vol.notna()).reindex(returns.index)
        valid_idx = regime.dropna().index
        return regime.loc[valid_idx], returns.loc[valid_idx], threshold

    base = history if history is not None else returns[benchmark]
    rolling_vol = base.rolling(window).std()
    signal = rolling_vol.shift(1)
    thr = rolling_vol.expanding(min_periods=min_history).median().shift(1)
    ok = signal.notna() & thr.notna()
    regime = pd.Series(np.where(signal > thr, "High-Vol", "Low-Vol"), index=base.index).where(ok)
    regime = regime.reindex(returns.index)
    valid_idx = regime.dropna().index
    return regime.loc[valid_idx], returns.loc[valid_idx], float(thr.reindex(valid_idx).iloc[-1])
