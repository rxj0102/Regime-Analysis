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
) -> tuple[pd.Series, pd.DataFrame, float | pd.Series]:
    """Classify each trading day as Low-Vol or High-Vol.

    Uses the full-sample median of the benchmark's rolling volatility as the
    threshold, producing a balanced (~50/50) split.

    Look-ahead note: the full-sample median is appropriate for retrospective
    attribution. A live system must use an expanding-window median to avoid
    introducing forward-looking bias.

    Args:
        returns:   DataFrame of daily log returns containing the benchmark column.
        benchmark: Ticker to use as the market benchmark (default: 'SPY').
        window:    Rolling window size in trading days (default: 30).
        method:    ``"full_sample"`` (default) splits at the median of the whole
                   series, suitable for retrospective attribution only.
                   ``"expanding"`` compares each day's rolling vol with the median
                   of rolling vol up to that day, so the label for day t uses
                   data through t only and could have been produced live.
        min_periods: Days of rolling vol required before the expanding median
                   is used (default 252); earlier days get no label.

    Returns:
        Tuple of:
        - regime:          pd.Series of 'Low-Vol'/'High-Vol' labels, aligned to
                           returns index (first ``window-1`` rows are NaN and excluded).
        - returns_aligned: returns restricted to days with a valid regime label.
        - threshold:       The daily-vol threshold used for classification.
    """
    rolling_vol = returns[benchmark].rolling(window).std()
    if method == "full_sample":
        threshold = float(rolling_vol.median())
        thr = pd.Series(threshold, index=rolling_vol.index)
    elif method == "expanding":
        thr = rolling_vol.expanding(min_periods=min_periods).median()
        threshold = thr
    else:
        raise ValueError(f"unknown method {method!r}")

    lab = np.where(rolling_vol > thr, "High-Vol", "Low-Vol").astype(object)
    # Warm-up days have no rolling vol (or no threshold yet): leave unlabelled.
    lab[(rolling_vol.isna() | thr.isna()).to_numpy()] = np.nan
    regime = pd.Series(lab, index=rolling_vol.index).reindex(returns.index)

    valid_idx = regime.dropna().index
    return regime.loc[valid_idx], returns.loc[valid_idx], threshold
