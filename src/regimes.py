"""Regime detection via SPY rolling-volatility median split."""

import numpy as np
import pandas as pd

from .config import REGIME_WINDOW


def classify_regimes(
    returns: pd.DataFrame,
    benchmark: str = "SPY",
    window: int = REGIME_WINDOW,
) -> tuple[pd.Series, pd.DataFrame, float]:
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

    Returns:
        Tuple of:
        - regime:          pd.Series of 'Low-Vol'/'High-Vol' labels, aligned to
                           returns index (first ``window-1`` rows are NaN and excluded).
        - returns_aligned: returns restricted to days with a valid regime label.
        - threshold:       The daily-vol threshold used for classification.
    """
    rolling_vol = returns[benchmark].rolling(window).std()
    threshold = float(rolling_vol.median())

    regime = pd.Series(
        np.where(rolling_vol > threshold, "High-Vol", "Low-Vol"),
        index=rolling_vol.index,
    ).reindex(returns.index)

    valid_idx = regime.dropna().index
    return regime.loc[valid_idx], returns.loc[valid_idx], threshold
