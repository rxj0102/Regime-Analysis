"""Data loading and distributional diagnostics."""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from .config import END_DATE, START_DATE, TICKERS


def load_prices(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Download split/dividend-adjusted closing prices from Yahoo Finance.

    auto_adjust=True ensures 'Close' is corrected for splits and dividends.
    Without it, NVDA's 4-for-1 split (Jul 2021) creates artificial return spikes.

    Args:
        tickers: Ticker symbols to download.
        start:   Start date string (YYYY-MM-DD).
        end:     End date string (YYYY-MM-DD).

    Returns:
        DataFrame of adjusted close prices; rows = dates, columns = tickers.
        Rows with any missing value are dropped.
    """
    import os
    from pathlib import Path

    # Yahoo serves slightly different adjusted history at different times, so the
    # download behind the README results is frozen in data/snapshot/. Set
    # REGIME_REFRESH_DATA=1 to download again.
    snap = Path(__file__).resolve().parents[1] / "data" / "snapshot" / f"close_{start}_{end}.csv"
    if snap.exists() and os.environ.get("REGIME_REFRESH_DATA") != "1":
        cached = pd.read_csv(snap, index_col=0, parse_dates=True)
        if set(tickers) <= set(cached.columns):
            return cached[list(tickers)].dropna(how="any")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    prices = pd.DataFrame({t: raw[(t, "Close")] for t in tickers}).dropna(how="any")
    if prices.empty:
        raise RuntimeError("yfinance returned no prices")
    snap.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(snap, float_format="%.6f", index_label="Date")
    return pd.read_csv(snap, index_col=0, parse_dates=True)[list(tickers)]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from an adjusted-price DataFrame.

    Args:
        prices: DataFrame of adjusted close prices.

    Returns:
        DataFrame of log returns with the first (NaN) row dropped.
    """
    return np.log(prices / prices.shift(1)).dropna()


def return_diagnostics(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute distributional diagnostics for each return series.

    Tests performed:
    - Excess kurtosis and skewness.
    - Jarque-Bera test for normality.
    - Augmented Dickey-Fuller test for stationarity.

    Args:
        returns: DataFrame of daily log returns.

    Returns:
        DataFrame indexed by ticker with columns: kurt, skew, jb_p, adf_p.
    """
    records: dict[str, dict] = {}
    for t in returns.columns:
        r = returns[t]
        _, jb_p = stats.jarque_bera(r)
        adf_p = adfuller(r, autolag="AIC")[1]
        records[t] = {
            "kurt":  float(stats.kurtosis(r)),
            "skew":  float(stats.skew(r)),
            "jb_p":  float(jb_p),
            "adf_p": float(adf_p),
        }
    return pd.DataFrame(records).T
