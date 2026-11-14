"""Data loading and distributional diagnostics."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from .config import END_DATE, START_DATE, TICKERS


def load_prices(
    tdef load_prices(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Total return indices for the open data stand ins (Ken French Data Library).

    TECH, FIN and STAPLES are the BusEq, Money and NoDur 12 industry portfolios
    (value weighted); SPY is the US market (Mkt-RF + RF). No Yahoo Finance.

    Returns:
        DataFrame of total return index levels; rows = dates, columns = tickers.
        Rows with any missing value are dropped.
    """
    from .opendata import french

    ind = french("12_Industry_Portfolios_daily", start=start, end=end)
    ff = french("F-F_Research_Data_Factors_daily", start=start, end=end)
    rets = pd.DataFrame({
        "TECH": ind["BusEq"], "FIN": ind["Money"], "STAPLES": ind["NoDur"],
        "SPY": ff["Mkt-RF"] + ff["RF"],
    })[list(tickers)].dropna(how="any")
    rets = rets[rets.index < pd.Timestamp(end)]
    return 100 * (1 + rets).cumprod()


: pd.DataFrame) -> pd.DataFrame:
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
