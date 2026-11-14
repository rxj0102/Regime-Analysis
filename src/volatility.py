"""Volatility modelling: EWMA (RiskMetrics) and GARCH(1,1)."""

import numpy as np
import pandas as pd
from arch import arch_model

from .config import EWMA_LAMBDA, EWMA_WARMUP, GARCH_DISTS, TICKERS


def fit_ewma(
    returns: pd.Series,
    lam: float = EWMA_LAMBDA,
    warmup: int = EWMA_WARMUP,
) -> tuple[pd.Series, float]:
    """Fit an EWMA volatility model (RiskMetrics approach).

    The burn-in period is replaced by a rolling variance estimate to reduce
    initialisation noise introduced by exponential-smoothing start-up.

    Args:
        returns: Daily log return series.
        lam:     Exponential decay factor (default: 0.94, RiskMetrics standard).
        warmup:  Number of initial periods replaced by rolling variance.

    Returns:
        Tuple of (ewma_vol series, 1-step-ahead variance forecast as daily vol).
    """
    ewma_var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    rolling_init = (
        returns.pow(2).rolling(warmup).mean().iloc[:warmup]
        .fillna(returns.pow(2).mean())
    )
    ewma_var.iloc[:warmup] = rolling_init

    ewma_vol = np.sqrt(ewma_var)
    # With adjust=False the last value already is lam * v_{T-1} + (1 - lam) * r_T^2,
    # i.e. the RiskMetrics forecast for T+1. Applying the update again would
    # count r_T twice.
    fcast = float(np.sqrt(ewma_var.iloc[-1]))
    return ewma_vol, fcast


def fit_garch(
    returns: pd.Series,
    dists: list[str] = GARCH_DISTS,
) -> dict:
    """Fit GARCH(1,1), selecting the best error distribution by AIC.

    Returns are scaled to percent internally for numerical stability and
    converted back before storage.

    Args:
        returns: Daily log return series.
        dists:   Candidate error distributions ('normal', 't', 'skewt').

    Returns:
        Dict with keys:
        - fit:         ARCHModelResult (or None).
        - dist:        Winning distribution string.
        - aic:         Best AIC value.
        - vol:         pd.Series of conditional daily volatility.
        - fcast:       1-step-ahead daily vol forecast.
        - persistence: alpha + beta (IGARCH warning if >= 1).
    """
    best_aic, best_fit, best_dist = np.inf, None, None

    for dist in dists:
        try:
            fit = arch_model(
                returns * 100, vol="Garch", p=1, q=1, dist=dist
            ).fit(disp="off", show_warning=False)
            if fit.aic < best_aic:
                best_aic, best_fit, best_dist = fit.aic, fit, dist
        except Exception:
            continue

    if best_fit is None:
        return dict(fit=None, dist=None, aic=np.inf,
                    vol=None, fcast=None, persistence=np.nan)

    garch_vol = best_fit.conditional_volatility / 100
    garch_fcast = (
        float(np.sqrt(best_fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100
    )
    params = best_fit.params
    persistence = float(params.get("alpha[1]", 0.0) + params.get("beta[1]", 0.0))

    return dict(
        fit=best_fit, dist=best_dist, aic=best_aic,
        vol=pd.Series(garch_vol.values, index=returns.index),
        fcast=garch_fcast,
        persistence=persistence,
    )


def fit_all_vol_models(returns: pd.DataFrame) -> dict:
    """Fit EWMA and GARCH models for every ticker in the returns DataFrame.

    Args:
        returns: DataFrame of daily log returns.

    Returns:
        Nested dict keyed by ticker with sub-keys:
        r, ewma_vol, ewma_fcast, garch_vol, garch_fcast, garch_dist, garch_fit.
    """
    vol_models: dict = {}
    for t in returns.columns:
        r = returns[t].dropna()
        ewma_vol, ewma_fcast = fit_ewma(r)
        garch = fit_garch(r)

        vol_models[t] = dict(
            r=r,
            ewma_vol=ewma_vol,
            ewma_fcast=ewma_fcast,
            garch_vol=garch["vol"],
            garch_fcast=garch["fcast"],
            garch_dist=garch["dist"],
            garch_fit=garch["fit"],
        )
    return vol_models
