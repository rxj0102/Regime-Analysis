"""Regime-conditional CAPM fitting with HC3 robust standard errors."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_capm(
    y: np.ndarray,
    x: np.ndarray,
    min_obs: int = 30,
) -> tuple[float, float, float, float, float, float]:
    """Fit OLS CAPM with HC3 heteroskedasticity-robust standard errors.

    HC3 is mandatory here: fat-tailed, regime-switching returns guarantee
    non-constant error variance, making plain OLS standard errors badly
    understated.

    Args:
        y:       Asset log return array.
        x:       Market (benchmark) log return array, same length as y.
        min_obs: Minimum valid (non-NaN) observations required to fit.

    Returns:
        Tuple: (alpha_daily, beta, beta_ci_lo, beta_ci_hi, r_squared, p_beta).
        Returns a 6-tuple of NaN if there are fewer than min_obs valid rows.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < min_obs:
        return (np.nan,) * 6

    res = sm.OLS(y[valid], sm.add_constant(x[valid])).fit(cov_type="HC3")
    ci = res.conf_int()
    return (
        float(res.params[0]),
        float(res.params[1]),
        float(ci[1, 0]),
        float(ci[1, 1]),
        float(res.rsquared),
        float(res.pvalues[1]),
    )


def compute_regime_betas(
    returns: pd.DataFrame,
    regime: pd.Series,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Compute full-sample and per-regime CAPM betas for all non-benchmark assets.

    Args:
        returns:   DataFrame of aligned daily log returns.
        regime:    Series of 'Low-Vol'/'High-Vol' labels, same index as returns.
        benchmark: Column name for the market benchmark.

    Returns:
        DataFrame with one row per asset and columns:
        {Full,LowVol,HighVol}_{alpha, beta, blo, bhi, r2, pb}, dBeta, dBeta_pct.
    """
    assets = [t for t in returns.columns if t != benchmark]
    records = []

    for t in assets:
        rec: dict = {"Ticker": t}
        segments = [
            ("Full",    pd.Series(True, index=returns.index)),
            ("LowVol",  regime == "Low-Vol"),
            ("HighVol", regime == "High-Vol"),
        ]
        for lbl, mask in segments:
            a, b, blo, bhi, r2, pb = fit_capm(
                returns.loc[mask, t].values,
                returns.loc[mask, benchmark].values,
            )
            rec.update({
                f"{lbl}_alpha": a,  f"{lbl}_beta": b,
                f"{lbl}_blo":   blo, f"{lbl}_bhi": bhi,
                f"{lbl}_r2":    r2,  f"{lbl}_pb":  pb,
            })

        lv, hv = rec["LowVol_beta"], rec["HighVol_beta"]
        rec["dBeta"] = (hv - lv) if not (np.isnan(lv) or np.isnan(hv)) else np.nan
        rec["dBeta_pct"] = (
            100 * (hv - lv) / abs(lv)
            if (not np.isnan(lv) and lv != 0) else np.nan
        )
        records.append(rec)

    return pd.DataFrame(records)
