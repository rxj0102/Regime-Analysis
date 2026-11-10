"""EWMA one-step forecast must apply the RiskMetrics update exactly once."""
import numpy as np
import pandas as pd

from src.volatility import fit_ewma


def test_ewma_forecast_counts_last_return_once():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 300))
    r.iloc[-1] = 0.08  # a large final shock makes double counting visible
    lam = 0.94
    _, fcast = fit_ewma(r, lam=lam, warmup=20)
    v = r.iloc[0] ** 2
    for x in r.iloc[1:]:
        v = lam * v + (1 - lam) * x ** 2
    assert np.isclose(fcast, np.sqrt(v), rtol=1e-12)
