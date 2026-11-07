import numpy as np
import pandas as pd

from src.backtest import backtest_var
from src.metrics import christoffersen_test


def _data(n=900, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, n)}, index=idx)
    regime = pd.Series(np.where(np.arange(n) % 2 == 0, "Low-Vol", "High-Vol"), index=idx)
    return r, regime


def test_iid_coverage_is_near_nominal():
    """With a lagged VaR on iid data, the 5% violation rate is near 5%."""
    r, regime = _data()
    out = backtest_var(r, regime)
    assert {"Full Sample", "Low-Vol", "High-Vol"} <= set(out["Regime"])
    # iid normal at 5%: coverage should be near nominal
    full = out[out.Regime == "Full Sample"].iloc[0]
    assert 0.03 < full["pi_hat"] < 0.08


def test_alternating_regime_has_no_adjacent_pairs():
    r, regime = _data()
    out = backtest_var(r, regime)
    # alternating labels: no two kept days are consecutive, so IND carries no information
    sub = out[out.Regime == "Low-Vol"].iloc[0]
    assert sub["lr_ind"] == 0.0


def test_adjacent_mask_ignores_gap_pairs():
    v = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    full = christoffersen_test(v)
    gaps = christoffersen_test(v, adjacent=np.zeros(7, dtype=bool))
    assert gaps["lr_ind"] == 0.0
    assert full["lr_ind"] > 0.0
