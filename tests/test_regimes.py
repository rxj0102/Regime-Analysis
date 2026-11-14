"""Regime classification: full-sample vs expanding median."""
import numpy as np
import pandas as pd
import pytest

from src.regimes import classify_regimes


def _rets(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    vol = np.where(np.arange(n) % 300 < 150, 0.008, 0.02)
    return pd.DataFrame({"SPY": rng.normal(0, vol), "X": rng.normal(0, 0.01, n)}, index=idx)


def test_expanding_labels_are_point_in_time():
    r = _rets()
    base, _, _ = classify_regimes(r, method="expanding")
    cut = r.index[800]
    alt = r.copy()
    alt.loc[alt.index > cut, "SPY"] *= 5  # change only the future
    new, _, _ = classify_regimes(alt, method="expanding")
    common = base.index[base.index <= cut]
    pd.testing.assert_series_equal(base.loc[common], new.loc[common])


def test_full_sample_labels_do_depend_on_the_future():
    r = _rets()
    base, _, _ = classify_regimes(r)
    cut = r.index[800]
    alt = r.copy()
    alt.loc[alt.index > cut, "SPY"] *= 5
    new, _, _ = classify_regimes(alt)
    common = base.index[base.index <= cut]
    assert not base.loc[common].equals(new.loc[common])


def test_expanding_needs_min_periods():
    r = _rets()
    reg, _, thr = classify_regimes(r, method="expanding", window=30, min_periods=252)
    assert isinstance(thr, pd.Series)
    first = reg.index[0]
    assert r.index.get_loc(first) >= 30 - 1 + 252 - 1


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        classify_regimes(_rets(), method="oracle")


def test_full_sample_warmup_days_are_unlabelled():
    r = _rets()
    reg, rets_a, _ = classify_regimes(r, window=30)
    assert reg.index[0] == r.index[29]
    assert len(rets_a) == len(r) - 29
