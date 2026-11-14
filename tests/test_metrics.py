"""Unit tests for src.metrics: calc_metrics and christoffersen_test."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import calc_metrics, christoffersen_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _random_returns(n: int = 500, mu: float = 0.0005, sigma: float = 0.01) -> pd.Series:
    return pd.Series(RNG.normal(mu, sigma, n))


# ---------------------------------------------------------------------------
# calc_metrics
# ---------------------------------------------------------------------------

class TestCalcMetrics:
    def test_returns_correct_keys(self):
        m = calc_metrics(_random_returns(), "Test")
        expected = {"Label", "N", "AnnRet", "AnnVol", "Sharpe",
                    "Sortino", "MaxDD", "Calmar", "VaR95", "CVaR95"}
        assert set(m.keys()) == expected

    def test_empty_series_returns_empty_dict(self):
        assert calc_metrics(pd.Series([], dtype=float), "Empty") == {}

    def test_single_obs_returns_empty_dict(self):
        assert calc_metrics(pd.Series([0.01]), "Single") == {}

    def test_ann_vol_positive(self):
        m = calc_metrics(_random_returns(), "Test")
        assert m["AnnVol"] > 0

    def test_cvar_le_var(self):
        """CVaR (expected shortfall) must be at least as bad as VaR."""
        m = calc_metrics(_random_returns(), "Test")
        assert m["CVaR95"] <= m["VaR95"]

    def test_max_drawdown_non_positive(self):
        m = calc_metrics(_random_returns(), "Test")
        assert m["MaxDD"] <= 0

    def test_n_matches_input_length(self):
        r = _random_returns(300)
        m = calc_metrics(r, "Test")
        assert m["N"] == 300

    def test_label_stored(self):
        m = calc_metrics(_random_returns(), "MyLabel")
        assert m["Label"] == "MyLabel"

    def test_constant_positive_returns_positive_sharpe(self):
        r = pd.Series([0.002] * 252)
        m = calc_metrics(r, "Test")
        assert m["Sharpe"] > 0

    def test_constant_negative_returns_negative_sharpe(self):
        r = pd.Series([-0.002] * 252)
        m = calc_metrics(r, "Test")
        assert m["Sharpe"] < 0


# ---------------------------------------------------------------------------
# christoffersen_test
# ---------------------------------------------------------------------------

class TestChristoffersenTest:
    def test_returns_correct_keys(self):
        v = pd.Series([0, 0, 1, 0] * 50)
        result = christoffersen_test(v)
        assert set(result.keys()) == {"T", "N", "pi_hat", "lr_pof", "p_pof",
                                       "lr_ind", "p_ind"}

    def test_p_values_in_unit_interval(self):
        v = pd.Series((RNG.uniform(size=500) < 0.05).astype(int))
        r = christoffersen_test(v, alpha=0.05)
        assert 0.0 <= r["p_pof"] <= 1.0
        assert 0.0 <= r["p_ind"] <= 1.0

    def test_no_violations_fails_pof(self):
        """Zero violations → violation rate = 0, POF must reject."""
        v = pd.Series([0] * 250)
        r = christoffersen_test(v, alpha=0.05)
        assert r["N"] == 0
        assert r["p_pof"] == 0.0

    def test_pi_hat_exact(self):
        v = pd.Series([1, 0] * 100)   # exactly 50% violation rate
        r = christoffersen_test(v, alpha=0.05)
        assert abs(r["pi_hat"] - 0.5) < 1e-10

    def test_t_equals_input_length(self):
        v = pd.Series([0] * 180 + [1] * 20)
        r = christoffersen_test(v, alpha=0.05)
        assert r["T"] == 200

    def test_n_counts_violations(self):
        v = pd.Series([0] * 190 + [1] * 10)
        r = christoffersen_test(v, alpha=0.05)
        assert r["N"] == 10

    def test_well_calibrated_model_passes_pof(self):
        """Calibrated at exactly alpha → POF should not reject at 5%."""
        rng = np.random.default_rng(0)
        v = pd.Series((rng.uniform(size=2000) < 0.05).astype(int))
        r = christoffersen_test(v, alpha=0.05)
        # With 2000 obs, a correctly calibrated model should pass POF
        assert r["p_pof"] > 0.01

    def test_independent_violations_pass_ind(self):
        """Purely iid violations → IND should not reject."""
        rng = np.random.default_rng(1)
        v = pd.Series((rng.uniform(size=2000) < 0.05).astype(int))
        r = christoffersen_test(v, alpha=0.05)
        assert r["p_ind"] > 0.01

    def test_clustered_violations_fail_ind(self):
        """All violations bunched at the end → IND should reject."""
        v = pd.Series([0] * 950 + [1] * 50)
        r = christoffersen_test(v, alpha=0.05)
        assert r["p_ind"] < 0.05


def test_point_in_time_regimes_do_not_use_future_data():
    import numpy as np
    import pandas as pd
    from src.regimes import classify_regimes
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=800)
    r = pd.DataFrame({"SPY": rng.normal(0, 0.01, 800)}, index=idx)
    base, _, _ = classify_regimes(r, point_in_time=True)
    shocked = r.copy()
    shocked.iloc[700:, 0] *= 10  # change only the future
    alt, _, _ = classify_regimes(shocked, point_in_time=True)
    common = base.index[base.index < idx[700]]
    pd.testing.assert_series_equal(base.loc[common], alt.loc[common])
