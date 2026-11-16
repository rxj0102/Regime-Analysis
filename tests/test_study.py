"""Checks on the study: no look-ahead, regressions, managed portfolios and statistics."""
import numpy as np
import pandas as pd
import pytest

from src import study
from src.data import INDUSTRY_NAMES, MARKET, build, load, parse_block
from src.study import Config

CFG = Config(warmup=200, min_months=24, n_boot=200)


def returns(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 250) % 2 == 0, 0.006, 0.018)       # alternating calm and turbulent years
    return pd.Series(rng.normal(0.0003, 1, n) * vol, index=pd.bdate_range("1990-01-01", periods=n))


@pytest.fixture(scope="module")
def r():
    return returns()


# ---------- data ----------

def test_snapshot_is_clean():
    d = load()
    assert list(d.columns) == [MARKET] + list(INDUSTRY_NAMES)
    assert d.index.is_monotonic_increasing and d.index.is_unique and not d.isna().any().any()
    assert d.index[0].year == 1926 and d.abs().max().max() < 0.5


def test_parser_and_build():
    f = ",Mkt-RF,SMB,HML,RF\n19260701, 0.10, 0, 0, 0.01\n19260702, 0.20, 0, 0, 0.01\n"
    i = "," + ",".join(INDUSTRY_NAMES) + "\n19260701," + ",".join(["0.51"] * 12) + "\n19260702," + ",".join(["-99.99"] + ["0.11"] * 11) + "\n"
    assert parse_block(f)["Mkt-RF"].iloc[1] == pytest.approx(0.002)
    out = build(f, i)
    assert len(out) == 1                                               # the day with a missing value is dropped
    assert out[MARKET].iloc[0] == pytest.approx(0.001) and out["NoDur"].iloc[0] == pytest.approx(0.005)


# ---------- regime ----------

def test_regime_uses_only_the_past(r):
    """Changing returns from day t on must leave every label up to day t unchanged."""
    t = 1500
    changed = r.copy()
    changed.iloc[t:] = 0.2
    a, b = study.regime(r, CFG), study.regime(changed, CFG)
    day = r.index[t]
    assert a.loc[:day].equals(b.loc[:day])
    assert not a.loc[r.index[t + 30]:].equals(b.loc[r.index[t + 30]:])


def test_regime_definition(r):
    high = study.regime(r, CFG)
    t = 1200
    vol = r.rolling(CFG.vol_window).std()
    expected = vol.iloc[t - 1] > vol.iloc[: t].median()
    assert high.loc[r.index[t]] == expected
    assert high.index[0] == r.index[CFG.warmup + CFG.vol_window - 1]   # first day with a full warm-up behind it


def test_regime_separates_calm_from_turbulent(r):
    out = study.market_by_regime(r, study.regime(r, CFG), CFG)
    assert out["vol_ratio"] > 1.8 and 0.2 < out["share_high"] < 0.8
    assert out["low"]["days"] + out["high"]["days"] == len(study.regime(r, CFG))


# ---------- regressions ----------

def test_ols_hac_recovers_coefficients_and_matches_white_without_lags():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(500), rng.normal(size=500)])
    y = X @ np.array([0.5, 2.0]) + rng.normal(size=500)
    b, t = study.ols_hac(y, X, 0)
    assert b == pytest.approx([0.5, 2.0], abs=0.15)
    u = y - X @ b
    xi = np.linalg.inv(X.T @ X)
    white = xi @ (X.T @ (X * (u ** 2)[:, None])) @ xi * 500 / 498
    assert t == pytest.approx(b / np.sqrt(np.diag(white)))


def test_beta_shift_is_detected_and_absent_when_absent(r):
    high = study.regime(r, CFG)
    rng = np.random.default_rng(2)
    noise = pd.Series(rng.normal(0, 0.004, len(r)), index=r.index)
    h = high.reindex(r.index, fill_value=False)
    shifted = r * (1.0 + 0.5 * h) + noise
    out = study.beta_by_regime(shifted, r, high, CFG)
    assert out["beta_low"] == pytest.approx(1.0, abs=0.05) and out["diff"] == pytest.approx(0.5, abs=0.05)
    assert out["t"] > 5
    flat = study.beta_by_regime(0.8 * r + noise, r, high, CFG)
    assert abs(flat["t"]) < 3 and flat["beta_high"] == pytest.approx(0.8, abs=0.05)


def test_holm_and_p_value():
    adj = study.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert study.p_two_sided(1.96) == pytest.approx(0.05, abs=0.0005)


# ---------- managed portfolios ----------

def test_monthly_aggregation(r):
    mo = study.monthly(r)
    jan = r.loc["1990-01"]
    assert mo["ret"].iloc[0] == pytest.approx(np.prod(1 + jan) - 1) and mo["rv"].iloc[0] == pytest.approx((jan ** 2).sum())


def test_weight_uses_only_earlier_months(r):
    """Change one month's returns. Weights up to and including that month must not move."""
    df = study.managed(r, CFG)
    month = df.index[20]
    changed = r.copy()
    mask = changed.index.to_period("M") == month
    changed[mask] = changed[mask] * 5
    again = study.managed(changed, CFG)
    pd.testing.assert_series_equal(df["weight"].loc[:month], again["weight"].loc[:month])
    assert again["weight"].loc[month + 1] < df["weight"].loc[month + 1]      # a wild month cuts next month's exposure


def test_weight_is_inverse_to_last_month_variance_and_capped(r):
    df = study.managed(r, CFG, cap=np.inf)
    rv = study.monthly(r)["rv"].shift(1).loc[df.index]
    hs = study.managed(r, CFG, cap=np.inf, hindsight=True)
    assert np.allclose(hs["weight"] * rv, (hs["weight"] * rv).iloc[0])       # one constant in hindsight
    assert not np.allclose(df["weight"] * rv, (df["weight"] * rv).iloc[0])   # a moving one in real time
    assert hs["managed_gross"].std() == pytest.approx(hs["unmanaged"].std())
    capped = study.managed(r, CFG)
    assert capped["weight"].max() <= CFG.cap and (capped["weight"] <= df["weight"] + 1e-12).all()


def test_managed_accounting(r):
    df = study.managed(r, CFG)
    assert np.allclose(df["managed_gross"], df["weight"] * df["unmanaged"])
    assert np.allclose(df["managed"], df["managed_gross"] - CFG.cost * df["turnover"])
    assert df["turnover"].iloc[0] == df["weight"].iloc[0]
    assert np.allclose(df["turnover"].iloc[1:], df["weight"].diff().abs().iloc[1:])


def test_management_helps_when_volatility_is_predictable_and_unpriced():
    """Same mean return in calm and wild years, so scaling down in the wild ones must help."""
    n = 12000
    vol = np.where((np.arange(n) // 250) % 2 == 0, 0.006, 0.018)
    long = pd.Series(0.0004 + np.random.default_rng(6).normal(0, 1, n) * vol, index=pd.bdate_range("1960-01-01", periods=n))
    st = study.managed_stats(study.managed(long, CFG), CFG)
    assert st["sharpe_managed"] > st["sharpe_unmanaged"] and st["alpha_t"] > 2


def test_alpha_is_zero_for_constant_exposure(r):
    df = study.managed(r, CFG)
    df["managed"] = 1.5 * df["unmanaged"]
    st = study.managed_stats(df, CFG, boot=False)
    assert st["alpha_ann"] == pytest.approx(0, abs=1e-12) and st["beta"] == pytest.approx(1.5)


def test_bootstrap_is_seeded_and_zero_for_identical_series(r):
    df = study.managed(r, CFG)
    u, m = df["unmanaged"].to_numpy(), df["managed"].to_numpy()
    assert study.sharpe_diff_interval(m, u, CFG) == study.sharpe_diff_interval(m, u, CFG)
    same = study.sharpe_diff_interval(u, u, CFG)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
