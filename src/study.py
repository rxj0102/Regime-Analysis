"""Do volatility regimes change risk, and can that be used?

Design, fixed before any result was computed:

  data       daily excess returns of the US market and 12 industries, July 1926 onward
  regime     day t is High when the market's volatility over the 22 days ending t-1 is above
             the median of that same measure over all days up to t-1. Nothing after t-1 is
             used, and the first 5 years only warm up the median.
  question 1 does the market's return or risk differ between regimes?
  question 2 do industry betas differ between regimes?
             r_i = a + b r_m + c High + d High r_m, test d, Holm across 12 industries
  question 3 can it be used? Volatility managed portfolios after Moreira and Muir (2017):
             in month m hold w = min(2, c / last month's variance) of the series, where c is
             set so that the managed series would have matched the unmanaged one's volatility
             over all earlier months. Everything uses daily returns up to the end of month m-1.
             Tested by the alpha of the managed series on the unmanaged one, and by the
             difference in Sharpe ratios, for the market and each industry. Costs are charged
             on every change in w.

Nothing is tuned. The leverage cap, windows and cost were set in advance. One thing was changed
after the first run and is disclosed in the README: the scaling constant c. The first version
used the average of past monthly variances, which held 1.6 times the series on average.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    vol_window: int = 22
    warmup: int = 1260
    nw_daily: int = 5
    nw_monthly: int = 6
    min_months: int = 60
    cap: float = 2.0
    cost: float = 0.0005
    block: int = 12
    n_boot: int = 2000
    seed: int = 0


# ---------- regime ----------

def regime(market: pd.Series, cfg: Config = Config()) -> pd.Series:
    """True on High volatility days. Uses market returns up to the previous day only."""
    vol = market.rolling(cfg.vol_window).std().shift(1)
    median = vol.expanding(min_periods=cfg.warmup).median()
    high = (vol > median).where(median.notna())
    return high.dropna().astype(bool)


# ---------- regressions ----------

def ols_hac(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West t-statistics."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = (y - X @ beta)[:, None] * X
    S = u.T @ u
    for j in range(1, lags + 1):
        g = u[j:].T @ u[:-j]
        S += (1.0 - j / (lags + 1.0)) * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv * n / (n - k)
    return beta, beta / np.sqrt(np.diag(cov))


def p_two_sided(t: float) -> float:
    return float(erfc(abs(t) / sqrt(2.0)))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def market_by_regime(market: pd.Series, high: pd.Series, cfg: Config) -> dict:
    r = market.loc[high.index].to_numpy()
    h = high.to_numpy()
    out = {}
    for name, mask in (("low", ~h), ("high", h)):
        x = r[mask]
        out[name] = {"days": int(mask.sum()), "ann_return": float(x.mean() * 252),
                     "ann_vol": float(x.std(ddof=1) * np.sqrt(252)),
                     "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
                     "worst_day": float(x.min()), "var_99": float(-np.quantile(x, 0.01))}
    X = np.column_stack([np.ones(len(r)), h.astype(float)])
    b, t = ols_hac(r, X, cfg.nw_daily)
    out["return_diff"] = {"ann_diff": float(b[1] * 252), "t": float(t[1])}
    out["vol_ratio"] = out["high"]["ann_vol"] / out["low"]["ann_vol"]
    out["share_high"] = float(h.mean())
    runs = np.diff(np.flatnonzero(np.diff(h.astype(int)) != 0))
    out["mean_spell_days"] = float(runs.mean()) if len(runs) else float("nan")
    return out


def beta_by_regime(asset: pd.Series, market: pd.Series, high: pd.Series, cfg: Config) -> dict:
    y = asset.loc[high.index].to_numpy()
    m = market.loc[high.index].to_numpy()
    h = high.to_numpy().astype(float)
    b, t = ols_hac(y, np.column_stack([np.ones(len(y)), m, h, h * m]), cfg.nw_daily)
    return {"beta_low": float(b[1]), "beta_high": float(b[1] + b[3]), "diff": float(b[3]), "t": float(t[3]),
            "p": p_two_sided(float(t[3]))}


# ---------- volatility managed portfolios ----------

def monthly(r: pd.Series) -> pd.DataFrame:
    """Monthly compounded return and realised variance (sum of squared daily returns)."""
    g = r.groupby(r.index.to_period("M"))
    return pd.DataFrame({"ret": g.apply(lambda x: float(np.prod(1.0 + x) - 1.0)), "rv": g.apply(lambda x: float((x ** 2).sum()))})


def managed(r: pd.Series, cfg: Config = Config(), cap: float | None = None, hindsight: bool = False) -> pd.DataFrame:
    """Unmanaged and managed monthly returns. The weight for month m uses data up to m-1.

    hindsight=True reproduces the textbook version for comparison: one constant c for the whole
    sample, chosen with all the data so that the two series have the same volatility.
    """
    cap = cfg.cap if cap is None else cap
    mo = monthly(r)
    inv = 1.0 / mo["rv"].shift(1)
    raw = inv * mo["ret"]                                   # unscaled managed return
    # scale so that, on past months only, the managed series was as volatile as the unmanaged one
    c = (mo["ret"].expanding(min_periods=cfg.min_months).std() / raw.expanding(min_periods=cfg.min_months).std()).shift(1)
    if hindsight:
        start = c.first_valid_index()
        c = pd.Series(mo["ret"].loc[start:].std() / raw.loc[start:].std(), index=mo.index).where(c.notna())
    w = (c * inv).clip(upper=cap)
    out = pd.DataFrame({"unmanaged": mo["ret"], "weight": w}).dropna()
    out["turnover"] = out["weight"].diff().abs().fillna(out["weight"].iloc[0])
    out["managed_gross"] = out["weight"] * out["unmanaged"]
    out["managed"] = out["managed_gross"] - cfg.cost * out["turnover"]
    return out


def sharpe_m(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def sharpe_diff_interval(a: np.ndarray, b: np.ndarray, cfg: Config) -> dict:
    """Paired circular block bootstrap of Sharpe(a) - Sharpe(b)."""
    rng = np.random.default_rng(cfg.seed)
    n = len(a)
    n_blocks = int(np.ceil(n / cfg.block))
    offsets = np.arange(cfg.block)
    draws = np.empty(cfg.n_boot)
    for i in range(cfg.n_boot):
        idx = ((rng.integers(0, n, n_blocks)[:, None] + offsets) % n).ravel()[:n]
        draws[i] = sharpe_m(a[idx]) - sharpe_m(b[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": sharpe_m(a) - sharpe_m(b), "lo": float(lo), "hi": float(hi)}


def managed_stats(df: pd.DataFrame, cfg: Config, column: str = "managed", boot: bool = True) -> dict:
    u, m = df["unmanaged"].to_numpy(), df[column].to_numpy()
    b, t = ols_hac(m, np.column_stack([np.ones(len(u)), u]), cfg.nw_monthly)
    cum = np.cumprod(1.0 + m)
    cum_u = np.cumprod(1.0 + u)
    out = {"months": int(len(u)), "alpha_ann": float(b[0] * 12), "alpha_t": float(t[0]), "p": p_two_sided(float(t[0])),
           "beta": float(b[1]), "sharpe_unmanaged": sharpe_m(u), "sharpe_managed": sharpe_m(m),
           "vol_unmanaged": float(u.std(ddof=1) * np.sqrt(12)), "vol_managed": float(m.std(ddof=1) * np.sqrt(12)),
           "max_drawdown_unmanaged": float((cum_u / np.maximum.accumulate(cum_u) - 1).min()),
           "max_drawdown_managed": float((cum / np.maximum.accumulate(cum) - 1).min()),
           "mean_weight": float(df["weight"].mean()), "max_weight": float(df["weight"].max()), "share_at_cap": float((df["weight"] >= cfg.cap - 1e-12).mean()),
           "turnover_per_year": float(df["turnover"].mean() * 12)}
    if boot:
        out["sharpe_diff"] = sharpe_diff_interval(m, u, cfg)
    return out
