# Regime-Dependent Risk & Performance Attribution

[![CI](https://github.com/rxj0102/Regime-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/Regime-Analysis/actions/workflows/ci.yml)

Empirical study showing that CAPM beta, volatility, VaR, and Sharpe ratio all vary materially across market regimes - and that full-sample averages are statistical artefacts that mask an asset's true risk profile.

---

## Problem & Methodology

Standard risk metrics (beta, Sharpe, VaR) are almost always reported as full-sample averages. This paper tests whether those averages are informative or misleading.

**Approach:**

1. **Regime detection** - SPY 30-day rolling volatility is split at its in-sample median into two balanced regimes: *Low-Vol* and *High-Vol* (~500 days each).
2. **Per-regime risk metrics** - annualised return, volatility, Sharpe, Sortino, max drawdown, Calmar, 95% VaR and CVaR are computed separately for each regime.
3. **Regime-conditional CAPM** - OLS with HC3 heteroskedasticity-robust standard errors is fit per regime, revealing beta instability (assets correlate more with the market during crises).
4. **Volatility forecasting** - EWMA (λ = 0.94, RiskMetrics) and GARCH(1,1) are compared; the best error distribution (Normal / Student-t / Skewed-t) is selected by AIC.
5. **VaR backtesting** - rolling historical VaR is validated with Kupiec's POF test (correct coverage) and Christoffersen's IND test (no violation clustering).
6. **Stress testing** - equal-weight portfolio P&L and CVaR are computed during the 2020 COVID crash and the 2022 rate shock.

---

## Corrections

- **VaR look-ahead.** The rolling historical VaR for day t included day t's own return. It is now lagged one day, so the VaR applied to day t uses data through t-1.
- **Regime slices.** VaR was previously estimated on each regime subset (non-contiguous days), and the independence test treated gaps as consecutive days. VaR is now estimated on the full series, and only truly adjacent trading days enter the transition counts.
- Warm-up days without a rolling volatility were labelled Low-Vol; they are now unlabelled.
- Two notebook cells used `row.skew`, `row.kurt` and `row.T`, which pandas resolves to Series methods, so the notebook could not run end to end. Fixed.
- The notebook was re-run on 2026-11-11 and Key Results now come from that run. The previous table claimed beta rises for every asset, negative High-Vol Sharpe ratios and stress losses of about 20% and 15%; the run does not support any of those.


## Key Results

From `notebooks/regime_analysis.ipynb`, run end to end on 2026-11-11 (yfinance adjusted closes, 2019-01-02 to 2023-12-29, risk-free rate 2.5%). Source files: `outputs/regime_metrics.csv`, `outputs/regime_betas.csv`, `outputs/var_backtest.csv`, `outputs/stress_portfolio.csv`, `outputs/regime_method_comparison.csv`. Regimes split SPY 30-day volatility at its full-sample median (614 days each).

**Risk-adjusted return by regime**

| Ticker | Full-sample Sharpe | Low-Vol Sharpe | High-Vol Sharpe | High/Low CVaR 95 |
|---|---|---|---|---|
| NVDA | 1.31 | 2.42 | 0.68 | 1.73x |
| JPM | 0.36 | 1.21 | 0.03 | 2.34x |
| PG | 0.40 | 0.60 | 0.31 | 1.88x |
| SPY | 0.53 | 1.47 | 0.20 | 2.39x |

Every asset earns a lower Sharpe in High-Vol periods, but none is negative under the full-sample split. Tail risk roughly doubles: High-Vol CVaR is 1.7 to 2.4 times Low-Vol CVaR.

**Market beta by regime** (HC3 standard errors)

| Ticker | Full beta | Low-Vol beta | High-Vol beta | Change |
|---|---|---|---|---|
| NVDA | 1.74 | 2.01 | 1.69 | -15.6% |
| JPM | 1.12 | 0.94 | 1.15 | +22.6% |
| PG | 0.59 | 0.43 | 0.61 | +41.4% |

Beta rises in High-Vol periods for JPM and PG but falls for NVDA, so a single full-sample beta misstates exposure in both directions.

**95% historical VaR backtest** (rolling, applied to day t with data through t-1; expected rate 5%)

| Ticker | Full-sample rate | POF | IND | Low-Vol rate | High-Vol rate |
|---|---|---|---|---|---|
| NVDA | 6.34% | fail | pass | 3.96% | 8.48% |
| JPM | 5.22% | pass | fail | 2.88% | 7.34% |
| PG | 6.16% | pass | pass | 4.32% | 7.83% |
| SPY | 5.22% | pass | fail | 3.42% | 6.85% |

For every asset the violation rate is below 5% in calm periods and above it in stress periods. Over the full sample, JPM and SPY fail the Christoffersen independence test (violations cluster).

**Stress test, equal-weight portfolio of the four assets** (2020 COVID crash and 2022 rate shock windows): -31.7% and -29.6%, about $317k and $296k on $1M.

**Point-in-time check.** The full-sample median uses future data. With an expanding median (`classify_regimes(method="expanding")`, labels from 2020-02-13), regimes agree with the full-sample labels on 92.4% of common days. The numbers move: under the expanding split JPM's High-Vol Sharpe turns negative (-0.14), and PG no longer shows a gap (High-Vol 0.21 vs Low-Vol 0.21). The other three assets keep a lower Sharpe in High-Vol periods. See `outputs/regime_method_comparison.csv`.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `yfinance` | Daily OHLCV download (split/dividend-adjusted) |
| `pandas` / `numpy` | Data manipulation, log returns, rolling statistics |
| `statsmodels` | OLS + HC3 robust SEs, ADF stationarity test |
| `arch` | GARCH(1,1) with Normal / Student-t / Skewed-t errors |
| `scipy` | Jarque-Bera, Student-t fitting, chi-squared CDF |
| `matplotlib` / `seaborn` | All visualisations |
| `pytest` | Unit tests for statistical functions |

---

## Repository Structure

```
Regime-Analysis/
├── src/                        # Importable Python package
│   ├── config.py               # Tickers, dates, constants, colour palettes
│   ├── data.py                 # Price download, log returns, diagnostics
│   ├── regimes.py              # Rolling-vol median-split classifier
│   ├── metrics.py              # Risk metrics + Christoffersen test
│   ├── capm.py                 # HC3 OLS CAPM, regime-beta table
│   ├── volatility.py           # EWMA and GARCH(1,1) fitting
│   ├── backtest.py             # Rolling VaR backtesting (Kupiec + Christoffersen)
│   ├── stress.py               # Historical scenario stress testing
│   └── plots.py                # All visualisation functions
├── notebooks/
│   └── regime_analysis.ipynb   # Clean, end-to-end analysis notebook
├── tests/
│   └── test_metrics.py         # Unit tests for metrics and backtesting
├── data/                       # Generated outputs (plots saved here)
├── requirements.txt
└── .gitignore
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the notebook**
```bash
cd notebooks
jupyter lab regime_analysis.ipynb
```

The notebook fetches data directly from Yahoo Finance - no local data files required.

**3. Run tests**
```bash
pytest tests/ -v
```

**4. Import modules directly**
```python
from src import data, regimes, metrics, capm, volatility, backtest, stress, plots
from src.config import setup_style

setup_style()
prices = data.load_prices()
rets = data.log_returns(prices)
regime, rets_a, threshold = regimes.classify_regimes(rets)
```

---

## Suggested Research-Level Improvements

### 1. Hidden Markov Model (HMM) Regime Detection
Replace the hard-threshold vol classifier with a 2-state (or 3-state) Gaussian HMM fitted on returns and/or realised vol. HMMs produce *probabilistic* regime assignments and learn transition probabilities from the data, avoiding the look-ahead bias of a full-sample median threshold. Expanding-window estimation preserves out-of-sample validity.

**Relevant tools:** `hmmlearn`, `pomegranate`

### 2. Regime-Switching GARCH (RS-GARCH)
The current pipeline fits a single GARCH model per asset, then slices results by regime post-hoc. A proper Markov-switching GARCH (Gray 1996; Haas et al. 2004) allows the GARCH parameters themselves to switch between regimes, capturing the empirical finding that vol clustering is more persistent in High-Vol states. This would yield regime-conditional VaR forecasts directly from the model rather than from retrospective slicing.

**Relevant tools:** `arch` (MS-GARCH extension), custom EM implementation

### 3. Out-of-Sample Rolling Backtest with Expanding Window
All current results are in-sample - the regime threshold, CAPM betas, and risk metrics are computed on the full 2019–2024 period. A rigorous evaluation would use an expanding (or rolling) window: estimate the model up to date *t*, generate risk forecasts for *t+1*, then roll forward. This enables proper walk-forward validation of VaR coverage and Sharpe stability, and tests whether the regime signal is exploitable in real time.