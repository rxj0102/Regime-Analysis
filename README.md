# Regime-Dependent Risk & Performance Attribution

[![CI](https://github.com/rxj0102/Regime-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/Regime-Analysis/actions/workflows/ci.yml)

Empirical study showing that CAPM beta, volatility, VaR, and Sharpe ratio all vary materially across market regimes — and that full-sample averages are statistical artefacts that mask an asset's true risk profile.

---

## Data

Free open data from the Ken French Data Library (no Yahoo Finance), daily, 2019-01-02 to 2023-12-29 (1,258 days):

| Label | Series | Replaces |
|---|---|---|
| TECH | 12 industry portfolio BusEq (business equipment), value weighted | NVDA |
| FIN | 12 industry portfolio Money (finance) | JPM |
| STAPLES | 12 industry portfolio NoDur (consumer nondurables) | PG |
| SPY | US market, Mkt-RF + RF | SPY |

Industry portfolios are diversified, so they are smoother than the single stocks the study used before. Read the results as sector level, not stock level.

---

## Problem & Methodology

Standard risk metrics (beta, Sharpe, VaR) are almost always reported as full-sample averages. This paper tests whether those averages are informative or misleading.

**Approach:**

1. **Regime detection**: the market's 30-day rolling volatility at t-1 is compared with its expanding median up to t-1, so labels use no future data (an earlier version used the full-sample median).
2. **Per-regime risk metrics** — annualised return, volatility, Sharpe, Sortino, max drawdown, Calmar, 95% VaR and CVaR are computed separately for each regime.
3. **Regime-conditional CAPM** — OLS with HC3 heteroskedasticity-robust standard errors is fit per regime, revealing beta instability (assets correlate more with the market during crises).
4. **Volatility forecasting** — EWMA (λ = 0.94, RiskMetrics) and GARCH(1,1) are compared; the best error distribution (Normal / Student-t / Skewed-t) is selected by AIC.
5. **VaR backtesting** — rolling historical VaR is validated with Kupiec's POF test (correct coverage) and Christoffersen's IND test (no violation clustering).
6. **Stress testing** — equal-weight portfolio P&L and CVaR are computed during the 2020 COVID crash and the 2022 rate shock.

---

## Corrections

- **VaR look-ahead.** The rolling historical VaR for day t included day t's own return. It is now lagged one day, so the VaR applied to day t uses data through t-1.
- **Regime slices.** VaR was previously estimated on each regime subset (non-contiguous days), and the independence test treated gaps as consecutive days. VaR is now estimated on the full series, and only truly adjacent trading days enter the transition counts.
- The notebook has been re-run after these fixes on the open data above.
- **Regime labels look-ahead.** Labels came from the same day's rolling vol against the full-sample median. They now use the prior day's rolling vol against an expanding median, which changes the main Sharpe result (see Key Results).
- **Stress beta.** `fit_capm` required 30 observations, so the 24-day COVID window always returned NaN. Stress windows now allow 20.


## Key Results

All numbers are copied from the executed notebook `07_regime_analysis.ipynb`.

**Regimes are labelled point in time.** Day t is High-Vol if the market's 30-day rolling vol at t-1 is above the expanding median of all rolling-vol readings up to t-1 (warmed up on Ken French market returns from 2010). By that standard most of 2019 to 2023 was a high volatility period: 854 High-Vol days and 403 Low-Vol days; the threshold drifts from 0.00765 to 0.00830 daily sigma.

| Asset | Low-Vol Sharpe | High-Vol Sharpe | Full Sharpe | Low-Vol CVaR95 | High-Vol CVaR95 | CVaR multiple |
|---|---|---|---|---|---|---|
| TECH | 0.950 | 0.770 | 0.784 | -2.72% | -4.44% | 1.63x |
| FIN | -0.064 | 0.529 | 0.388 | -2.42% | -4.56% | 1.89x |
| STAPLES | -0.833 | 0.717 | 0.356 | -1.82% | -3.17% | 1.75x |
| SPY | 0.147 | 0.736 | 0.588 | -2.10% | -3.86% | 1.84x |

| Asset | Full beta | Low-Vol beta | High-Vol beta | Change |
|---|---|---|---|---|
| TECH | 1.161 | 1.241 | 1.152 | -7.1% |
| FIN | 1.071 | 1.004 | 1.079 | +7.5% |
| STAPLES | 0.686 | 0.597 | 0.697 | +16.7% |

| Finding | Detail |
|---|---|
| **Look-ahead flips the Sharpe story** | With the earlier full-sample, same-day labelling, every asset had a much lower Sharpe in High-Vol (market: 1.63 Low-Vol vs 0.19 High-Vol). Labelled point in time, the market, FIN and STAPLES do better in High-Vol (market: 0.15 vs 0.74). Same-day labels file a crash day under High-Vol because the crash itself raises the rolling vol, which builds the Sharpe collapse into the split. Known-in-advance high volatility was followed by higher, not lower, risk-adjusted returns in this sample (which includes the 2020 rebound). |
| **Tail risk still rises** | High-Vol CVaR95 is 1.6x to 1.9x Low-Vol CVaR95 for every asset. That part of the regime effect survives the fix. |
| **Beta shifts are modest and mixed** | FIN and STAPLES become more market exposed in High-Vol (+7.5% and +16.7%); TECH beta falls 7.1%. |
| **VaR violations cluster in High-Vol** | Rolling VaR coverage is close to 5% in both regimes, but High-Vol exceedances fail the Christoffersen IND test for FIN, STAPLES and SPY (p = 0.0008, 0.037, 0.007). |
| **Skewed-t GARCH wins** | Skewed-t errors give the best AIC for all four series; alpha + beta persistence is 0.966 to 0.987. |
| **Stress P&L** | Equal weight portfolio: -34.9% in the 2020 COVID window and -21.3% in the 2022 rate shock window. Stress betas in the COVID window (24 days): TECH 1.06, FIN 1.22, STAPLES 0.87. |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `src/opendata.py` | Ken French Data Library download and cache |
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
│   (none; the notebook is 07_regime_analysis.ipynb at the repo root)
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
jupyter lab 07_regime_analysis.ipynb
```

The notebook downloads Ken French data on first run (no API key) and caches it in `data/open_cache/`.

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
The regime threshold is now point in time, but CAPM betas and risk metrics are still estimated in-sample over 2019 to 2023. A rigorous evaluation would use an expanding (or rolling) window: estimate the model up to date *t*, generate risk forecasts for *t+1*, then roll forward. This enables proper walk-forward validation of VaR coverage and Sharpe stability, and tests whether the regime signal is exploitable in real time.
## Data snapshot

`data/open_data_manifest.json` records every open data file behind the results above (SHA-256, rows, first and last date). Ken French and FRED occasionally revise history; after a fresh download, check whether the inputs still match with:

```bash
python -c "from src.opendata import verify_manifest; print(verify_manifest('data/open_data_manifest.json'))"
```
