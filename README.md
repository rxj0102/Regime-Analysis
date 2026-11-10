# Regime-Dependent Risk & Performance Attribution

[![CI](https://github.com/rxj0102/Regime-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/Regime-Analysis/actions/workflows/ci.yml)

Empirical study showing that CAPM beta, volatility, VaR, and Sharpe ratio all vary materially across market regimes — and that full-sample averages are statistical artefacts that mask an asset's true risk profile.

---

## Problem & Methodology

Standard risk metrics (beta, Sharpe, VaR) are almost always reported as full-sample averages. This paper tests whether those averages are informative or misleading.

**Approach:**

1. **Regime detection** — SPY 30-day rolling volatility is split at its in-sample median into two balanced regimes: *Low-Vol* and *High-Vol* (~500 days each).
2. **Per-regime risk metrics** — annualised return, volatility, Sharpe, Sortino, max drawdown, Calmar, 95% VaR and CVaR are computed separately for each regime.
3. **Regime-conditional CAPM** — OLS with HC3 heteroskedasticity-robust standard errors is fit per regime, revealing beta instability (assets correlate more with the market during crises).
4. **Volatility forecasting** — EWMA (λ = 0.94, RiskMetrics) and GARCH(1,1) are compared; the best error distribution (Normal / Student-t / Skewed-t) is selected by AIC.
5. **VaR backtesting** — rolling historical VaR is validated with Kupiec's POF test (correct coverage) and Christoffersen's IND test (no violation clustering).
6. **Stress testing** — equal-weight portfolio P&L and CVaR are computed during the 2020 COVID crash and the 2022 rate shock.

---

## Corrections

- **VaR look-ahead.** The rolling historical VaR for day t included day t's own return. It is now lagged one day, so the VaR applied to day t uses data through t-1.
- **Regime slices.** VaR was previously estimated on each regime subset (non-contiguous days), and the independence test treated gaps as consecutive days. VaR is now estimated on the full series, and only truly adjacent trading days enter the transition counts.
- Backtest tables and figures in the notebook predate this change; re-run it to refresh them.


## Key Results

| Finding | Detail |
|---|---|
| **Beta non-stationarity** | All three assets show Δβ > 0 across regimes — market exposure increases by 20–60% in High-Vol periods, exactly when diversification is needed most. |
| **Sharpe reversal** | Full-sample Sharpe ratios are positive for all assets, yet High-Vol Sharpe ratios are negative — the average conceals regime-specific loss. |
| **CVaR multiplier** | High-Vol CVaR is 2–4× Low-Vol CVaR across assets; Historical VaR underestimates tail risk under normal regimes and overestimates it during crises. |
| **VaR violation clustering** | Full-sample rolling VaR fails the Christoffersen IND test — violations cluster during stress, violating the i.i.d. assumption of standard backtesting. |
| **GARCH > EWMA** | GARCH(1,1) with Student-t or Skewed-t errors wins by AIC on all four assets; EWMA consistently underestimates vol persistence (α + β). |
| **Stress P&L** | Equal-weight portfolio lost ~20% (≈ $200k on $1M) in the COVID crash and ~15% in the 2022 rate shock. |

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

The notebook fetches data directly from Yahoo Finance — no local data files required.

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
All current results are in-sample — the regime threshold, CAPM betas, and risk metrics are computed on the full 2019–2024 period. A rigorous evaluation would use an expanding (or rolling) window: estimate the model up to date *t*, generate risk forecasts for *t+1*, then roll forward. This enables proper walk-forward validation of VaR coverage and Sharpe stability, and tests whether the regime signal is exploitable in real time.

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)
- Files shipped in the `data/` folder of this repository

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (22 tests) finishes in about 34 seconds on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.

