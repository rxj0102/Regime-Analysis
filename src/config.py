"""Global configuration: tickers, date range, model constants, and plot styles."""

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Universe & data range
# ---------------------------------------------------------------------------
TICKERS: list[str] = ["NVDA", "JPM", "PG", "SPY"]
START_DATE: str = "2019-01-01"
END_DATE: str = "2024-01-01"

# ---------------------------------------------------------------------------
# Risk-free rate
# 3-month T-bill average 2019-2023. Using 0% overstates Sharpe materially
# for 2022-23 when the Fed Funds rate hit 4-5%.
# ---------------------------------------------------------------------------
RF_ANNUAL: float = 0.025
RF_DAILY: float = RF_ANNUAL / 252

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------
REGIME_WINDOW: int = 30  # rolling vol window (trading days)

# ---------------------------------------------------------------------------
# Volatility models
# ---------------------------------------------------------------------------
EWMA_LAMBDA: float = 0.94   # RiskMetrics daily decay factor
EWMA_WARMUP: int = 60        # burn-in days replaced by rolling variance
GARCH_DISTS: list[str] = ["normal", "t", "skewt"]

# ---------------------------------------------------------------------------
# Stress scenarios
# ---------------------------------------------------------------------------
STRESS_EVENTS: dict[str, tuple[str, str, str]] = {
    "2020 COVID Crash": ("2020-02-19", "2020-03-23", "SPY -34% in 23 trading days"),
    "2022 Rate Shock":  ("2022-01-03", "2022-06-16", "SPY -24% over ~120 days"),
}

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "Low-Vol":     "#2ecc71",
    "High-Vol":    "#e74c3c",
    "Full Sample": "#3498db",
}
TICKER_COLORS: dict[str, str] = {
    "NVDA": "#9b59b6",
    "JPM":  "#e67e22",
    "PG":   "#1abc9c",
    "SPY":  "#2c3e50",
}


def setup_style() -> None:
    """Apply project-wide matplotlib/seaborn style."""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
