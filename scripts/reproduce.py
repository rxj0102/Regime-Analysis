"""Run the study on the frozen data and write results/.

  results/metrics.json           everything the README reports
  results/managed_monthly.csv    monthly unmanaged and managed returns and weights
  results/figures/alpha.png      alpha of volatility management, market and 12 industries
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import study  # noqa: E402
from src.data import INDUSTRY_NAMES, MARKET, load  # noqa: E402

LABEL = {MARKET: "Market", **INDUSTRY_NAMES}


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def figure(m: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.spines.top": False, "axes.spines.right": False})
    keys = sorted(m["managed"], key=lambda k: m["managed"][k]["alpha_ann"])
    alpha = np.array([m["managed"][k]["alpha_ann"] for k in keys]) * 100
    half = np.array([abs(m["managed"][k]["alpha_ann"] / m["managed"][k]["alpha_t"]) * 1.96 for k in keys]) * 100
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(y, alpha, height=0.6, color=["#2a78d6" if k == MARKET else "#8a8980" for k in keys])
    ax.errorbar(alpha, y, xerr=half, fmt="none", ecolor=ink, elinewidth=1, capsize=3)
    ax.axvline(0, color=ink, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[k] for k in keys])
    ax.set_xlabel("Alpha of the managed series on the unmanaged one, percent per year, after costs", color=muted)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("What scaling by past volatility adds, with 95% interval", loc="left", fontsize=10.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "alpha.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = study.Config()
    data = load()
    market = data[MARKET]
    high = study.regime(market, cfg)

    betas = {k: study.beta_by_regime(data[k], market, high, cfg) for k in INDUSTRY_NAMES}
    for k, p in study.holm({k: v["p"] for k, v in betas.items()}).items():
        betas[k]["p_holm"] = p

    managed, frames = {}, {}
    for k in data.columns:
        df = study.managed(data[k], cfg)
        frames[k] = df
        managed[k] = study.managed_stats(df, cfg)
        managed[k]["gross"] = study.managed_stats(df, cfg, "managed_gross", boot=False)
        managed[k]["uncapped"] = study.managed_stats(study.managed(data[k], cfg, cap=np.inf), cfg, boot=False)
        managed[k]["hindsight_uncapped_gross"] = study.managed_stats(
            study.managed(data[k], cfg, cap=np.inf, hindsight=True), cfg, "managed_gross", boot=False)
    for k, p in study.holm({k: v["p"] for k, v in managed.items()}).items():
        managed[k]["p_holm"] = p

    mk = frames[MARKET]
    n = len(mk)
    thirds = {f"{mk.index[a]} to {mk.index[b - 1]}": study.managed_stats(mk.iloc[a:b], cfg)
              for a, b in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n))}
    payload = rounded({
        "sample": {"data_first": str(data.index[0].date()), "data_last": str(data.index[-1].date()),
                   "regime_first": str(high.index[0].date()), "regime_days": int(len(high)),
                   "managed_first": str(mk.index[0]), "managed_last": str(mk.index[-1]), "managed_months": n,
                   "vol_window": cfg.vol_window, "cap": cfg.cap, "cost_bps": cfg.cost * 1e4,
                   "n_industries": len(INDUSTRY_NAMES)},
        "market_by_regime": study.market_by_regime(market, high, cfg),
        "betas": betas, "managed": managed, "market_thirds": thirds,
        "labels": LABEL,
    })
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    pd.concat({k: v[["unmanaged", "weight", "managed"]] for k, v in frames.items()}, axis=1).round(8).to_csv(
        out / "managed_monthly.csv", index_label="month")
    figure(payload, out / "figures")

    print(json.dumps(payload["sample"]))
    print(json.dumps(payload["market_by_regime"]))
    for k, v in betas.items():
        print(f"{k:6s} beta low {v['beta_low']:.2f} high {v['beta_high']:.2f} diff {v['diff']:+.3f} t {v['t']:+.2f} holm {v['p_holm']:.3f}")
    for k, v in managed.items():
        d = v["sharpe_diff"]
        print(f"{k:6s} alpha {v['alpha_ann']:+.3f} t {v['alpha_t']:+.2f} holm {v['p_holm']:.3f} SR {v['sharpe_unmanaged']:.2f}->{v['sharpe_managed']:.2f} "
              f"diff {d['diff']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}] w {v['mean_weight']:.2f} cap {v['share_at_cap']:.2f} to {v['turnover_per_year']:.1f} "
              f"gross a {v['gross']['alpha_ann']:+.3f} uncapped a {v['uncapped']['alpha_ann']:+.3f} t {v['uncapped']['alpha_t']:+.2f} dd {v['max_drawdown_unmanaged']:.2f}->{v['max_drawdown_managed']:.2f}")
    print({k: (round(v["hindsight_uncapped_gross"]["alpha_ann"], 3), round(v["hindsight_uncapped_gross"]["alpha_t"], 2),
               round(v["hindsight_uncapped_gross"]["sharpe_managed"], 2)) for k, v in managed.items()})
    for k, v in thirds.items():
        print(k, f"alpha {v['alpha_ann']:+.3f} t {v['alpha_t']:+.2f} SR {v['sharpe_unmanaged']:.2f}->{v['sharpe_managed']:.2f} [{v['sharpe_diff']['lo']:+.2f},{v['sharpe_diff']['hi']:+.2f}]")


if __name__ == "__main__":
    main()
