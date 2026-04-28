"""
Act 11 — Mechanism summary bars (compact)

Uses derived per-trial diagnostics to show *how* algorithms differ,
not just final HV/IGD.

Outputs:
  - synthetic_pairwise_mechanism_summary_bars.pdf
  - synthetic_pairwise_mechanism_summary_bars.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

EXTRA = ROOT / "benchmark_pairwise_2problems_20260313" / "pairwise_all10_extra_metrics_per_trial.csv"
PROC  = ROOT / "benchmark_pairwise_2problems_20260313" / "pairwise_all10_process_metrics_per_trial.csv"

sys.path.insert(0, str(HERE.parent))
from plot_style import apply_style, save_fig

apply_style()

ALGO_ORDER = ["EGBO_Novelty_v1", "EGBO", "Traditional_NEHVI"]
ALGO_LABELS = {
    "EGBO_Novelty_v1": "Novelty Aware EGBO",
    "EGBO": "EGBO",
    "Traditional_NEHVI": "qNEHVI",
}
ALGO_COLORS = {
    "EGBO_Novelty_v1": "#8E24AA",
    "EGBO": "#2E7D32",
    "Traditional_NEHVI": "#1565C0",
}


def build_summary() -> pd.DataFrame:
    d1 = pd.read_csv(EXTRA)
    d2 = pd.read_csv(PROC)

    # metric 1: boundary coverage (higher better)
    m1 = (
        d1.groupby("algorithm")["boundary_coverage_final"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    m1["metric"] = "Boundary coverage\n(final Pareto) ↑"
    m1 = m1.rename(columns={"mean": "value_mean", "std": "value_std"})

    # metric 2: spread over optimisation (higher better)
    m2 = (
        d1.groupby("algorithm")["avg_front_spread_over_batches"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    m2["metric"] = "Front spread\n(over batches) ↑"
    m2 = m2.rename(columns={"mean": "value_mean", "std": "value_std"})

    # metric 3: exploration score from selection percentile (higher better)
    # mean_selected_acq_percentile: lower means selecting lower-ranked acq points (more exploratory)
    # convert to exploration score = 1 - percentile
    d2 = d2.copy()
    d2["exploration_score"] = 1.0 - d2["mean_selected_acq_percentile"]
    m3 = (
        d2.groupby("algorithm")["exploration_score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    m3["metric"] = "Exploration score\n(1 - acq percentile) ↑"
    m3 = m3.rename(columns={"mean": "value_mean", "std": "value_std"})

    out = pd.concat([m1, m2, m3], ignore_index=True)
    out["algorithm"] = pd.Categorical(out["algorithm"], categories=ALGO_ORDER, ordered=True)
    metric_order = [
        "Boundary coverage\n(final Pareto) ↑",
        "Front spread\n(over batches) ↑",
        "Exploration score\n(1 - acq percentile) ↑",
    ]
    out["metric"] = pd.Categorical(out["metric"], categories=metric_order, ordered=True)
    out = out.sort_values(["metric", "algorithm"]).reset_index(drop=True)
    return out


def plot_bars(df: pd.DataFrame) -> None:
    metrics = list(df["metric"].cat.categories)
    n_alg = len(ALGO_ORDER)
    x = np.arange(len(metrics))
    width = 0.22
    offsets = np.linspace(-(n_alg - 1) / 2, (n_alg - 1) / 2, n_alg) * width

    fig, ax = plt.subplots(figsize=(8.2, 3.4), constrained_layout=True)

    for i, alg in enumerate(ALGO_ORDER):
        sub = df[df["algorithm"] == alg].set_index("metric")
        vals = [sub.loc[m, "value_mean"] if m in sub.index else np.nan for m in metrics]
        errs = [sub.loc[m, "value_std"] if m in sub.index else np.nan for m in metrics]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            yerr=errs,
            capsize=2,
            error_kw={"linewidth": 0.8},
            color=ALGO_COLORS[alg],
            label=ALGO_LABELS[alg],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("Metric value (higher is better)", fontweight="bold")
    ax.set_ylim(0, None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, borderaxespad=0.0)

    save_fig(fig, HERE / "synthetic_pairwise_mechanism_summary_bars")


def main() -> None:
    df = build_summary()
    df.to_csv(HERE / "synthetic_pairwise_mechanism_summary_bars.csv", index=False)
    plot_bars(df)
    print("Done.")


if __name__ == "__main__":
    main()
