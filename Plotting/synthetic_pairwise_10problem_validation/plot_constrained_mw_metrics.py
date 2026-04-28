"""
Act 11 — Constrained MW (MW3/MW5/MW7) mechanism figures.

Compares:
    EGBO_Novelty_v1   → Novelty Aware EGBO (novel)
  EGBO              → EGBO (standard)
  Traditional_NEHVI → qNEHVI

Input:
  benchmark_pairwise_2problems_20260313/pairwise_all10_process_metrics_per_trial.csv

Outputs:
  - synthetic_pairwise_mw_constraints_stagnation_hv.pdf
  - synthetic_pairwise_mw_constraints_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "benchmark_pairwise_2problems_20260313" / "pairwise_all10_process_metrics_per_trial.csv"

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

PROBLEM_ORDER = ["MW3", "MW5", "MW7"]


def _prepare_long_summary(df: pd.DataFrame, value_col: str, metric_name: str) -> pd.DataFrame:
    by_problem = (
        df.groupby(["problem", "algorithm"])[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    pooled = (
        df.groupby(["algorithm"])[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    pooled["problem"] = "Pooled"

    out = pd.concat([by_problem, pooled], ignore_index=True)
    out["metric"] = metric_name
    return out


def build_summary() -> tuple[pd.DataFrame, dict[str, float]]:
    df = pd.read_csv(DATA)
    df = df[df["problem"].isin(PROBLEM_ORDER)].copy()
    df = df[df["algorithm"].isin(ALGO_ORDER)].copy()

    s_stag = _prepare_long_summary(
        df=df,
        value_col="max_stagnation",
        metric_name="max_stagnation",
    )
    s_hv = _prepare_long_summary(
        df=df,
        value_col="hv_final",
        metric_name="hv_final",
    )

    summary = pd.concat([s_stag, s_hv], ignore_index=True)

    summary["problem"] = pd.Categorical(
        summary["problem"], categories=PROBLEM_ORDER + ["Pooled"], ordered=True
    )
    summary["algorithm"] = pd.Categorical(
        summary["algorithm"], categories=ALGO_ORDER, ordered=True
    )
    summary = summary.sort_values(["metric", "problem", "algorithm"]).reset_index(drop=True)

    # pooled effects for manuscript callouts
    pooled_stag = (
        summary[(summary["metric"] == "max_stagnation") & (summary["problem"] == "Pooled")]
        .set_index("algorithm")["mean"]
    )
    pooled_hv = (
        summary[(summary["metric"] == "hv_final") & (summary["problem"] == "Pooled")]
        .set_index("algorithm")["mean"]
    )

    effects = {
        "stagnation_reduction_vs_egbo_pct": 100.0
        * (1.0 - pooled_stag["EGBO_Novelty_v1"] / pooled_stag["EGBO"]),
        "stagnation_reduction_vs_nehvi_pct": 100.0
        * (1.0 - pooled_stag["EGBO_Novelty_v1"] / pooled_stag["Traditional_NEHVI"]),
        "hv_uplift_vs_egbo_pct": 100.0
        * (
            (pooled_hv["EGBO_Novelty_v1"] - pooled_hv["EGBO"])
            / pooled_hv["EGBO"]
        ),
        "hv_uplift_vs_nehvi_pct": 100.0
        * (
            (pooled_hv["EGBO_Novelty_v1"] - pooled_hv["Traditional_NEHVI"])
            / pooled_hv["Traditional_NEHVI"]
        ),
    }

    return summary, effects


def _plot_grouped_bars(ax: plt.Axes, data: pd.DataFrame, ylabel: str) -> None:
    x_labels = PROBLEM_ORDER + ["Pooled"]
    x = np.arange(len(x_labels))
    width = 0.22
    offsets = np.linspace(-(len(ALGO_ORDER) - 1) / 2, (len(ALGO_ORDER) - 1) / 2, len(ALGO_ORDER)) * width

    for i, alg in enumerate(ALGO_ORDER):
        sub = data[data["algorithm"] == alg].set_index("problem")
        vals = [sub.loc[p, "mean"] if p in sub.index else np.nan for p in x_labels]
        errs = [sub.loc[p, "std"] if p in sub.index else np.nan for p in x_labels]
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
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel, fontweight="bold")


def plot(summary: pd.DataFrame, effects: dict[str, float]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)

    d_stag = summary[summary["metric"] == "max_stagnation"].copy()
    d_hv = summary[summary["metric"] == "hv_final"].copy()

    _plot_grouped_bars(ax1, d_stag, ylabel="Max stagnation batches (lower is better)")
    _plot_grouped_bars(ax2, d_hv, ylabel="Final hypervolume (higher is better)")

    ax1.set_title("Convergence stall behavior")
    ax2.set_title("End-of-run quality")

    ax1.text(
        0.01,
        0.98,
        (
            f"Novelty Aware EGBO vs EGBO:\n"
            f"{effects['stagnation_reduction_vs_egbo_pct']:.1f}% fewer stagnation batches"
        ),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    ax2.text(
        0.01,
        0.98,
        (
            f"Novelty Aware EGBO vs EGBO:\n"
            f"{effects['hv_uplift_vs_egbo_pct']:.1f}% higher pooled final HV"
        ),
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))

    save_fig(fig, HERE / "synthetic_pairwise_mw_constraints_stagnation_hv")


def main() -> None:
    summary, effects = build_summary()

    # attach pooled effect rows in CSV for transparency
    effect_rows = pd.DataFrame(
        [
            {"metric": "effect", "problem": "Pooled", "algorithm": "Adaptive_vs_EGBO", "mean": effects["stagnation_reduction_vs_egbo_pct"], "std": np.nan, "n": np.nan, "notes": "% fewer stagnation batches"},
            {"metric": "effect", "problem": "Pooled", "algorithm": "Adaptive_vs_EGBO", "mean": effects["hv_uplift_vs_egbo_pct"], "std": np.nan, "n": np.nan, "notes": "% final HV uplift"},
            {"metric": "effect", "problem": "Pooled", "algorithm": "Adaptive_vs_qNEHVI", "mean": effects["stagnation_reduction_vs_nehvi_pct"], "std": np.nan, "n": np.nan, "notes": "% fewer stagnation batches"},
            {"metric": "effect", "problem": "Pooled", "algorithm": "Adaptive_vs_qNEHVI", "mean": effects["hv_uplift_vs_nehvi_pct"], "std": np.nan, "n": np.nan, "notes": "% final HV uplift"},
        ]
    )

    out_csv = pd.concat([summary.assign(notes=""), effect_rows], ignore_index=True)
    out_csv.to_csv(HERE / "synthetic_pairwise_mw_constraints_summary.csv", index=False)

    plot(summary, effects)

    print("Pooled effects:")
    for k, v in effects.items():
        print(f"  {k}: {v:.3f}%")
    print("Done.")


if __name__ == "__main__":
    main()
