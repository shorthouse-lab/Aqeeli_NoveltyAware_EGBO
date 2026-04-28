"""
Act 11 — Pairwise synthetic benchmark (10 problems × 3 algorithms)

Compares:
    EGBO_Novelty_v1   → "Novelty Aware EGBO" (purple #8E24AA)
  EGBO              → "EGBO"         (green  #2E7D32)
  Traditional_NEHVI → "qNEHVI"       (blue   #1565C0)

Input:
  benchmark_pairwise_2problems_20260313/pairwise_all10_hv_igd_summary.csv

Outputs (in this directory):
  synthetic_pairwise_pairwise_hv_bars.pdf   — HV / Best HV per problem
  synthetic_pairwise_pairwise_igd_bars.pdf  — Best IGD / IGD per problem  (higher = closer to best)
    synthetic_pairwise_pairwise_igd_raw_bars.pdf — Raw IGD per problem (lower = better)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────────
HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parents[1]
OUT    = HERE
DATA   = ROOT / "benchmark_pairwise_2problems_20260313" / "pairwise_all10_hv_igd_summary.csv"

sys.path.insert(0, str(HERE.parent))
from plot_style import apply_style, save_fig

apply_style()

# ── algorithm metadata ──────────────────────────────────────────────────────
ALGO_ORDER = ["EGBO_Novelty_v1", "EGBO", "Traditional_NEHVI"]
ALGO_LABELS = {
    "EGBO_Novelty_v1":   "Novelty Aware EGBO",
    "EGBO":              "EGBO",
    "Traditional_NEHVI": "qNEHVI",
}
ALGO_COLORS = {
    "EGBO_Novelty_v1":   "#8E24AA",   # purple  (matches act10)
    "EGBO":              "#2E7D32",   # green   (matches act10 / plot_style)
    "Traditional_NEHVI": "#1565C0",   # blue    (matches act10 / plot_style)
}

# ── problem metadata ────────────────────────────────────────────────────────
# (decision vars, objectives) as used in the pairwise run
PROBLEM_DIMS = {
    "DTLZ1":     (8, 3),
    "DTLZ2_5obj":(8, 5),
    "DTLZ3":     (8, 3),
    "MW3":       (3, 3),
    "MW5":       (2, 2),
    "MW7":       (2, 2),
    "ZDT1":      (8, 2),
    "ZDT2":      (8, 2),
    "ZDT3":      (8, 2),
    "ZDT4":      (8, 2),
}

# Desired left-to-right order on x-axis: DTLZ, MW, ZDT
PROBLEM_ORDER = [
    "DTLZ1", "DTLZ2_5obj", "DTLZ3",
    "MW3",   "MW5",         "MW7",
    "ZDT1",  "ZDT2",        "ZDT3",  "ZDT4",
]

def problem_label(p: str) -> str:
    base = p.replace("_5obj", "").replace("_", " ")
    nv, no = PROBLEM_DIMS.get(p, ("?", "?"))
    return f"{base}\n({nv}D, {no}obj)"


# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
df = df[df["algorithm"].isin(ALGO_ORDER)].copy()

# Ensure problems are in our desired order
df["problem"] = pd.Categorical(df["problem"], categories=PROBLEM_ORDER, ordered=True)
df = df.sort_values(["problem", "algorithm"])
problems = PROBLEM_ORDER  # fixed x order


# ── helpers ──────────────────────────────────────────────────────────────────
def bar_params(n_alg: int) -> tuple[float, np.ndarray]:
    """Return (bar_width, per-algorithm x offsets) for n_alg grouped bars."""
    width = 0.22
    offsets = np.linspace(-(n_alg - 1) / 2, (n_alg - 1) / 2, n_alg) * width
    return width, offsets


def annotate_winner(ax, x_pos: float, y_max: float, symbol: str = "★",
                    color: str = "0.4") -> None:
    ax.text(x_pos, y_max + 0.01, symbol, ha="center", va="bottom",
            fontsize=7, color=color)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 1 — HV / Best HV per problem
# ══════════════════════════════════════════════════════════════════════════════
def plot_hv_bars(df: pd.DataFrame) -> None:
    agg = df[["problem", "algorithm", "hv_mean", "hv_std"]].copy()

    # normalise per problem: rel = hv / max(hv across algos)
    max_hv = agg.groupby("problem")["hv_mean"].transform("max")
    # For problems where every algorithm has hv=0 (ZDT4), keep bars at 0
    agg["rel"]     = np.where(max_hv > 0, agg["hv_mean"] / max_hv, 0.0)
    agg["rel_std"] = np.where(max_hv > 0, agg["hv_std"]  / max_hv, 0.0)

    n_alg   = len(ALGO_ORDER)
    x       = np.arange(len(problems))
    width, offsets = bar_params(n_alg)

    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)

    for i, alg in enumerate(ALGO_ORDER):
        sub  = agg[agg["algorithm"] == alg].set_index("problem")
        vals = [sub.loc[p, "rel"]     if p in sub.index else np.nan for p in problems]
        errs = [sub.loc[p, "rel_std"] if p in sub.index else np.nan for p in problems]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            yerr=errs,
            label=ALGO_LABELS[alg],
            color=ALGO_COLORS[alg],
            capsize=2,
            error_kw={"linewidth": 0.8},
        )

    # dashed ceiling at 1.0
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels([problem_label(p) for p in problems],
                       rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("HV / Best HV (per problem)", fontweight="bold", labelpad=8)
    # prevent error bars from clipping at the top
    ymax = float(np.nanmax(agg["rel"] + agg["rel_std"]))
    ax.set_ylim(0, max(1.10, ymax * 1.08))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, borderaxespad=0.0)

    save_fig(fig, OUT / "synthetic_pairwise_pairwise_hv_bars")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2 — IGD / Best IGD per problem  (lower = closer to best = better)
# ══════════════════════════════════════════════════════════════════════════════
def plot_igd_bars(df: pd.DataFrame) -> None:
    agg = df[["problem", "algorithm", "igd_mean", "igd_std"]].copy()

    # scaled IGD: algo_igd / best_igd  (winner=1, worse>1, lower is better)
    min_igd = agg.groupby("problem")["igd_mean"].transform("min")
    agg["rel"]     = agg["igd_mean"] / min_igd
    # error propagation for ratio algo/min: σ ≈ igd_std / min_igd
    agg["rel_std"] = agg["igd_std"] / min_igd

    n_alg   = len(ALGO_ORDER)
    x       = np.arange(len(problems))
    width, offsets = bar_params(n_alg)

    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)

    for i, alg in enumerate(ALGO_ORDER):
        sub  = agg[agg["algorithm"] == alg].set_index("problem")
        vals = [sub.loc[p, "rel"]     if p in sub.index else np.nan for p in problems]
        errs = [sub.loc[p, "rel_std"] if p in sub.index else np.nan for p in problems]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            yerr=errs,
            label=ALGO_LABELS[alg],
            color=ALGO_COLORS[alg],
            capsize=2,
            error_kw={"linewidth": 0.8},
        )

    # dashed floor at 1.0 (= best possible)
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels([problem_label(p) for p in problems],
                       rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Scaled IGD / best IGD (lower is better)", fontweight="bold", labelpad=8)
    ymax = float(np.nanmax(agg["rel"] + agg["rel_std"]))
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 2.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, borderaxespad=0.0)

    save_fig(fig, OUT / "synthetic_pairwise_pairwise_igd_bars")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 3 — Raw IGD per problem  (lower = better)
# ══════════════════════════════════════════════════════════════════════════════
def plot_raw_igd_bars(df: pd.DataFrame) -> None:
    agg = df[["problem", "algorithm", "igd_mean", "igd_std"]].copy()

    n_alg = len(ALGO_ORDER)
    x = np.arange(len(problems))
    width, offsets = bar_params(n_alg)

    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)

    for i, alg in enumerate(ALGO_ORDER):
        sub = agg[agg["algorithm"] == alg].set_index("problem")
        vals = [sub.loc[p, "igd_mean"] if p in sub.index else np.nan for p in problems]
        errs = [sub.loc[p, "igd_std"] if p in sub.index else np.nan for p in problems]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            yerr=errs,
            label=ALGO_LABELS[alg],
            color=ALGO_COLORS[alg],
            capsize=2,
            error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([problem_label(p) for p in problems],
                       rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Raw IGD (lower is better)", fontweight="bold", labelpad=8)
    ymax = float(np.nanmax(agg["igd_mean"] + agg["igd_std"]))
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, borderaxespad=0.0)

    save_fig(fig, OUT / "synthetic_pairwise_pairwise_igd_raw_bars")


# ── run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading: {DATA}")
    print(f"Problems: {problems}")
    print(f"Algorithms: {ALGO_ORDER}\n")

    plot_hv_bars(df)
    plot_igd_bars(df)
    plot_raw_igd_bars(df)

    print("\nDone. Outputs in:", OUT)
