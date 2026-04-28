"""
plot_style.py — shared style configuration for all benchmarking plots.

Import at the top of every act script:
    from plot_style import (
        apply_style, save_fig,
        ALGO_COLORS, ALGO_LABELS,
        GEN_COLORS, GEN_LABELS,
        RULE_COLORS, RULE_LABELS,
        FIG_SINGLE, FIG_DOUBLE, FIG_SQUARE,
    )
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Figure sizes (inches) ────────────────────────────────────────────────────
# Single-column journal width
FIG_SINGLE = (3.5, 2.625)
# Double-column journal width (same height)
FIG_DOUBLE = (7.0, 2.625)
# Square (heatmaps, etc.)
FIG_SQUARE = (3.5, 3.5)
# Wide square
FIG_WIDE_SQUARE = (7.0, 5.25)

# ── Algorithm colours ────────────────────────────────────────────────────────
ALGO_COLORS = {
    "EGBO":              "#2E7D32",   # forest green
    "Traditional_NEHVI": "#1565C0",   # steel blue
    "NEHVI_SMS":         "#E65100",   # amber
    "NEHVI_AGE2":        "#6A1B9A",   # plum
    "Feryal":            "#00695C",   # teal
    "Ikhlas":            "#C62828",   # rose
    "Karima":            "#283593",   # indigo
}

# Display names for legends / axis labels
ALGO_LABELS = {
    "EGBO":              "qNEHVI + U-NSGA-III",
    "Traditional_NEHVI": "qNEHVI",
    "NEHVI_SMS":         "qNEHVI + SMS-EMOA",
    "NEHVI_AGE2":        "qNEHVI + AGE-MOEA-II",
    "Feryal":            "Feryal (3-gen)",
    "Ikhlas":            "Ikhlas (3-gen)",
    "Karima":            "Karima (4-gen)",
}

# ── Generator colours (Act 4) ────────────────────────────────────────────────
GEN_COLORS = {
    "qnehvi": "#BDBDBD",   # light grey  (exploitation)
    "nsga3":  "#2E7D32",   # green
    "sms":    "#E65100",   # amber
    "age2":   "#6A1B9A",   # plum
}

GEN_LABELS = {
    "qnehvi": "qNEHVI",
    "nsga3":  "NSGA-III",
    "sms":    "SMS-EMOA",
    "age2":   "AGE-MOEA-II",
}

# ── Selection-rule colours (Act 5) ───────────────────────────────────────────
RULE_COLORS = {
    "Acquisition":          "#2E7D32",   # green  (best)
    "HV-Contribution":      "#1565C0",   # blue
    "Feasibility-First":    "#E65100",   # amber
    "Random-Nondominated":  "#C62828",   # rose   (worst)
}

RULE_LABELS = {
    "Acquisition":          "Acquisition (proposed)",
    "HV-Contribution":      "HV Contribution",
    "Feasibility-First":    "Feasibility-First",
    "Random-Nondominated":  "Random Non-dominated",
}

# ── Relative HV helper ───────────────────────────────────────────────────────
def compute_relative_hv(agg, ref_algo, algo_col="algorithm",
                        mean_col="mean", std_col="std"):
    """
    Compute mean HV as % relative to ref_algo, matched on all other columns.
    Adds 'rel_hv' (% change from reference) and 'rel_hv_std' (propagated
    uncertainty) columns to the returned DataFrame.
    """
    key_cols = [
        c for c in agg.columns
        if c not in [algo_col, mean_col, std_col, "rel_hv", "rel_hv_std"]
    ]
    ref = (
        agg[agg[algo_col] == ref_algo]
        .rename(columns={mean_col: "_ref_mean", std_col: "_ref_std"})
        [key_cols + ["_ref_mean", "_ref_std"]]
    )
    result = agg.merge(ref, on=key_cols, how="left")
    result["rel_hv"] = (
        100.0 * (result[mean_col] - result["_ref_mean"]) / result["_ref_mean"]
    )
    result["rel_hv_std"] = 100.0 * np.sqrt(
        (result[std_col] / result["_ref_mean"]) ** 2
        + (result[mean_col] * result["_ref_std"] / result["_ref_mean"] ** 2) ** 2
    )
    return result.drop(columns=["_ref_mean", "_ref_std"])


# ── Global rcParams ──────────────────────────────────────────────────────────
def apply_style() -> None:
    """Call once at the top of each plot script."""
    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "figure.dpi":              150,
            "savefig.dpi":             300,
            "figure.constrained_layout.use": True,
            "font.family":             "sans-serif",
            "font.size":               9,
            "axes.labelsize":          10,
            "axes.titlesize":          10,
            "axes.titleweight":        "normal",
            "xtick.labelsize":         8,
            "ytick.labelsize":         8,
            "lines.linewidth":         1.8,
            "lines.markersize":        5,
            "patch.linewidth":         0.6,
            "legend.fontsize":         8,
            "legend.frameon":          False,
            "legend.borderpad":        0.4,
            "legend.labelspacing":     0.3,
        },
    )


# ── Save helper ──────────────────────────────────────────────────────────────
def save_fig(fig: plt.Figure, path: str | Path, despine: bool = True) -> None:
    """
    Finalise and save a figure as PDF.

    - Calls sns.despine() on every axes.
    - Ensures all legends are frameless.
    - Saves as PDF only (vector, suitable for journal submission).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if despine:
        for ax in fig.axes:
            sns.despine(ax=ax)

    # Remove legend frames on every axes
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(False)

    # PDF only — no PNG
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {pdf_path}")
    plt.close(fig)
