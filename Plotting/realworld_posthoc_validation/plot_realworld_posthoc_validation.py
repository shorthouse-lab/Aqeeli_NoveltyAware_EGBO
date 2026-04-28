"""
Act 10 — Post hoc real-world validation (SDL5, Suzuki, ADA coatings, GDSC CRC5)

Compares a merged display slot:
- EGBO_Novelty_v1 on the original three datasets  -> Novelty Aware EGBO
- EGBO_Novelty on GDSC CRC5                      -> Novelty Aware EGBO
- EGBO_Paper_2024 / EGBO                          -> EGBO
- Traditional_NEHVI                               -> qNEHVI

Input defaults:
    --data-root: benchmark_paper_vs_adaptive_v5_matched_vs_nehvi_realworld_q4_20260312/
    --phase8-root: benchmark_results_phase8_gdsc_realworld/
    --phase8-analysis-dir: benchmark_results_phase8_gdsc_realworld/analysis_phase8_gdsc/

Outputs:
- realworld_posthoc_realworld_convergence.pdf
- realworld_posthoc_realworld_final_hv_boxplot.pdf
- realworld_posthoc_realworld_novelty_gain_vs_baselines.pdf
- realworld_posthoc_realworld_final_igd_boxplot.pdf
- realworld_posthoc_realworld_novelty_igd_reduction_vs_baselines.pdf
- realworld_posthoc_realworld_convergence.csv
- realworld_posthoc_realworld_final_hv.csv
- realworld_posthoc_realworld_summary.csv
- realworld_posthoc_realworld_final_igd.csv
- realworld_posthoc_realworld_igd_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style, save_fig, FIG_DOUBLE

apply_style()

LEGACY_PROBLEMS = {
    "SDL5": "SDL5",
    "REIZMAN_SUZUKI_CASE1": "Suzuki",
    "ADA_COATINGS_2022_09": "ADA coatings",
}

PHASE8_PROBLEMS = {
    "GDSC_CRC5": "GDSC CRC5",
}

PROBLEMS = {**LEGACY_PROBLEMS, **PHASE8_PROBLEMS}
PROBLEM_ORDER = ["SDL5", "Suzuki", "ADA coatings", "GDSC CRC5"]

DISPLAY_ALGORITHMS = {
    "adaptive_slot": "Novelty Aware EGBO",
    "egbo": "EGBO",
    "nehvi": "qNEHVI",
}

DISPLAY_ALGORITHM_ORDER = ["adaptive_slot", "egbo", "nehvi"]

SOURCE_ALGO_TO_SLOT = {
    "EGBO_Novelty_v1": "adaptive_slot",
    "EGBO_Novelty": "adaptive_slot",
    "EGBO_Paper_2024": "egbo",
    "EGBO": "egbo",
    "Traditional_NEHVI": "nehvi",
}

ALGO_COLORS = {
    "adaptive_slot": "#8E24AA",   # purple
    "egbo": "#2E7D32",            # green
    "nehvi": "#1565C0",           # blue
}


def _present_problem_order(df: pd.DataFrame) -> list[str]:
    present = set(df["problem"].unique())
    return [problem for problem in PROBLEM_ORDER if problem in present]


def load_hv_curves(data_root: Path, phase8_root: Path | None = None) -> pd.DataFrame:
    rows = []
    legacy_algorithms = ["EGBO_Novelty_v1", "EGBO_Paper_2024", "Traditional_NEHVI"]
    for problem_key, problem_name in LEGACY_PROBLEMS.items():
        for algo_key in legacy_algorithms:
            algo_dir = data_root / problem_key / algo_key
            if not algo_dir.exists():
                print(f"Warning: missing folder {algo_dir}")
                continue

            for hv_file in sorted(algo_dir.glob("trial_*_hv.csv")):
                trial = int(hv_file.stem.split("_")[1])
                hv = np.atleast_1d(np.loadtxt(hv_file, delimiter=","))
                for b, v in enumerate(hv, start=1):
                    rows.append(
                        {
                            "problem_key": problem_key,
                            "problem": problem_name,
                            "algorithm_key": SOURCE_ALGO_TO_SLOT[algo_key],
                            "algorithm": DISPLAY_ALGORITHMS[SOURCE_ALGO_TO_SLOT[algo_key]],
                            "source_algorithm_key": algo_key,
                            "trial": trial,
                            "batch": b,
                            "hv": float(v),
                        }
                    )

    if phase8_root is not None:
        phase8_algorithms = ["EGBO_Novelty", "EGBO", "Traditional_NEHVI"]
        problem_key = "GDSC_CRC5"
        problem_name = PHASE8_PROBLEMS[problem_key]
        for algo_key in phase8_algorithms:
            algo_dir = phase8_root / problem_key / algo_key
            if not algo_dir.exists():
                print(f"Warning: missing folder {algo_dir}")
                continue

            for hv_file in sorted(algo_dir.glob("trial_*_hv.csv")):
                trial = int(hv_file.stem.split("_")[1])
                hv = np.atleast_1d(np.loadtxt(hv_file, delimiter=","))
                for b, v in enumerate(hv, start=1):
                    rows.append(
                        {
                            "problem_key": problem_key,
                            "problem": problem_name,
                            "algorithm_key": SOURCE_ALGO_TO_SLOT[algo_key],
                            "algorithm": DISPLAY_ALGORITHMS[SOURCE_ALGO_TO_SLOT[algo_key]],
                            "source_algorithm_key": algo_key,
                            "trial": trial,
                            "batch": b,
                            "hv": float(v),
                        }
                    )

    if not rows:
        raise RuntimeError(f"No trial_*_hv.csv files found under {data_root}")

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conv = (
        df.groupby(["problem", "problem_key", "algorithm", "algorithm_key", "batch"], as_index=False)
        .agg(hv_mean=("hv", "mean"), hv_std=("hv", "std"), n=("hv", "size"))
    )
    conv["hv_std"] = conv["hv_std"].fillna(0.0)

    final_df = (
        df.sort_values("batch")
        .groupby(["problem", "problem_key", "algorithm", "algorithm_key", "trial"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    summary = (
        final_df.groupby(["problem", "problem_key", "algorithm", "algorithm_key"], as_index=False)
        .agg(
            n_trials=("hv", "size"),
            final_hv_mean=("hv", "mean"),
            final_hv_std=("hv", "std"),
            final_hv_median=("hv", "median"),
            final_hv_min=("hv", "min"),
            final_hv_max=("hv", "max"),
        )
    )
    summary["final_hv_std"] = summary["final_hv_std"].fillna(0.0)

    return conv, final_df, summary


def _normalize_points(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    scale = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    return (X - lo) / scale


def _igd(approx: np.ndarray, ref: np.ndarray) -> float:
    # IGD(ref -> approx): mean distance from each reference point to nearest approx point
    # Uses explicit broadcasting to avoid scipy dependency.
    d = approx[None, :, :] - ref[:, None, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    return float(np.mean(np.min(dist, axis=1)))


def compute_empirical_igd(data_root: Path, phase8_analysis_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute empirical final IGD from per-trial Pareto objective files.

    Reference set (per legacy problem): union of all trial final Pareto sets across
    Adaptive-slot, EGBO, qNEHVI from the current run.
    """
    sets = {}
    legacy_algorithms = ["EGBO_Novelty_v1", "EGBO_Paper_2024", "Traditional_NEHVI"]
    for problem_key in LEGACY_PROBLEMS:
        sets[problem_key] = {a: {} for a in legacy_algorithms}
        for algo_key in legacy_algorithms:
            algo_dir = data_root / problem_key / algo_key
            if not algo_dir.exists():
                continue
            for pf_file in sorted(algo_dir.glob("trial_*_pareto_objectives_batch_*.csv")):
                trial = int(pf_file.stem.split("_")[1])
                arr = np.atleast_2d(np.loadtxt(pf_file, delimiter=","))
                if arr.size == 0:
                    continue
                sets[problem_key][algo_key][trial] = arr.astype(float)

    rows = []
    for problem_key in LEGACY_PROBLEMS:
        union = []
        for algo_key in legacy_algorithms:
            union.extend(list(sets[problem_key][algo_key].values()))
        if not union:
            continue

        ref = np.vstack(union)
        lo = np.min(ref, axis=0)
        hi = np.max(ref, axis=0)
        ref_n = _normalize_points(ref, lo, hi)

        for algo_key in legacy_algorithms:
            for trial, approx in sets[problem_key][algo_key].items():
                approx_n = _normalize_points(approx, lo, hi)
                igd = _igd(approx_n, ref_n)
                rows.append(
                    {
                        "problem_key": problem_key,
                        "problem": LEGACY_PROBLEMS[problem_key],
                        "algorithm_key": SOURCE_ALGO_TO_SLOT[algo_key],
                        "algorithm": DISPLAY_ALGORITHMS[SOURCE_ALGO_TO_SLOT[algo_key]],
                        "source_algorithm_key": algo_key,
                        "trial": int(trial),
                        "final_igd": float(igd),
                    }
                )

    if phase8_analysis_dir is not None:
        phase8_trial_metrics = phase8_analysis_dir / "phase8_gdsc_trial_metrics.csv"
        if phase8_trial_metrics.exists():
            phase8_df = pd.read_csv(phase8_trial_metrics)
            phase8_df = phase8_df[phase8_df["algorithm"].isin(["EGBO_Novelty", "EGBO", "Traditional_NEHVI"])].copy()
            phase8_df["problem_key"] = "GDSC_CRC5"
            phase8_df["problem"] = PHASE8_PROBLEMS["GDSC_CRC5"]
            phase8_df["source_algorithm_key"] = phase8_df["algorithm"]
            phase8_df["algorithm_key"] = phase8_df["algorithm"].map(SOURCE_ALGO_TO_SLOT)
            phase8_df["algorithm"] = phase8_df["algorithm_key"].map(DISPLAY_ALGORITHMS)
            rows.extend(
                phase8_df[["problem_key", "problem", "algorithm_key", "algorithm", "source_algorithm_key", "trial", "final_igd"]]
                .to_dict(orient="records")
            )

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    igd_df = pd.DataFrame(rows)
    igd_summary = (
        igd_df.groupby(["problem", "problem_key", "algorithm", "algorithm_key"], as_index=False)
        .agg(
            n_trials=("final_igd", "size"),
            final_igd_mean=("final_igd", "mean"),
            final_igd_std=("final_igd", "std"),
            final_igd_median=("final_igd", "median"),
            final_igd_min=("final_igd", "min"),
            final_igd_max=("final_igd", "max"),
        )
    )
    igd_summary["final_igd_std"] = igd_summary["final_igd_std"].fillna(0.0)
    return igd_df, igd_summary


def plot_convergence(conv: pd.DataFrame, out_dir: Path) -> None:
    ordered_problems = _present_problem_order(conv)
    ordered_algos = DISPLAY_ALGORITHM_ORDER

    fig, axes = plt.subplots(1, len(ordered_problems), figsize=(FIG_DOUBLE[0] * len(ordered_problems) / 3.0, FIG_DOUBLE[1]), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, prob in zip(axes, ordered_problems):
        subp = conv[conv["problem"] == prob]
        for algo_key in ordered_algos:
            sub = subp[subp["algorithm_key"] == algo_key].sort_values("batch")
            if sub.empty:
                continue
            ax.plot(
                sub["batch"],
                sub["hv_mean"],
                marker="o",
                markersize=3,
                color=ALGO_COLORS[algo_key],
                label=DISPLAY_ALGORITHMS[algo_key],
            )
            ax.fill_between(
                sub["batch"],
                sub["hv_mean"] - sub["hv_std"],
                sub["hv_mean"] + sub["hv_std"],
                color=ALGO_COLORS[algo_key],
                alpha=0.15,
            )

        ax.set_title(prob, fontweight="bold")
        ax.set_xlabel("Batch")

    axes[0].set_ylabel("Hypervolume", fontweight="bold", labelpad=8)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3)

    save_fig(fig, out_dir / "realworld_posthoc_realworld_convergence")


def plot_final_boxplot(final_df: pd.DataFrame, out_dir: Path) -> None:
    ordered_problems = _present_problem_order(final_df)
    ordered_algos = DISPLAY_ALGORITHM_ORDER

    fig, axes = plt.subplots(1, len(ordered_problems), figsize=(FIG_DOUBLE[0] * len(ordered_problems) / 3.0, FIG_DOUBLE[1]), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, prob in zip(axes, ordered_problems):
        subp = final_df[final_df["problem"] == prob]

        data = []
        labels = []
        colors = []
        for algo_key in ordered_algos:
            vals = subp[subp["algorithm_key"] == algo_key]["hv"].to_numpy()
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(DISPLAY_ALGORITHMS[algo_key])
            colors.append(ALGO_COLORS[algo_key])

        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.0},
        )

        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        ax.set_title(prob, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)

    axes[0].set_ylabel("Final hypervolume", fontweight="bold", labelpad=8)
    save_fig(fig, out_dir / "realworld_posthoc_realworld_final_hv_boxplot")


def plot_novelty_gain(summary: pd.DataFrame, out_dir: Path) -> None:
    probs = _present_problem_order(summary)

    ref = summary.pivot(index="problem", columns="algorithm", values="final_hv_mean")

    novelty = ref.get("Novelty Aware EGBO")
    egbo = ref.get("EGBO")
    nehvi = ref.get("qNEHVI")

    gain_vs_egbo = 100.0 * (novelty - egbo) / egbo
    gain_vs_nehvi = 100.0 * (novelty - nehvi) / nehvi

    x = np.arange(len(probs))
    w = 0.38

    fig, ax = plt.subplots(figsize=(FIG_DOUBLE[0], FIG_DOUBLE[1] * 0.95))
    ax.bar(x - w / 2, [gain_vs_egbo[p] for p in probs], w, label="vs EGBO", color="#43A047")
    ax.bar(x + w / 2, [gain_vs_nehvi[p] for p in probs], w, label="vs qNEHVI", color="#1E88E5")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(probs, fontweight="bold")
    ax.set_ylabel("Novelty Aware EGBO gain in final HV (%)", fontweight="bold", labelpad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    save_fig(fig, out_dir / "realworld_posthoc_realworld_novelty_gain_vs_baselines")


def plot_igd_boxplot(igd_df: pd.DataFrame, out_dir: Path) -> None:
    if igd_df.empty:
        return

    ordered_problems = _present_problem_order(igd_df)
    ordered_algos = DISPLAY_ALGORITHM_ORDER

    fig, axes = plt.subplots(1, len(ordered_problems), figsize=(FIG_DOUBLE[0] * len(ordered_problems) / 3.0, FIG_DOUBLE[1]), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, prob in zip(axes, ordered_problems):
        subp = igd_df[igd_df["problem"] == prob]
        data = []
        labels = []
        colors = []
        for algo_key in ordered_algos:
            vals = subp[subp["algorithm_key"] == algo_key]["final_igd"].to_numpy()
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(DISPLAY_ALGORITHMS[algo_key])
            colors.append(ALGO_COLORS[algo_key])

        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.0},
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        ax.set_title(prob, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)

    axes[0].set_ylabel("Final IGD (lower is better)", fontweight="bold", labelpad=8)
    save_fig(fig, out_dir / "realworld_posthoc_realworld_final_igd_boxplot")


def plot_novelty_igd_reduction(igd_summary: pd.DataFrame, out_dir: Path) -> None:
    if igd_summary.empty:
        return

    probs = _present_problem_order(igd_summary)
    ref = igd_summary.pivot(index="problem", columns="algorithm", values="final_igd_mean")

    novelty = ref.get("Novelty Aware EGBO")
    egbo = ref.get("EGBO")
    nehvi = ref.get("qNEHVI")

    # Positive means Novelty Aware EGBO has lower IGD (better)
    red_vs_egbo = 100.0 * (egbo - novelty) / egbo
    red_vs_nehvi = 100.0 * (nehvi - novelty) / nehvi

    x = np.arange(len(probs))
    w = 0.38

    fig, ax = plt.subplots(figsize=(FIG_DOUBLE[0], FIG_DOUBLE[1] * 0.95))
    ax.bar(x - w / 2, [red_vs_egbo[p] for p in probs], w, label="vs EGBO", color="#43A047")
    ax.bar(x + w / 2, [red_vs_nehvi[p] for p in probs], w, label="vs qNEHVI", color="#1E88E5")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(probs, fontweight="bold")
    ax.set_ylabel("Novelty Aware EGBO IGD reduction (%)", fontweight="bold", labelpad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    save_fig(fig, out_dir / "realworld_posthoc_realworld_novelty_igd_reduction_vs_baselines")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post hoc real-world validation plots")
    p.add_argument(
        "--data-root",
        type=str,
        default="benchmark_paper_vs_adaptive_v5_matched_vs_nehvi_realworld_q4_20260312",
        help="Benchmark output root containing SDL5/REIZMAN_SUZUKI_CASE1/ADA_COATINGS_2022_09",
    )
    p.add_argument(
        "--phase8-root",
        type=str,
        default="benchmark_results_phase8_gdsc_realworld",
        help="Optional Phase 8 benchmark root containing GDSC_CRC5/EGBO_Novelty, EGBO, Traditional_NEHVI",
    )
    p.add_argument(
        "--phase8-analysis-dir",
        type=str,
        default="benchmark_results_phase8_gdsc_realworld/analysis_phase8_gdsc",
        help="Optional Phase 8 analysis directory containing phase8_gdsc_trial_metrics.csv",
    )
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    phase8_root = Path(args.phase8_root).resolve()
    phase8_analysis_dir = Path(args.phase8_analysis_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_hv_curves(data_root, phase8_root=phase8_root if phase8_root.exists() else None)
        conv, final_df, summary = build_summary(df)
        igd_df, igd_summary = compute_empirical_igd(
            data_root,
            phase8_analysis_dir=phase8_analysis_dir if phase8_analysis_dir.exists() else None,
        )

        conv.to_csv(out_dir / "realworld_posthoc_realworld_convergence.csv", index=False)
        final_df.to_csv(out_dir / "realworld_posthoc_realworld_final_hv.csv", index=False)
        summary.to_csv(out_dir / "realworld_posthoc_realworld_summary.csv", index=False)
        if not igd_df.empty:
            igd_df.to_csv(out_dir / "realworld_posthoc_realworld_final_igd.csv", index=False)
            igd_summary.to_csv(out_dir / "realworld_posthoc_realworld_igd_summary.csv", index=False)
    except Exception as exc:
        print(f"Warning: could not rebuild from raw benchmark folders ({exc}).")
        print("Falling back to existing CSV tables in output directory for figure regeneration.")
        conv = pd.read_csv(out_dir / "realworld_posthoc_realworld_convergence.csv")
        final_df = pd.read_csv(out_dir / "realworld_posthoc_realworld_final_hv.csv")
        summary = pd.read_csv(out_dir / "realworld_posthoc_realworld_summary.csv")
        igd_file = out_dir / "realworld_posthoc_realworld_final_igd.csv"
        igd_summary_file = out_dir / "realworld_posthoc_realworld_igd_summary.csv"
        igd_df = pd.read_csv(igd_file) if igd_file.exists() else pd.DataFrame()
        igd_summary = pd.read_csv(igd_summary_file) if igd_summary_file.exists() else pd.DataFrame()

    plot_convergence(conv, out_dir)
    plot_final_boxplot(final_df, out_dir)
    plot_novelty_gain(summary, out_dir)
    plot_igd_boxplot(igd_df, out_dir)
    plot_novelty_igd_reduction(igd_summary, out_dir)

    print(f"Saved figures and tables to {out_dir}")


if __name__ == "__main__":
    main()
