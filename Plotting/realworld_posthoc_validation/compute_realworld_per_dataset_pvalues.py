from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent

ALGORITHMS = ["Novelty Aware EGBO", "EGBO", "qNEHVI"]
DATASETS = ["Suzuki", "ADA coatings", "GDSC CRC5", "SDL5"]


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    prev = 0.0
    for i, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - i))
        val = max(val, prev)
        out[idx] = val
        prev = val
    return out


def compute(metric_file: str, value_col: str, higher_is_better: bool, metric_name: str) -> pd.DataFrame:
    df = pd.read_csv(BASE / metric_file)
    df = df[df["problem"].isin(DATASETS) & df["algorithm"].isin(ALGORITHMS)].copy()

    rows = []
    for ds in DATASETS:
        d = df[df["problem"] == ds]
        pivot = d.pivot(index="trial", columns="algorithm", values=value_col)
        pivot = pivot[ALGORITHMS].dropna()

        pvals = []
        tmp = []
        for a, b in combinations(ALGORITHMS, 2):
            x = pivot[a].to_numpy()
            y = pivot[b].to_numpy()
            res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
            p = float(res.pvalue)
            pvals.append(p)
            mean_a = float(x.mean())
            mean_b = float(y.mean())
            if higher_is_better:
                favored = a if mean_a > mean_b else b
            else:
                favored = a if mean_a < mean_b else b
            tmp.append({
                "dataset": ds,
                "metric": metric_name,
                "algo_a": a,
                "algo_b": b,
                "n": int(len(x)),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "wilcoxon_W": float(res.statistic),
                "p_value": p,
                "favored_by_mean": favored,
            })

        p_holm = holm_adjust(pvals)
        for row, ph in zip(tmp, p_holm):
            row["p_holm"] = ph
            row["significant_0_05"] = bool(ph < 0.05)
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    hv = compute("realworld_posthoc_realworld_final_hv.csv", "hv", True, "HV")
    igd = compute("realworld_posthoc_realworld_final_igd.csv", "final_igd", False, "IGD")

    hv.to_csv(BASE / "realworld_posthoc_per_dataset_pairwise_hv.csv", index=False)
    igd.to_csv(BASE / "realworld_posthoc_per_dataset_pairwise_igd.csv", index=False)

    print("Wrote:")
    print(BASE / "realworld_posthoc_per_dataset_pairwise_hv.csv")
    print(BASE / "realworld_posthoc_per_dataset_pairwise_igd.csv")


if __name__ == "__main__":
    main()
