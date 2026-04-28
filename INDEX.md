# Release Review Bundle Index

This folder contains the **pre-push review package** for the EGBO / EGBO_Novelty release.

## What this bundle includes

- Reproducible runner code for:
  - `EGBO`
  - `EGBO_Novelty` (novelty-aware variant)
- Post-hoc analysis scripts used for manuscript statistics
- Canonical aggregate CSV/PDF outputs
- Release metadata (`manifest.json`, environment files, checklist)

## Folder map

- `novelty_aware_EGBO_v2/`
  - `run_egbo_family_benchmarks.py` — core algorithm implementations (includes `EGBO_Novelty`)
  - `run_realworld_posthoc_benchmarks.py` — real-world/post-hoc runner
  - `README.md`, `QUICKSTART.md` — usage notes

- `run_synthetic_pairwise_egbo_novelty_nehvi.py`
  - synthetic pairwise runner (EGBO vs EGBO_Novelty vs Traditional_NEHVI)

- `benchmark_data_export/`
  - canonical aggregate benchmark exports (comparison, convergence, robustness, etc.)

- `additional_aggregate_data/`
  - `noise_sweep/` — aggregated noise robustness tables (HV/IGD, AUC, convergence)
  - `scaling/` — aggregated dimensional scaling tables (e.g., `scaling_dimension_metrics.csv`, quality summaries)
  - `constraints/` — aggregated feasibility/constraint tables (e.g., `constraints_feasibility_metrics.csv`, boundary-shift summaries)

- `Plotting/realworld_posthoc_validation/`
  - post-hoc real-world stats scripts + final CSV/PDF outputs

- `Plotting/synthetic_pairwise_10problem_validation/`
  - synthetic mechanism/constraint summary plots and CSVs

- `results/aggregate/manifest.json`
  - artifact-level provenance template for release

- `docs/RELEASE_CHECKLIST.md`
  - final pre-tag verification checklist

- `.gitignore`, `requirements.txt`, `environment.yml`
  - reproducibility and packaging support files

## Suggested review order

1. Check release scope in `docs/RELEASE_CHECKLIST.md`
2. Review algorithms/runners in `novelty_aware_EGBO_v2/` and `run_synthetic_pairwise_egbo_novelty_nehvi.py`
3. Verify aggregate outputs in `benchmark_data_export/`, `additional_aggregate_data/`, and `Plotting/realworld_posthoc_validation/`
4. Confirm provenance fields in `results/aggregate/manifest.json`
5. Approve bundle for push/tag

## Quick local validation commands

From repository root:

```bash
python run_synthetic_pairwise_egbo_novelty_nehvi.py --quick
python Plotting/realworld_posthoc_validation/compute_realworld_per_dataset_pvalues.py
python Plotting/realworld_posthoc_validation/compute_realworld_friedman_posthoc.py
```

If all outputs match the committed aggregate files, the bundle is ready for publication.
