# Release Checklist (Runners + Aggregate Data)

Target tag: `v1.0.0-paper`

## Scope
- [ ] Include runner code for `EGBO` and `EGBO_Novelty`
- [ ] Include aggregate outputs only (no bulky raw trial folders)
- [ ] Include analysis scripts needed to regenerate statistics/figures

## Required code files
- [ ] `novelty_aware_EGBO_v2/run_egbo_family_benchmarks.py`
- [ ] `run_synthetic_pairwise_egbo_novelty_nehvi.py`
- [ ] `novelty_aware_EGBO_v2/run_realworld_posthoc_benchmarks.py`
- [ ] `Plotting/realworld_posthoc_validation/compute_realworld_friedman_posthoc.py`
- [ ] `Plotting/realworld_posthoc_validation/compute_realworld_per_dataset_pvalues.py`
- [ ] `Plotting/realworld_posthoc_validation/plot_realworld_posthoc_validation.py`

## Required aggregate outputs
- [ ] `benchmark_data_export/algorithm_comparison.csv`
- [ ] `benchmark_data_export/algorithm_comparison_summary.csv`
- [ ] `benchmark_data_export/hv_convergence_all_runs.csv`
- [ ] `benchmark_data_export/generator_contribution_per_trial.csv`
- [ ] `Plotting/realworld_posthoc_validation/realworld_posthoc_per_dataset_pairwise_hv.csv`
- [ ] `Plotting/realworld_posthoc_validation/realworld_posthoc_per_dataset_pairwise_igd.csv`
- [ ] `Plotting/realworld_posthoc_validation/realworld_posthoc_friedman_posthoc_report.md`

## Reproducibility validation
- [ ] Environment installs from `requirements.txt` or `environment.yml`
- [ ] Synthetic runner quick test completes
- [ ] Posthoc runner quick test completes
- [ ] Stats scripts regenerate CSV outputs without errors
- [ ] Figure scripts regenerate manuscript figures

## Metadata
- [ ] Update `results/aggregate/manifest.json`
- [ ] Add/verify `README.md` run commands
- [ ] Add `LICENSE`
- [ ] Add `CITATION.cff`

## Git release
- [ ] `git status` clean
- [ ] Commit message: "release: runners + aggregate outputs for paper"
- [ ] Tag `v1.0.0-paper`
- [ ] Create GitHub release with short notes and artifact summary
