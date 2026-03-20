# Aqeeli et al. Novelty Aware EGBO
Github repository for code included in Aqeeli et al. Novelty-Aware Evolutionary Bayesian Optimisation for Multi-Objective Discovery Science

his release bundle contains the **code and aggregate outputs** needed to reproduce the manuscript-level comparisons for:
- `EGBO`
- `EGBO_Novelty` (novelty-aware variant)
- `Traditional_NEHVI` baseline

## What is included

- **Runner**
  - `run_synthetic_pairwise_egbo_novelty_nehvi.py`
- **Canonical aggregate results**
  - `benchmark_data_export/`
- **Additional aggregate data**
  - `additional_aggregate_data/noise_sweep/`
  - `additional_aggregate_data/scaling/`
  - `additional_aggregate_data/constraints/`
- **Post-hoc analysis outputs and scripts**
  - `Plotting/realworld_posthoc_validation/`
  - `Plotting/synthetic_pairwise_10problem_validation/`
- **Reproducibility files**
  - `requirements.txt`
  - `environment.yml`
  - `results/aggregate/manifest.json`

## Quick start

From this folder (`release_review_bundle/`):

```bash
python run_synthetic_pairwise_egbo_novelty_nehvi.py --quick
```

## Basic example: novelty-aware EGBO run

```bash
python run_synthetic_pairwise_egbo_novelty_nehvi.py \
  --problems MW5,ZDT1,DTLZ2_5obj \
  --algorithms EGBO,EGBO_Novelty_v1,Traditional_NEHVI \
  --trials 5 --batches 6 --batch-size 8 \
  --qnehvi-candidates 8 --evo-candidates 72 \
  --output-dir ./benchmark_results_pairwise_release
```

## Notes on interpretation

- **Hypervolume (HV):** higher is better
- **IGD:** lower is better
- The novelty-aware variant combines acquisition quality and novelty for candidate selection, improving coverage/diversity especially in multimodal settings.

## Suggested review order

1. Run the quick smoke test above.
2. Inspect `benchmark_data_export/` for primary aggregate tables.
3. Inspect `additional_aggregate_data/` for noise/scaling/constraints summaries.
4. Review post-hoc scripts in `Plotting/realworld_posthoc_validation/`.

## Reproducibility

Use either:

```bash
conda env create -f environment.yml
conda activate evo-bo-repro
```

or install from:

```bash
pip install -r requirements.txt
```

---

Note: This code and some files were curated and cleaned with the help of LLMs - origial code was NOT written with LLMs.
