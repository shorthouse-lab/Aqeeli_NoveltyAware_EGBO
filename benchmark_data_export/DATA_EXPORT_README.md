# Comprehensive Benchmark Data Export Summary

## Overview
All benchmark results have been exported into structured CSV files for visualization and analysis. The dataset includes:
- **8 Problems** (5 unconstrained + 3 constrained)
- **5 Algorithms** (EGBO, Feryal, Ikhlas, Karima, Traditional_NEHVI)
- **25 trials per configuration** = 1,000+ total runs
- **200,000+ data records** across all files

## Generated Files

### 1. **HV Convergence & Performance**

#### `hv_convergence_all_runs.csv` (1.1 MB, 60,000+ records)
Contains HV values across all batches for every trial.
- **Columns**: `source`, `problem`, `algorithm`, `trial`, `batch`, `hv`
- **Use case**: Plot HV convergence curves with error bands
- **Example**: Track how each algorithm's HV improves over 20 optimization batches

#### `algorithm_comparison.csv` (2.5 KB, 41 records)
Summary statistics for each problem-algorithm pair (mean, std, min, max).
- **Columns**: `source`, `problem`, `problem_type`, `objectives`, `algorithm`, `mean_hv`, `std_hv`, `min_hv`, `max_hv`, `num_trials`
- **Use case**: Compare algorithm performance rankings per problem
- **Key finding**: Traditional_NEHVI fixed results on MW3 now show mean HV = 1.362 (was 0.0)

#### `algorithm_comparison_pivot_mean_hv.csv` (571 B)
Pivot table with algorithms as columns and problems as rows (Mean HV values).
- **Use case**: Quick visual comparison matrix across all algorithm-problem pairs

### 2. **Trial Details**

#### `trial_summary.csv` (76 KB, 1,000 records)
Per-trial statistics including HV, sample count, feasibility, and execution time.
- **Columns**: `source`, `problem`, `problem_type`, `algorithm`, `trial`, `final_hv`, `num_batches`, `num_samples`, `num_feasible`, `exec_time_sec`
- **Use case**: Analyze sample efficiency vs HV achieved
- **Insights**: Track which trials performed best and how many feasible solutions were found

### 3. **Source Tracking & Attribution**

#### `sample_source_tracking.csv` (small, extracted but source files as lists)
Intended to track which generator suggested each evaluated sample.
- **Columns**: `source_batch`, `problem`, `algorithm`, `trial`, `sample_idx`, `suggested_by`, `is_initial`
- **Note**: Some format issues with source_labels.json (list vs dict format)
- **Use case**: Analyze generator contribution to optimization success

### 4. **Pareto Front Data**

#### `pareto_objectives_data.csv` (8.3 MB, 181,425 records)
All objective values for every sample evaluated in every trial.
- **Columns**: `source_batch`, `problem`, `algorithm`, `trial`, `sample_idx`, `feasible`, `obj_1`, `obj_2`, [, `obj_3`]
- **Use case**: Reconstruct and visualize Pareto fronts for any algorithm-problem-trial combination
- **Capability**: Compute regret curves (theoretical vs empirical Pareto fronts)

## Key Findings from Current Data

### MW3 (Constrained, 2-objective)
| Algorithm | Mean HV | Std HV | Status |
|-----------|---------|--------|--------|
| EGBO | 2.002 | 0.016 | ✓ |
| Feryal | 1.993 | 0.031 | ✓ |
| Ikhlas | 2.004 | 0.024 | ✓ |
| Karima | 2.007 | 0.021 | ✓ |
| Traditional_NEHVI (OLD) | 0.0 | 0.0 | ✗ BUG |
| Traditional_NEHVI (FIXED) | 1.362 | 0.140 | ✓ NEW |

### Unconstrained Problems (ZDT & DTLZ)
Traditional_NEHVI shows exceptionally high HV values compared to other algorithms, indicating a scaling issue with reference points. This needs investigation.

### MW5 & MW7 (Constrained)
Traditional_NEHVI still shows 0.0 HV - waiting for full rerun to complete.

## How to Use These Files

### Plot HV Convergence Curves
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('hv_convergence_all_runs.csv')
for algo in df['algorithm'].unique():
    algo_data = df[df['algorithm'] == algo]
    mean_hv = algo_data.groupby('batch')['hv'].mean()
    std_hv = algo_data.groupby('batch')['hv'].std()
    plt.plot(mean_hv.index, mean_hv.values, label=algo)
    plt.fill_between(mean_hv.index, mean_hv - std_hv, mean_hv + std_hv, alpha=0.2)
plt.legend()
plt.show()
```

### Compare Algorithm Performance
```python
import pandas as pd

comp = pd.read_csv('algorithm_comparison.csv')
pivot = comp.pivot_table(index='problem', columns='algorithm', values='mean_hv')
print(pivot)  # View algorithm rankings per problem
```

### Reconstruct Pareto Fronts
```python
import pandas as pd

pareto = pd.read_csv('pareto_objectives_data.csv')
# Get feasible solutions for MW3 trial 1
mw3_feasible = pareto[(pareto['problem'] == 'MW3') & 
                       (pareto['trial'] == 1) & 
                       (pareto['feasible'] == True)]
# Plot Pareto front
plt.scatter(mw3_feasible['obj_1'], mw3_feasible['obj_2'])
plt.show()
```

### Analyze Sample Efficiency
```python
import pandas as pd

trials = pd.read_csv('trial_summary.csv')
for algo in trials['algorithm'].unique():
    algo_trials = trials[trials['algorithm'] == algo]
    print(f"{algo}:")
    print(f"  Avg HV: {algo_trials['final_hv'].mean():.4f}")
    print(f"  Avg Samples: {algo_trials['num_samples'].mean():.0f}")
    print(f"  Avg Feasible: {algo_trials['num_feasible'].mean():.0f}")
```

## File Statistics

| File | Size | Records | Records/MB |
|------|------|---------|-----------|
| hv_convergence_all_runs.csv | 1.1 MB | 60,000 | 54.5K |
| pareto_objectives_data.csv | 8.3 MB | 181,425 | 21.9K |
| trial_summary.csv | 76 KB | 1,000 | 13K |
| algorithm_comparison.csv | 2.5 KB | 41 | 16K |
| **Total** | **~9.5 MB** | **~242K** | - |

## Next Steps

1. **Complete MW5 & MW7 rerun** - Wait for `benchmark_results_nehvi_mw_fixed` to finish
2. **Re-export data** - After rerun completes, re-run `export_benchmark_data.py`
3. **Create visualizations** - Use the CSV files with matplotlib/seaborn/plotly
4. **Statistical analysis** - Perform significance tests across algorithms
5. **Publication-ready plots** - Generate final comparison figures for paper

## Notes

- ✓ All data is self-contained in CSV format (no special libraries needed to read)
- ✓ Source columns identify which batch data came from (Full_experiment_6thNov or NEHVI_MW_Fixed)
- ✓ HV values normalized appropriately per problem
- ⚠ Traditional_NEHVI unconstrained results show anomalously high HV values (investigation needed)
- ⚠ Some source_labels.json files are stored as lists rather than dicts (doesn't affect other data)
- ℹ MW3 NEHVI fixed results show reasonable convergence behavior

---
Generated: 2026-01-15
Export Tool: `export_benchmark_data.py`
