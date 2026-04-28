# Act 10 — Friedman and post hoc statistics

Algorithms: Novelty Aware EGBO, EGBO, qNEHVI
Datasets: Suzuki, ADA coatings, GDSC CRC5, SDL5

## Dataset-level HV means (4 blocks)

- Friedman: $\chi^2=1.5000$, $p=0.472367$
- Iman–Davenport: $F=0.6923$, $p=0.536377$
- Nemenyi CD ($\alpha=0.05$): 1.6568
- Average ranks (best = lowest):
  - Novelty Aware EGBO: 1.5000
  - EGBO: 2.2500
  - qNEHVI: 2.2500
- Pairwise Wilcoxon (Holm-adjusted):
  - Novelty Aware EGBO vs EGBO: p=0.375000, p_holm=0.750000, RBC=0.6000, sig=no, favored=Novelty Aware EGBO
  - Novelty Aware EGBO vs qNEHVI: p=0.250000, p_holm=0.750000, RBC=0.8000, sig=no, favored=Novelty Aware EGBO
  - EGBO vs qNEHVI: p=1.000000, p_holm=1.000000, RBC=0.0000, sig=no, favored=qNEHVI

## Dataset-level IGD means (4 blocks)

- Friedman: $\chi^2=3.5000$, $p=0.173774$
- Iman–Davenport: $F=2.3333$, $p=0.177979$
- Nemenyi CD ($\alpha=0.05$): 1.6568
- Average ranks (best = lowest):
  - Novelty Aware EGBO: 1.2500
  - qNEHVI: 2.2500
  - EGBO: 2.5000
- Pairwise Wilcoxon (Holm-adjusted):
  - Novelty Aware EGBO vs EGBO: p=0.125000, p_holm=0.375000, RBC=-1.0000, sig=no, favored=Novelty Aware EGBO
  - Novelty Aware EGBO vs qNEHVI: p=0.375000, p_holm=0.750000, RBC=-0.6000, sig=no, favored=Novelty Aware EGBO
  - EGBO vs qNEHVI: p=1.000000, p_holm=1.000000, RBC=0.0000, sig=no, favored=EGBO

## Trial-aligned HV (4 datasets × 10 trials = 40 blocks)

- Friedman: $\chi^2=16.4545$, $p=0.000267$
- Iman–Davenport: $F=10.0987$, $p=0.000126$
- Nemenyi CD ($\alpha=0.05$): 0.5239
- Average ranks (best = lowest):
  - Novelty Aware EGBO: 1.4875
  - qNEHVI: 2.2250
  - EGBO: 2.2875
- Pairwise Wilcoxon (Holm-adjusted):
  - Novelty Aware EGBO vs EGBO: p=0.000073, p_holm=0.000218, RBC=0.7583, sig=yes, favored=Novelty Aware EGBO
  - Novelty Aware EGBO vs qNEHVI: p=0.001773, p_holm=0.003545, RBC=0.5744, sig=yes, favored=Novelty Aware EGBO
  - EGBO vs qNEHVI: p=0.402413, p_holm=0.402413, RBC=-0.1538, sig=no, favored=qNEHVI

## Trial-aligned IGD (4 datasets × 10 trials = 40 blocks)

- Friedman: $\chi^2=12.7534$, $p=0.001701$
- Iman–Davenport: $F=7.3964$, $p=0.001145$
- Nemenyi CD ($\alpha=0.05$): 0.5239
- Average ranks (best = lowest):
  - Novelty Aware EGBO: 1.5750
  - qNEHVI: 2.1125
  - EGBO: 2.3125
- Pairwise Wilcoxon (Holm-adjusted):
  - Novelty Aware EGBO vs EGBO: p=0.000108, p_holm=0.000325, RBC=-0.7964, sig=yes, favored=Novelty Aware EGBO
  - Novelty Aware EGBO vs qNEHVI: p=0.004969, p_holm=0.009937, RBC=-0.5154, sig=yes, favored=Novelty Aware EGBO
  - EGBO vs qNEHVI: p=0.648496, p_holm=0.648496, RBC=-0.0871, sig=no, favored=EGBO
