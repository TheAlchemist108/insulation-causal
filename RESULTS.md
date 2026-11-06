# Results (first run, 6 November 2026)

NEED anonymised data 2026, 50k sample. Gas-heated homes in England and Wales, cohorts installed 2007–2012. Doubly robust DiD with cross-fitted ML nuisances; 95% CIs clustered by home. Saving % = 1 − exp(ATT).

## Average effect

| Specification | Saving | 95% CI | Treated homes |
|---|---|---|---|
| **Main: cavity wall insulation (CWI), post years g+2..g+3** | **8.8%** | 7.5–10.2% | 1,801 |
| CWI, post years g+1..g+2 | 9.6% | 8.3–10.7% | 1,757 |
| CWI, kWh levels | 1,437 kWh/yr | 1,262–1,612 | 1,801 |
| CWI, no pre-gas covariate | 10.6% | 9.2–12.0% | 1,801 |
| Loft insulation only | 2.4% | 1.2–3.7% | 2,877 |
| Placebo: CWI effect on electricity | 0.2% | −1.4–1.8% | 1,991 |

- **Benchmark:** DESNZ's own matched-comparison estimates for 2011 installs are 9.2% (CWI) and 2.8% (loft).
- **Event study:** savings appear in the install gas year (7.5%), reach about 10% by year 1, and hold through year 5 with no sign of decay. Pre-install coefficients are small (−1.5 to −0.5%), but e = −2 is just significant (−1.4%, CI −2.6 to −0.2): treated homes used slightly more gas two years before install than one year before. This is a mild pre-trend, about one-seventh of the effect size.

## Who saves most (CWI)

| Group | Saving | 95% CI |
|---|---|---|
| IMD Q1 (most deprived) | 6.4% | 2.5–10.1% |
| IMD Q2 | 8.7% | 4.9–12.4% |
| IMD Q3 | 8.5% | 5.3–11.6% |
| IMD Q4 | 10.0% | 7.0–12.9% |
| IMD Q5 (least deprived) | 9.7% | 7.6–11.6% |
| **Q1 − Q5 gap** | 3.6 log points smaller in Q1 | −1.1 to +8.2 (not significant) |
| Council tax A | 5.0% | 0.8–9.1% |
| Council tax B–C | 8.4–8.6% | |
| Council tax D–F | 10.9–11.5% | |

## Causal forest (exploratory)

- **Variable importance:** baseline gas use (0.44) ≫ property age (0.20) > council tax band (0.14) > IMD (0.10) > floor area (0.06).
- **Projection of effects on covariates (treated homes, descriptive):** holding the other variables fixed, the IMD coefficient is about zero (+0.002 per quintile). Larger savings go with higher baseline gas use (−0.022 per log point) and higher council tax band (−0.011 per band).
- **Held-out calibration:** a forest fitted on half the homes ranks the other half reasonably well.
  - Largest-predicted tercile: 10.6% saving (7.6–13.6%).
  - Smallest-predicted tercile: 6.7% (3.1–10.1%).
  - The smallest-saving tercile has 26% IMD Q1 homes versus 11% in the largest, and 30% council tax A versus 14%.
  - The forest's predicted spread is wider than the realised one, so its individual effect estimates are noisy and overstated.

## Reading

The raw deprivation gradient is real in direction, but in this sample it is mostly explained by low baseline gas use and low council tax band rather than area deprivation as such. That pattern fits the prebound effect: homes already using little gas, many of them in deprived areas, have less to save, or take the gain as warmth. With about 300 treated homes per quintile, the sample cannot separate these explanations with confidence. The 4-million-row licensed file would.
