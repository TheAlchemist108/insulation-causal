# Do home insulation retrofits cut gas use, and who saves most?

Causal analysis of cavity wall and loft insulation installed under UK government schemes, 2007–2012, using the DESNZ National Energy Efficiency Data-Framework (NEED) anonymised data 2026 (50,000 homes in England and Wales, gas use 2005–2024).

The headline model was fixed in `SPEC.md` (committed 6 November 2026) before any gas or electricity values were analysed.

## Design in one paragraph

A home's gas use after installation is compared with its own use before, and that change is compared with the change in never-insulated homes over the same years (difference-in-differences). Cohorts installed in each year 2007–2012 are stacked. Differences between treated and comparison homes in property type, age, size, council tax band, deprivation, region and baseline gas use are adjusted for with a doubly robust estimator whose two models (propensity and comparison-group outcome) are machine-learned and cross-fitted (double/debiased ML). Variation in effects is estimated by subgroup with the same estimator, and explored with a causal forest checked on held-out data.

## Reproduce

```
pip install pandas numpy scikit-learn econml statsmodels matplotlib odfpy
# put anon2026_50k.csv in data/raw/ (URL and SHA-256 in data/MANIFEST.txt)
python src/run_main.py     # ATT, robustness, subgroups, event study  (~4 min)
python src/run_forest.py   # causal forest + held-out calibration     (~10 min)
python src/plots.py
```

Seed: 20261106. Outputs go to `outputs/`.

## Files

| File | What it does |
|---|---|
| `SPEC.md` | Main specification, fixed before outcomes were analysed; deviations logged at the end |
| `src/prep.py` | Builds stacked cohort samples and the differenced outcome |
| `src/drdid.py` | Cross-fitted doubly robust DiD with home-clustered standard errors |
| `src/run_main.py` | Main estimate, robustness table, IMD / council tax / age subgroups, event study |
| `src/run_forest.py` | Causal forest (exploratory), its projection on covariates, held-out calibration |
| `src/plots.py` | Figures |

## Data licence

NEED anonymised data are published by DESNZ under the Open Government Licence v3.0.
