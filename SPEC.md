# Main specification (fixed before outcomes were analysed)

**Study:** Do home insulation retrofits cut gas use, and who saves most? England and Wales, NEED anonymised data 2026
**Author:** Kaustubh Ramekar
**Written:** 6 November 2026, before any gas or electricity values were analysed. Only treatment, property and validity-flag columns had been tabulated (counts of installs by year, covariate balance, flag frequencies).

This is not a formal pre-analysis plan. It fixes the headline model so the main estimate is not chosen after seeing results. Everything not listed here is exploratory and reported as such. Changes after this commit are logged at the bottom with the date and reason.

## Data
- NEED anonymised data 2026, 50,000-row public sample (DESNZ, published 11 June 2026, OGL). File `anon2026_50k.csv`, SHA-256 recorded in `data/MANIFEST.txt`.
- Current domestic properties in England and Wales. Gas consumption 2005–2024, weather corrected, rounded to 100 kWh.

## Sample
- Gas is the main heating fuel (`MAIN_HEAT_FUEL = 1`).
- **Treated (primary): cavity wall insulation (CWI) only.** `CWI_FLAG = 1`, install year g in 2007–2012, and no loft insulation (LI) installed between g−2 and g+3.
- **Controls:** never treated by either measure (`CWI_FLAG = 0` and `LI_FLAG = 0`).
- Stacked design: for each cohort g, the treated homes of that cohort plus all controls. Controls appear once per cohort; inference clusters by home.
- A home enters a cohort stack only if gas is valid (`GasValFlag = V`) in all four window years.

## Timing
- Pre years: g−2, g−1. Post years: g+2, g+3.
- Years g and g+1 are excluded. Before 2015 a gas year runs October–September, and the install date is a calendar year, so g+1 can be partly pre-install.

## Outcome
- Δ = mean of ln(gas) over post years − mean of ln(gas) over pre years. Effects are in log points; % saving = 1 − exp(ATT).

## Covariates (all fixed before installation)
- Property type, age band, floor area band, council tax band (G, H, I merged), IMD quintile (England or Wales index, with a nation indicator), region, cohort, and pre-period mean ln(gas).
- **Not used:** EPC band (mostly recorded after installation, so affected by treatment); conservatory flag (81% missing).

## Estimand and estimator
- **ATT:** average effect of CWI on Δ for treated homes, pooled across cohorts with weights equal to treated counts.
- **Estimator:** doubly robust difference-in-differences (Sant'Anna and Zhao, 2020) with cross-fitted machine-learning nuisances (Chang, 2020): propensity p(X) and control outcome model m0(X) = E[Δ | D = 0, X], gradient-boosted trees, 5 folds grouped by home. Propensity clipped to [0.01, 0.99].
- **Inference:** standard errors from the influence function, clustered by home.

## Heterogeneity
- **Headline question:** do savings differ by deprivation? Specifically, are savings smaller in IMD quintile 1 than in quintile 5 (prebound / comfort-taking)?
- **Test:** subgroup ATTs by IMD quintile using the same DR-DiD estimator, and the difference Q1 − Q5 with a clustered SE.
- **Causal forest (exploratory):** CausalForestDML on Δ, with heterogeneity variables IMD quintile, council tax band, property age band, property type and floor area band. Reported as exploratory; any striking split is re-estimated on a held-out half.

## Checks
1. Event study: DR-DiD of ln(gas in g+e) − ln(gas in g−1), e = −4…+5, base e = −1. Pre-period coefficients should be near zero.
2. Robustness: post years g+1, g+2; outcome in kWh levels; without the pre-period gas covariate; loft insulation only as a second treatment.
3. Placebo outcome: electricity use (CWI should have at most a small effect).

## Known limits
- Controls may have been insulated outside government schemes, which biases the ATT towards zero.
- Sample is properties that still exist in 2025 and are matched to meters.
- No tenure variable in the public file; council tax band is used as the wealth proxy.
- `MAIN_HEAT_FUEL` is defined from 2022–2024 consumption.

## Deviations
(none yet)
