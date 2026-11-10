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
All entries below were made on 6 November 2026, after the main results had been seen.

1. **Event study covariates.** The spec did not say whether the event study includes the pre-period gas covariate. It does not: the event-study outcome is measured relative to year g−1, so a pre-gas covariate would partly overlap the outcome. Property, deprivation, region and cohort covariates are kept.
2. **Forest analysis additions (exploratory).**
   - The forest's heterogeneity variables also include pre-period mean ln(gas). IMD, council tax band, age band and floor area band are entered as ordered numbers.
   - Added a linear projection of the forest's effect estimates for treated homes on the covariates. It is descriptive only and has no valid standard error.
   - The held-out check is a tercile calibration: the forest is fitted on a random half of homes, the other half is ranked by predicted effect, and the DR-DiD ATT is estimated within each tercile.
3. **Extra subgroups.** Subgroup ATTs by council tax band (bands with at least 30 treated homes) and by property age band (1930–72 and 1973–99 only, since almost no pre-1930 or post-1999 homes are treated).

### Revision deviations (10 November 2026)

All entries below were made on 10 November 2026, after the first-version results had been seen and in response to a referee-style review of the first draft. None changes the main specification or its estimate (8.8%, reproduced exactly); each is an addition. Code: `src/rev_core.py`, `src/rev_het.py`, `src/rev_twfe.py`, `src/rev_overlap.py`; outputs in `outputs/revision/`. Seed 20261106 throughout.

4. **Balance and overlap (D4).** Added standardised mean differences of every covariate column, raw and with the estimator's comparison-group weights p/(1−p), and quantiles of the fitted propensity by group. Because weighted imbalance remained in pre-1930 homes, added three overlap checks: clipping propensities at 0.001 instead of 0.01, trimming rows with propensity below 0.01 instead of clipping them, and excluding pre-1930 homes. `drdid.nuisances` gained a `clip` argument (default 0.01, unchanged).
5. **Baseline-covariate bracket (D5).** The main specification's gas covariate uses the same years (g−2, g−1) as the outcome's pre-period. Added (a) pure DiD without the gas covariate (already reported as a robustness check), (b) lagged-outcome adjustment: outcome = post-period mean ln gas, adjusting for the pre-period level, no differencing (Ding and Li, 2019, bracketing), and (c) an earlier-baseline covariate, mean ln gas in g−4 and g−3. Because gas data start in 2005, (c) is possible only for cohorts 2009–2012; on those cohorts the same-window and no-covariate versions are re-estimated for comparison. `prep.stacked` gained a `base` argument.
6. **Joint event study and honest DiD (D6).** The event study is re-estimated with home-level influence functions so that the nine coefficients have a joint covariance matrix (point estimates unchanged). Added a joint Wald test of the three pre-installation coefficients and Rambachan–Roth sensitivity analysis for the average of e = 2, 3 (relative magnitudes, M̄ = 0 to 2 in steps of 0.25, conditional least-favourable intervals; smoothness, M = 0 to 0.03 in steps of 0.0025, fixed-length intervals), using the Python package `honestdid` 0.1.1. This bounds the event-study analogue of the main estimate (base g−1, no gas covariate), not the main estimate itself.
7. **Multiplicity and power (D7).** The IMD Q1−Q5 contrast stays the single heterogeneity test committed in advance. All other subgroup results are labelled exploratory. Holm-adjusted p-values are reported for three contrasts: IMD Q1 − Q5, council tax band A − bands D–F (pooled), and age band 1930–72 − 1973–99. The minimum detectable gap at 80% power and 5% two-sided significance is reported as 2.80 × the contrast's standard error.
8. **Absolute savings by earlier baseline (D8).** Savings in kWh and in log points by tercile of earlier gas use (g−4, g−3), cohorts 2009–2012, to test whether low pre-retrofit use predicts smaller savings once regression to the mean is removed.
9. **Inferential best linear projection (D9).** Replaces the first version's descriptive projection of forest estimates (which had no valid standard errors) with a projection of the CATT on IMD quintile, council tax band, floor area band, age band and baseline ln gas, using the doubly robust DiD score with home-clustered sandwich standard errors (Semenova and Chernozhukov, 2021). Run on the main sample (same-window baseline) and on cohorts 2009–2012 (earlier baseline). The first version's descriptive projection is retained in `outputs/forest_projection.csv` for the record but no longer reported.
10. **GATES, CLAN and AUTOC over repeated splits (D10).** Replaces the first version's single held-out split (terciles 10.6% / 9.0% / 6.7%, retained in `outputs/forest_calibration.csv`) with 20 random home-level splits. On each, a causal forest (400 trees, otherwise as in deviation 2) is fitted on one half and ranks the other; GATES by tercile, CLAN for top versus bottom tercile, and a targeting curve and its area (AUTOC, adapted to the ATT from Yadlowsky et al., 2025; 300 home-level Bayesian bootstrap draws, nuisances fixed). Splits are combined by the median rule of Chernozhukov et al. (2025): per-split 97.5% intervals, medians of bounds, p = min(1, 2 × median p).
11. **Two-way fixed effects comparison (D11).** To compare with Peñasco and Díaz Anadón (2023), who reported fading savings from a fixed-effects dynamic DiD on an earlier NEED release, added conventional TWFE event-study regressions of ln gas on home and year fixed effects and event-time indicators −5 to +5 (endpoints binned, −1 omitted), gas years 2005–2017, valid years only: (T1) all gas-heated homes with separate CWI and LI indicators; (T2) only the units of the main design. Standard errors clustered by home.
12. **Property type (D12).** Prompted by the revision results: among treated homes, 70% of council tax band A homes are flats or terraces against 7% in bands D–F, where 52% are detached. Added property-type indicators (semi-detached as reference) to the best linear projection of D9, on both samples, and subgroup ATTs by property type (`src/rev_blp_type.py`; projection functions moved unchanged to `src/rev_het_blp.py`). Exploratory.
