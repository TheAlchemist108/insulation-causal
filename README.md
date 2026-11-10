# Who saves from insulation? Persistent but unequal gas savings from cavity wall insulation in England and Wales

Code, logged specification, results and LaTeX source for the paper of the same name by Kaustubh Ramekar.

The study estimates the causal effect of cavity wall and loft insulation, installed under UK government schemes in 2007–2012, on household gas use, and asks which homes saved least. It uses the DESNZ National Energy Efficiency Data-Framework (NEED) anonymised data 2026 (50,000 homes in England and Wales, gas use 2005–2024).

The headline model was fixed in `SPEC.md` and committed on 6 November 2026, before any gas or electricity values were analysed. Every later change is logged there as a dated deviation (D1–D12). `RESULTS.md` summarises every number in the paper and the file it comes from.

## Design in one paragraph

A home's gas use after installation is compared with its own use before, and that change is compared with the change in never-insulated homes over the same years (difference-in-differences). Cohorts installed in each year 2007–2012 are stacked, each compared only with never-insulated homes. Differences between treated and comparison homes in property type, age, size, council tax band, deprivation, region and baseline gas use are adjusted for with a doubly robust estimator. Its two nuisance models (propensity and comparison-group outcome) are fitted with gradient-boosted trees and cross-fitting (double/debiased ML). The revision adds a Ding–Li bracket, an earlier-baseline check against regression to the mean, Rambachan–Roth sensitivity analysis, a two-way fixed effects comparison, and inferential heterogeneity analysis (best linear projection, GATES/CLAN and a targeting test over 20 sample splits).

## Reproduce

```
pip install -r requirements.txt
# put anon2026_50k.csv in data/raw/ (URL and SHA-256 in data/MANIFEST.txt)
python src/run_main.py        # main ATT, robustness, subgroups, event study      (~4 min)
python src/run_forest.py      # first-version causal forest (exploratory)         (~4 min)
python src/rev_core.py        # balance, bracket, early baseline, joint event study,
                              # honest DiD, contrasts/MDE, kWh by baseline        (~15 min)
python src/rev_overlap.py     # propensity clipping / trimming / pre-1930 checks  (~5 min)
python src/rev_twfe.py        # two-way fixed effects comparison                  (~1 min)
python src/rev_het.py 20      # BLP, GATES/CLAN/AUTOC over 20 splits              (~60 min)
python src/rev_blp_type.py    # BLP with property type, subgroups by type         (~3 min)
python src/make_tables.py     # paper tables from outputs/
python src/figures.py         # paper figures from outputs/
cd paper && python build_tex.py && latexmk -pdf main.tex && python audit.py
```

The paper's master text is `paper/paper_source.md` (exported from the editable draft). `build_tex.py` converts it to `main.tex`, using the figures in `paper/figures/`; `audit.py` recomputes every key number from `outputs/` and checks it against the text. `paper/arxiv_submission.zip` is the self-contained arXiv upload (`main.tex` + figures), tested to compile on its own.

Seed: 20261106. First-version outputs are in `outputs/`; revision outputs in `outputs/revision/`.

## Files

| File | What it does |
|---|---|
| `SPEC.md` | Main specification, fixed before outcomes were analysed; deviations D1–D12 logged at the end |
| `RESULTS.md` | Every reported number and its source file |
| `src/prep.py` | Builds stacked cohort samples and the differenced outcome |
| `src/drdid.py` | Cross-fitted doubly robust DiD with home-clustered standard errors |
| `src/run_main.py` | Main estimate, robustness, IMD / council tax / age subgroups, event study |
| `src/run_forest.py` | First-version causal forest (exploratory; superseded by `rev_het.py` for inference) |
| `src/rev_core.py` | Revision: balance, Ding–Li bracket, early baseline, joint event study, honest DiD, contrasts, MDE |
| `src/rev_overlap.py` | Revision: sensitivity to propensity clipping and trimming |
| `src/rev_twfe.py` | Revision: two-way fixed effects event studies for comparison with earlier NEED work |
| `src/rev_het.py`, `src/rev_het_blp.py` | Revision: best linear projection, GATES/CLAN and AUTOC over repeated splits |
| `src/rev_blp_type.py` | Revision: projection and subgroups by property type |
| `src/make_tables.py`, `src/figures.py` | Paper tables and figures, built only from `outputs/` |
| `paper/paper_source.md` | Master text of the paper |
| `paper/build_tex.py`, `paper/template.tex` | Build `main.tex` from the master text |
| `paper/audit.py` | Checks every key number in the paper against `outputs/` |
| `paper/main.tex`, `paper/main.pdf`, `paper/arxiv_submission.zip` | LaTeX source, compiled PDF and arXiv package |

## Data licence

NEED anonymised data are published by DESNZ under the Open Government Licence v3.0. The raw file is not redistributed here; `data/MANIFEST.txt` gives its URL and hash.
