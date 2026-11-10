"""Build the paper's tables (markdown) directly from outputs/, so no number is typed by hand.
Writes paper/tables/*.md."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
O, V = R / "outputs", R / "outputs" / "revision"
T = R / "paper" / "tables"; T.mkdir(parents=True, exist_ok=True)
pct = lambda v: -(np.exp(v) - 1) * 100
f1 = lambda v: f"{(round(v, 1) + 0.0):.1f}".replace("-", "\u2212")
ci = lambda lo, hi: f"{f1(lo)} to {f1(hi)}" if min(lo, hi) < 0 else f"{f1(lo)}–{f1(hi)}"
def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)

# ---- Table 1: average effects and robustness
a = pd.read_csv(O / "att_table.csv"); b = pd.read_csv(V / "bracket.csv"); ov = pd.read_csv(V / "overlap_sensitivity.csv")
rows = []
def logrow(label, att, se, n, cohorts="2007–12"):
    rows.append((label, cohorts, f1(pct(att)) + "%", ci(pct(att + 1.96 * se), pct(att - 1.96 * se)), f"{int(n):,}"))
m = a.iloc[0]; logrow("**Main specification** (committed in advance)", m.att, m.se, m.n_treated)
for lab, src in [("Pure DiD: no gas covariate", b.iloc[1]), ("Lagged-outcome adjustment", b.iloc[2])]:
    logrow(lab, src.att, src.se, src.n_treated)
for lab, i in [("Early baseline covariate (g−4, g−3)", 3), ("Same-window covariate, same cohorts", 4), ("Pure DiD, same cohorts", 5)]:
    logrow(lab, b.iloc[i].att, b.iloc[i].se, b.iloc[i].n_treated, "2009–12")
logrow("Post years g+1 to g+2", a.iloc[1].att, a.iloc[1].se, a.iloc[1].n_treated)
k = a.iloc[2]
rows.append(("Outcome in kWh", "2007–12", f"{-k.att:,.0f} kWh/yr", f"{-k.hi:,.0f}–{-k.lo:,.0f}", f"{int(k.n_treated):,}"))
for lab, i in [("Propensities clipped at 0.001", 0), ("Pre-1930 homes excluded", 1), ("Propensity below 0.01 trimmed", 2)]:
    logrow(lab, ov.iloc[i].att, ov.iloc[i].se, ov.iloc[i].n_treated)
logrow("Loft insulation only (second treatment)", a.iloc[4].att, a.iloc[4].se, a.iloc[4].n_treated)
logrow("Placebo: CWI effect on electricity", a.iloc[5].att, a.iloc[5].se, a.iloc[5].n_treated)
t1 = pd.DataFrame(rows, columns=["Specification", "Cohorts", "Gas saving", "95% CI", "Treated homes"])
(T / "t1_average.md").write_text(md(t1))

# ---- Table 2: contrasts with Holm and MDE
c = pd.read_csv(V / "contrasts.csv")
lab = {"IMD Q1 minus IMD Q5": "IMD Q1 − Q5 (committed in advance)", "CTB A minus CTB D-F": "Council tax A − bands D–F",
       "Built 1930-72 minus Built 1973-99": "Built 1930–72 − built 1973–99"}
t2 = pd.DataFrame({"Contrast (gap in saving, log points × 100)": [lab[x] for x in c.contrast],
                   "Savings (%)": [f"{f1(pct(r.att_a))} vs {f1(pct(r.att_b))}" for r in c.itertuples()],
                   "Gap": [f1(100 * r.diff) for r in c.itertuples()], "95% CI": [ci(100 * r.lo, 100 * r.hi) for r in c.itertuples()],
                   "p": [f"{r.p:.3f}" for r in c.itertuples()], "Holm p": [f"{r.p_holm:.3f}" for r in c.itertuples()],
                   "MDE (80%)": [f1(100 * r.mde_80) for r in c.itertuples()]})
(T / "t2_contrasts.md").write_text(md(t2))

# ---- Table 3: savings by early-baseline tercile
kw = pd.read_csv(V / "kwh_by_baseline.csv")
t3 = pd.DataFrame({"Tercile of gas use in g−4, g−3": [x.capitalize() for x in kw.baseline_tercile],
                   "Mean baseline (kWh/yr)": [f"{v:,.0f}" for v in kw.mean_base_kwh_treated],
                   "Saving (%)": [f1(v) for v in kw.pct_saving], "95% CI (%)": [ci(r.pct_lo, r.pct_hi) for r in kw.itertuples()],
                   "Saving (kWh/yr)": [f"{-v:,.0f}" for v in kw.kwh_att], "95% CI (kWh)": [f"{-r.kwh_hi:,.0f}–{-r.kwh_lo:,.0f}" for r in kw.itertuples()],
                   "Treated homes": kw.n_treated})
(T / "t3_baseline.md").write_text(md(t3))

# ---- Table C1: event study (DR joint + TWFE)
es = pd.read_csv(V / "event_study_joint.csv"); tw = pd.read_csv(V / "twfe_event.csv")
t1w, t2w = tw[tw.spec.str.startswith("T1")].set_index("e"), tw[tw.spec.str.startswith("T2")].set_index("e")
rows = []
for e in range(-5, 6):
    if e == -1:
        rows.append((str(e).replace("-", "\u2212"), "0 (base)", "", "0 (base)", "", "0 (base)")); continue
    r = es.set_index("e").loc[e] if e in set(es.e) else None
    dr = (f1(pct(r.att)), ci(pct(r.hi), pct(r.lo)), f"{int(r.n_treated):,}") if r is not None else ("–", "", "")
    rows.append((str(e).replace("-", "\u2212"), dr[0], dr[1], f"{f1(pct(t1w.loc[e].coef))} ({ci(pct(t1w.loc[e].hi), pct(t1w.loc[e].lo))})",
                 dr[2], f"{f1(pct(t2w.loc[e].coef))} ({ci(pct(t2w.loc[e].hi), pct(t2w.loc[e].lo))})"))
tc = pd.DataFrame(rows, columns=["Year e", "This paper: saving (%)", "95% CI", "Two-way FE, all homes: saving (95% CI)",
                                 "Treated homes", "Two-way FE, main-design units: saving (95% CI)"])
tc = tc[["Year e", "This paper: saving (%)", "95% CI", "Treated homes", "Two-way FE, all homes: saving (95% CI)",
         "Two-way FE, main-design units: saving (95% CI)"]]
(T / "tC_event.md").write_text(md(tc))

# ---- Table A1: balance (selected covariates) + summary
bal = pd.read_csv(V / "balance.csv")
nice = {"PROP_AGE_BAND_1": "Built before 1930", "PROP_AGE_BAND_2": "Built 1930–72", "PROP_AGE_BAND_3": "Built 1973–99",
        "PROP_AGE_BAND_4": "Built 2000 or later", "PROP_TYPE_Mid terrace": "Mid terrace", "PROP_TYPE_Semi detached": "Semi-detached",
        "PROP_TYPE_Flat": "Flat", "PROP_TYPE_Detached": "Detached", "REGION_E12000002": "North West", "REGION_E12000007": "London",
        "IMD_1": "IMD quintile 1", "IMD_5": "IMD quintile 5", "CTB_A": "Council tax band A", "CTB_D": "Council tax band D",
        "pre_level": "Pre-period mean ln gas (g−2, g−1)", "FLOOR_AREA_BAND_1": "Floor area band 1 (smallest)",
        "FLOOR_AREA_BAND_4": "Floor area band 4"}
sel = bal[bal.covariate.isin(nice)].copy()
sel["order"] = sel.covariate.map(list(nice).index); sel = sel.sort_values("order")
ta = pd.DataFrame({"Covariate": sel.covariate.map(nice), "Treated mean": [f"{v:.3f}" for v in sel.mean_treated],
                   "Comparison, raw": [f"{v:.3f}" for v in sel.mean_control_raw], "Comparison, weighted": [f"{v:.3f}" for v in sel.mean_control_weighted],
                   "SMD raw": [f"{v:.2f}".replace("-", "\u2212") for v in sel.smd_raw], "SMD weighted": [f"{v:.2f}".replace("-", "\u2212") for v in sel.smd_weighted]})
(T / "tA_balance.md").write_text(md(ta))
pq = pd.read_csv(V / "propensity_quantiles.csv")
tp = pd.DataFrame({"Quantile": [f"{q:.2f}" for q in pq["quantile"]], "Treated rows": [f"{v:.4f}" for v in pq.treated],
                   "Comparison rows": [f"{v:.4f}" for v in pq.control]})
(T / "tA_propensity.md").write_text(md(tp))
print(md(t1)); print(); print(md(t2)); print(); print(md(t3)); print(); print(md(tc)); print(); print(md(ta)); print(); print(md(tp))

# ---- Table 4: honest-DiD sensitivity (selected values)
rm_, sd_ = pd.read_csv(V / "honest_rm.csv"), pd.read_csv(V / "honest_sd.csv")
info = json.load(open(V / "core_info.json"))["honest"]
rows = [("None (parallel trends holds exactly)", "–", f1(pct(info["target_est"])),
         ci(pct(info["target_est"] + 1.96 * info["target_se"]), pct(info["target_est"] - 1.96 * info["target_se"])), "No")]
for r in rm_[rm_.Mbar.isin([0.5, 1.0, 1.25, 1.5])].itertuples():
    rows.append(("Relative magnitudes", f"M̄ = {r.Mbar:g}", "", ci(pct(r.ub), pct(r.lb)), "Yes" if r.lb <= 0 <= r.ub else "No"))
for r in sd_[sd_.M.round(4).isin([0.0, 0.0025, 0.005, 0.0075])].itertuples():
    rows.append(("Smoothness", f"M = {r.M:g}", "", ci(pct(r.ub), pct(r.lb)), "Yes" if r.lb <= 0 <= r.ub else "No"))
t4 = pd.DataFrame(rows, columns=["Restriction on post-installation violations", "Value", "Estimate (%)", "Robust 95% CI for saving (%)", "Includes zero"])
(T / "t4_honest.md").write_text(md(t4))
print(md(t4))

# ---- Table 5: best linear projection (change in saving per unit, log points x 100; positive = larger saving)
blp = pd.concat([pd.read_csv(V / "blp.csv"), pd.read_csv(V / "blp_with_type.csv")])
models = [("main sample, joint (same-window baseline)", "A"), ("main sample, joint + property type", "B"),
          ("cohorts 2009-12, joint (early baseline g-4,g-3)", "C"), ("cohorts 2009-12, joint + property type (early baseline)", "D")]
terms = ["Constant (ATT)", "IMD quintile", "Council tax band", "Floor area band", "Age band", "Baseline ln gas",
         "Type: detached", "Type: bungalow", "Type: flat", "Type: terraced"]
tl = {"Constant (ATT)": "Average saving (ATT)", "IMD quintile": "IMD quintile (1 = most deprived)",
      "Council tax band": "Council tax band (A = 1)", "Floor area band": "Floor area band", "Age band": "Age band",
      "Baseline ln gas": "Baseline ln gas", "Type: detached": "Detached (vs semi-detached)", "Type: bungalow": "Bungalow",
      "Type: flat": "Flat", "Type: terraced": "Terraced"}
rows = []
for t in terms:
    row = [tl[t]]
    for mname, _ in models:
        r = blp[(blp.model == mname) & (blp.term == t)]
        if len(r) == 0:
            row.append(""); continue
        r = r.iloc[0]
        star = "*" if r.p < 0.05 else ""
        row.append(f"{f1(-100 * r.coef)}{star} [{f1(-100 * r.hi)}, {f1(-100 * r.lo)}]")
    rows.append(row)
t5 = pd.DataFrame(rows, columns=["Term", "(A) Main sample", "(B) A + property type", "(C) 2009–12, early baseline", "(D) C + property type"])
(T / "t5_blp.md").write_text(md(t5))

# ---- Table 5b: savings by property type
st = pd.read_csv(V / "subgroups_type.csv")
t5b = pd.DataFrame({"Property type": st.group, "Saving (%)": [f1(v) for v in st.pct_saving],
                    "95% CI": [ci(r.pct_lo, r.pct_hi) for r in st.itertuples()], "Treated homes": [f"{int(n):,}" for n in st.n_treated]})
(T / "t5b_type.md").write_text(md(t5b))
print(md(t5)); print(); print(md(t5b))
