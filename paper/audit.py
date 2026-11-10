"""Consistency audit: every key number in the paper is recomputed from outputs/ and searched for
in the paper text (paper_source.md); stale first-version claims must be absent.
Usage: python audit.py   (exit code 1 on any failure)"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent; R = P.parent
O, V = R / "outputs", R / "outputs" / "revision"
doc = (P / "paper_source.md").read_text().replace("\\[", "[").replace("\\]", "]").replace("\\*", "*")
pct = lambda v: -(np.exp(v) - 1) * 100
f1 = lambda v: f"{(round(v, 1) + 0.0):.1f}".replace("-", "−")
fails = []
def need(s, why):
    if s not in doc:
        fails.append(f"MISSING  {s!r}  ({why})")
def forbid(pattern, why):
    m = re.search(pattern, doc)
    if m:
        fails.append(f"STALE    {m.group(0)!r}  ({why})")

a = pd.read_csv(O / "att_table.csv"); b = pd.read_csv(V / "bracket.csv"); core = json.load(open(V / "core_info.json"))
m = a.iloc[0]
need(f"{f1(pct(m.att))}% (95% CI {f1(pct(m.hi))}–{f1(pct(m.lo))}%)", "main estimate")
need(f"{-a.iloc[2].att:,.0f} kWh a year", "kWh estimate")
need(f"{f1(pct(b.iloc[2].att))}–{f1(pct(b.iloc[1].att))}%", "Ding-Li bracket")
need(f"from {f1(pct(b.iloc[4].att))}% to {f1(pct(b.iloc[3].att))}%", "early-baseline comparison")
need(f"Loft insulation saved {f1(pct(a.iloc[4].att))}%", "loft")
need(f"({f1(pct(a.iloc[5].att))}%, CI {f1(pct(a.iloc[5].hi))} to {f1(pct(a.iloc[5].lo))}%)", "placebo")
es = pd.read_csv(V / "event_study_joint.csv").set_index("e")
need(f"average {f1(np.mean([pct(es.loc[e].att) for e in range(1, 6)]))}%", "event-study years 1-5 mean")
need(f"still {f1(pct(es.loc[5].att))}%", "year 5")
need(f"Savings reach {f1(pct(es.loc[0].att))}%", "year 0")
need(f"used {f1(-pct(es.loc[-2].att))}% more gas", "pre-trend e=-2")
w = core["honest"]["pre_trend_wald"]
need(f"χ² = {w['wald']:.2f} with 3 degrees of freedom, p = {w['p']:.3f}", "Wald test")
need(f"(p = {w['p']:.3f})", "Wald p in 7.4")
c = pd.read_csv(V / "contrasts.csv").set_index("contrast")
q = c.loc["IMD Q1 minus IMD Q5"]
need(f"is {f1(100 * q['diff'])} log points (95% CI {f1(100 * q.lo)} to {f1(100 * q.hi)}, p = {q.p:.2f})", "IMD contrast")
need(f"{f1(100 * q.mde_80)} log points, nearly twice", "IMD MDE")
k = c.loc["CTB A minus CTB D-F"]
need(f"Holm-adjusted p = {k.p_holm:.3f}", "CTB Holm p (abstract)")
need(f"{f1(pct(k.att_a))}% against {f1(pct(k.att_b))}%", "CTB savings")
need(f"{f1(100 * k['diff'])} log points (95% CI {f1(100 * k.lo)}–{f1(100 * k.hi)})", "CTB gap")
sub = pd.read_csv(O / "subgroups.csv").set_index("group")
need(f"{f1(sub.loc['IMD Q1'].pct_saving)}% against {f1(sub.loc['IMD Q5'].pct_saving)}%", "IMD savings in abstract")
st = pd.read_csv(V / "subgroups_type.csv").set_index("group")
need(f"{f1(st.loc['Detached'].pct_saving)}% in detached houses", "detached")
need(f"{f1(st.loc['Bungalow'].pct_saving)}% in bungalows", "bungalow")
kw = pd.read_csv(V / "kwh_by_baseline.csv")
need(", ".join(f"{f1(v)}%" for v in kw.pct_saving[:2]) + f" and {f1(kw.pct_saving[2])}%", "baseline terciles")
ga = pd.read_csv(V / "gates.csv").set_index("group"); au = json.load(open(V / "autoc.json"))
need(f"saved {f1(ga.loc['G1 largest predicted saving'].pct_saving)}% and those it ranks lowest {f1(ga.loc['G3 smallest predicted saving'].pct_saving)}%", "GATES")
need(f"adjusted p = {ga.loc['G3 minus G1'].p_adj:.2f}", "GATES gap p")
need(f"{f1(100 * au['autoc_median'])} log points (95% CI {f1(100 * au['lo'])} to {f1(100 * au['hi'])}, adjusted p = {au['p_adj']:.2f})", "AUTOC")
hd = core["honest"]
need(f"| {f1(pct(hd['target_est']))} |", "honest target")
blp = pd.read_csv(V / "blp.csv")
r = blp[(blp.model == "main sample, IMD only") & (blp.term == "IMD quintile")].iloc[0]
need(f"{f1(-100 * r.coef)} points of saving per quintile towards the least deprived (95% CI {f1(-100 * r.hi)} to {f1(-100 * r.lo)}, p = {r.p:.2f})", "IMD-only BLP")
tw = pd.read_csv(V / "twfe_event.csv"); t1 = tw[tw.spec.str.startswith("T1")].set_index("e"); t2 = tw[tw.spec.str.startswith("T2")].set_index("e")
v1 = sorted(round(pct(t1.loc[e].coef), 1) for e in [1, 2, 4, 5])
need(f"{f1(v1[0])}–{f1(v1[-1])}% in years one, two, four and five and {f1(pct(t1.loc[3].coef))}% in year three", "TWFE T1")
v2 = sorted(round(pct(t2.loc[e].coef), 1) for e in range(1, 6))
need(f"{f1(v2[0])}–{f1(v2[-1])}% in years one to five", "TWFE T2")
pre = sorted(round(-pct(t1.loc[e].coef), 1) for e in [-5, -4, -3, -2])
need(f"{f1(pre[0])}–{f1(pre[-1])}%, three of them significant", "TWFE pre-trends")
ov = pd.read_csv(V / "overlap_sensitivity.csv")
need(f"all give {f1(ov.pct_saving.min())}–{f1(ov.pct_saving.max())}%", "overlap checks")
# every generated table appears verbatim (after escaping)
for t in sorted((P / "tables").glob("*.md")):
    if t.stem in ("tA_propensity", "t7b_clan") or True:
        rows = [l for l in t.read_text().splitlines()[2:]]
        for l in rows:
            if t.stem == "tA_propensity" and l.startswith("| 0.01"):
                continue      # the paper omits the 0.01 quantile row
            if l not in doc:
                fails.append(f"TABLE    {t.name}: row not in paper: {l[:90]}")
# stale first-version claims
forbid(r"Warmth or savings", "old title")
forbid(r"runs through baseline", "old mechanism claim")
forbid(r"8\.3–10\.7", "old rounding of post g+1..g+2 CI")
forbid(r"settle at about 10%", "old abstract phrase")
forbid(r"\[RESULT NEEDED", "placeholder left")
print("\n".join(fails) if fails else "AUDIT PASSED: all checked numbers match outputs; no stale claims found")
sys.exit(1 if fails else 0)
