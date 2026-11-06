"""Main ATT, robustness table, event study and IMD subgroups. Writes outputs/*.csv."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from prep import load, stacked, event_pair, design
from drdid import nuisances, att, diff_se

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)
d = load()


def fit(s):
    X = design(s)
    p, m0 = nuisances(X, s["D"], s["dY"], s["home_id"])
    return p, m0


def row(label, s, p, m0, mask=None, unit="log"):
    est, se, nt, _ = att(s["D"], s["dY"], p, m0, s["home_id"], mask)
    r = dict(spec=label, att=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se, n_treated=nt,
             n_rows=int(len(s) if mask is None else mask.sum()))
    if unit == "log":
        r["pct_saving"] = -(np.exp(est) - 1) * 100
    return r


rows = []
# Main specification
s = stacked(d, "CWI")
p, m0 = fit(s)
rows.append(row("MAIN: CWI, ln gas, post g+2..g+3", s, p, m0))
treated_p = p[s["D"].to_numpy() == 1]
overlap = dict(p_treated_min=float(treated_p.min()), p_treated_max=float(treated_p.max()),
               share_treated_clipped_hi=float((treated_p >= 0.99).mean()))

# Subgroups by IMD quintile (headline heterogeneity), same nuisances
sub = []
psi_store = {}
for q in [1, 2, 3, 4, 5]:
    m = (s["IMD"] == q).to_numpy()
    est, se, nt, psi = att(s["D"], s["dY"], p, m0, s["home_id"], m)
    psi_store[q] = (psi, s["home_id"].to_numpy()[m], m.sum())
    sub.append(dict(group=f"IMD Q{q}", att=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se,
                    pct_saving=-(np.exp(est) - 1) * 100, n_treated=nt))
a1, a5 = sub[0]["att"], sub[4]["att"]
se15 = diff_se(psi_store[1][0], psi_store[5][0], psi_store[1][1], psi_store[5][1],
               psi_store[1][2], psi_store[5][2])
sub.append(dict(group="Q1 minus Q5", att=a1 - a5, se=se15, lo=a1 - a5 - 1.96 * se15,
                hi=a1 - a5 + 1.96 * se15, pct_saving=np.nan, n_treated=np.nan))
for band in ["A", "B", "C", "D", "E", "F", "G+"]:
    m = (s["CTB"] == band).to_numpy()
    if s["D"].to_numpy()[m].sum() < 30:
        continue
    est, se, nt, _ = att(s["D"], s["dY"], p, m0, s["home_id"], m)
    sub.append(dict(group=f"Council tax {band}", att=est, se=se, lo=est - 1.96 * se,
                    hi=est + 1.96 * se, pct_saving=-(np.exp(est) - 1) * 100, n_treated=nt))
for ab, lab in [(2, "built 1930-72"), (3, "built 1973-99")]:
    m = (s["PROP_AGE_BAND"] == ab).to_numpy()
    est, se, nt, _ = att(s["D"], s["dY"], p, m0, s["home_id"], m)
    sub.append(dict(group=lab, att=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se,
                    pct_saving=-(np.exp(est) - 1) * 100, n_treated=nt))
pd.DataFrame(sub).to_csv(OUT / "subgroups.csv", index=False)

# Robustness
s2 = stacked(d, "CWI", post=(1, 2)); rows.append(row("CWI, post g+1..g+2", s2, *fit(s2)))
s3 = stacked(d, "CWI", kind="level"); rows.append(row("CWI, kWh levels", s3, *fit(s3), unit="kwh"))
s4 = stacked(d, "CWI", pre_cov=False); rows.append(row("CWI, no pre-gas covariate", s4, *fit(s4)))
s5 = stacked(d, "LI"); rows.append(row("Loft insulation only", s5, *fit(s5)))
s6 = stacked(d, "CWI", fuel="E"); rows.append(row("PLACEBO: CWI on ln electricity", s6, *fit(s6)))
pd.DataFrame(rows).to_csv(OUT / "att_table.csv", index=False)

# Event study, base e = -1
es = []
for e in [-4, -3, -2, 0, 1, 2, 3, 4, 5]:
    se_ = event_pair(d, e)
    est, se, nt, _ = att(se_["D"], se_["dY"], *fit(se_), se_["home_id"])
    es.append(dict(e=e, att=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se, n_treated=nt))
es.append(dict(e=-1, att=0.0, se=0.0, lo=0.0, hi=0.0, n_treated=np.nan))
pd.DataFrame(es).sort_values("e").to_csv(OUT / "event_study.csv", index=False)

# Mean pre-period gas of treated (to convert % to kWh)
info = dict(overlap, mean_pre_kwh_treated=float(s.loc[s.D == 1, "pre_kwh"].mean()))
json.dump(info, open(OUT / "info.json", "w"), indent=1)
print(pd.DataFrame(rows).round(4).to_string()); print(pd.DataFrame(sub).round(4).to_string())
print(pd.DataFrame(es).sort_values("e").round(4).to_string()); print(info)
