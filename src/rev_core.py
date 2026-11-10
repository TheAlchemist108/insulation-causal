"""Revision analyses, part 1 (deviations D4-D8 in SPEC.md).

D4  covariate balance and propensity distribution for the main specification
D5  Ding-Li bracket: pure DiD, lagged-dependent-variable (LDV) adjustment, and an
    earlier-baseline (g-4, g-3) covariate on cohorts 2009-2012
D6  joint event study (home-level influence functions) + honest-DiD sensitivity
D7  minimum detectable IMD Q1-Q5 gap and Holm-adjusted subgroup contrasts
D8  absolute (kWh) savings by tercile of the earlier baseline

Writes outputs/revision/*.csv|json. Seed 20261106 throughout (via drdid.SEED).
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent))
from prep import load, stacked, event_pair, design
from drdid import nuisances, att, diff_se

OUT = Path(__file__).resolve().parents[1] / "outputs" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
d = load()
pct = lambda v: -(np.exp(v) - 1) * 100


def fit(s):
    X = design(s)
    p, m0 = nuisances(X, s["D"], s["dY"], s["home_id"])
    return X, p, m0


def summ(label, s, p, m0, y="dY", mask=None, log=True):
    est, se, nt, psi = att(s["D"], s[y], p, m0, s["home_id"], mask)
    r = dict(spec=label, att=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se, n_treated=nt,
             cohorts="-".join(str(c) for c in sorted(s["cohort"].unique())[::max(1, s["cohort"].nunique() - 1)]))
    if log:
        r.update(pct_saving=pct(est), pct_lo=pct(r["hi"]), pct_hi=pct(r["lo"]))
    return r, psi


# ---------------------------------------------------------------- main spec (reproduction check)
s = stacked(d, "CWI")
X, p, m0 = fit(s)
main, _ = summ("MAIN (reproduced)", s, p, m0)
D = s["D"].to_numpy()

# ---------------------------------------------------------------- D4 balance
w = np.where(D == 1, 1.0, p / (1 - p))
bal = []
for c in X.columns:
    x = X[c].to_numpy()
    xt, xc = x[D == 1], x[D == 0]
    sd = np.sqrt((xt.var() + xc.var()) / 2)
    mc_w = np.average(xc, weights=w[D == 0])
    bal.append(dict(covariate=c, mean_treated=xt.mean(), mean_control_raw=xc.mean(), mean_control_weighted=mc_w,
                    smd_raw=(xt.mean() - xc.mean()) / sd if sd > 0 else 0.0,
                    smd_weighted=(xt.mean() - mc_w) / sd if sd > 0 else 0.0))
bal = pd.DataFrame(bal)
bal.to_csv(OUT / "balance.csv", index=False)
qs = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
ps = pd.DataFrame({"quantile": qs, "treated": np.quantile(p[D == 1], qs), "control": np.quantile(p[D == 0], qs)})
ps.to_csv(OUT / "propensity_quantiles.csv", index=False)
np.save(OUT / "propensity_main.npy", np.c_[p, D])
bal_summary = dict(n_covariates=len(bal), max_abs_smd_raw=float(bal.smd_raw.abs().max()),
                   max_abs_smd_weighted=float(bal.smd_weighted.abs().max()),
                   n_raw_over_0_1=int((bal.smd_raw.abs() > 0.1).sum()),
                   n_weighted_over_0_1=int((bal.smd_weighted.abs() > 0.1).sum()),
                   share_controls_p_at_floor=float((p[D == 0] <= 0.01).mean()),
                   share_treated_p_at_floor=float((p[D == 1] <= 0.01).mean()))

# ---------------------------------------------------------------- D5 bracket + earlier baseline
rows = [main]
s_nd = stacked(d, "CWI", pre_cov=False); _, p_nd, m_nd = fit(s_nd)
rows.append(summ("Pure DiD (no gas covariate)", s_nd, p_nd, m_nd)[0])
s_ldv = stacked(d, "CWI")            # same rows as main; outcome = post-period mean log gas
X_l = design(s_ldv)
p_l, m_l = nuisances(X_l, s_ldv["D"], s_ldv["post_m"], s_ldv["home_id"])
rows.append(summ("LDV: post-period level, adjusting for pre-period level", s_ldv, p_l, m_l, y="post_m")[0])
# Earlier baseline (g-4, g-3); cohorts 2009-2012
s_b = stacked(d, "CWI", base=(-4, -3)); _, p_b, m_b = fit(s_b)
rows.append(summ("Cohorts 2009-12: DiD + early baseline (g-4,g-3) covariate", s_b, p_b, m_b)[0])
s_b2 = s_b.copy(); s_b2.attrs["base"] = False      # same rows, same-window covariate
_, p_b2, m_b2 = fit(s_b2)
rows.append(summ("Cohorts 2009-12: DiD + same-window (g-2,g-1) covariate", s_b2, p_b2, m_b2)[0])
s_b3 = s_b.copy(); s_b3.attrs["pre_cov"] = False
_, p_b3, m_b3 = fit(s_b3)
rows.append(summ("Cohorts 2009-12: pure DiD (no gas covariate)", s_b3, p_b3, m_b3)[0])
pd.DataFrame(rows).to_csv(OUT / "bracket.csv", index=False)

# ---------------------------------------------------------------- D7 MDE + Holm
sub = {}
for name, m in {"IMD Q1": s["IMD"] == 1, "IMD Q5": s["IMD"] == 5, "CTB A": s["CTB"] == "A",
                "CTB D-F": s["CTB"].isin(["D", "E", "F"]), "Built 1930-72": s["PROP_AGE_BAND"] == 2,
                "Built 1973-99": s["PROP_AGE_BAND"] == 3}.items():
    m = m.to_numpy()
    est, se, nt, psi = att(D, s["dY"], p, m0, s["home_id"], m)
    sub[name] = dict(est=est, se=se, nt=nt, psi=psi, g=s["home_id"].to_numpy()[m], n=m.sum())
con = []
for a, b in [("IMD Q1", "IMD Q5"), ("CTB A", "CTB D-F"), ("Built 1930-72", "Built 1973-99")]:
    A, B = sub[a], sub[b]
    dif = A["est"] - B["est"]
    se = diff_se(A["psi"], B["psi"], A["g"], B["g"], A["n"], B["n"])
    con.append(dict(contrast=f"{a} minus {b}", diff=dif, se=se, lo=dif - 1.96 * se, hi=dif + 1.96 * se,
                    p=2 * stats.norm.sf(abs(dif / se)), mde_80=(1.96 + stats.norm.ppf(0.8)) * se,
                    att_a=A["est"], att_b=B["est"], n_a=A["nt"], n_b=B["nt"]))
con = pd.DataFrame(con)
order = con["p"].argsort().to_numpy()
holm = np.empty(len(con)); running = 0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (len(con) - rank) * con.loc[i, "p"]))
    holm[i] = running
con["p_holm"] = holm
con.to_csv(OUT / "contrasts.csv", index=False)

# ---------------------------------------------------------------- D8 absolute savings by early-baseline tercile
s_bk = stacked(d, "CWI", base=(-4, -3), kind="level"); _, p_k, m_k = fit(s_bk)
cuts = np.quantile(s_bk.loc[s_bk.D == 1, "base_kwh"], [1 / 3, 2 / 3])
terc = np.digitize(s_bk["base_kwh"], cuts)
kt = []
for k, lab in enumerate(["lowest", "middle", "highest"]):
    m = terc == k
    est, se, nt, _ = att(s_bk["D"], s_bk["dY"], p_k, m0=m_k, groups=s_bk["home_id"], mask=m)
    estl, sel, _, _ = att(s_b["D"], s_b["dY"], p_b, m_b, s_b["home_id"],
                          np.digitize(s_b["base_kwh"], cuts) == k)
    kt.append(dict(baseline_tercile=lab, mean_base_kwh_treated=float(s_bk.loc[m & (s_bk.D == 1), "base_kwh"].mean()),
                   kwh_att=est, kwh_se=se, kwh_lo=est - 1.96 * se, kwh_hi=est + 1.96 * se,
                   log_att=estl, log_se=sel, pct_saving=pct(estl), pct_lo=pct(estl + 1.96 * sel),
                   pct_hi=pct(estl - 1.96 * sel), n_treated=nt))
pd.DataFrame(kt).to_csv(OUT / "kwh_by_baseline.csv", index=False)

# ---------------------------------------------------------------- D6 joint event study + honest DiD
E = [-4, -3, -2, 0, 1, 2, 3, 4, 5]
phi, est = {}, {}
for e in E:
    se_ = event_pair(d, e)
    _, pe, me = fit(se_)
    b, sb, nt, psi = att(se_["D"], se_["dY"], pe, me, se_["home_id"])
    est[e] = (b, sb, nt)
    phi[e] = pd.Series(psi / len(psi), index=se_["home_id"].to_numpy()).groupby(level=0).sum()
allh = sorted(set().union(*[set(v.index) for v in phi.values()]))
Phi = np.column_stack([phi[e].reindex(allh, fill_value=0.0).to_numpy() for e in E])
V = Phi.T @ Phi
beta = np.array([est[e][0] for e in E])
es = pd.DataFrame({"e": E, "att": beta, "se": np.sqrt(np.diag(V)), "n_treated": [est[e][2] for e in E]})
es["lo"], es["hi"] = es.att - 1.96 * es.se, es.att + 1.96 * es.se
es["pct_saving"] = pct(es.att)
es.to_csv(OUT / "event_study_joint.csv", index=False)
pd.DataFrame(V, index=E, columns=E).to_csv(OUT / "event_study_vcov.csv")
pre_idx = [0, 1, 2]
wald = float(beta[pre_idx] @ np.linalg.solve(V[np.ix_(pre_idx, pre_idx)], beta[pre_idx]))
pre_test = dict(wald=wald, df=3, p=float(stats.chi2.sf(wald, 3)))

import honestdid as hd
l_vec = np.array([0, 0, 0.5, 0.5, 0, 0])          # average of e = 2, 3 (main post window)
orig = hd.constructOriginalCS(beta, V, 3, 6, l_vec=l_vec)
rm = hd.createSensitivityResults_relativeMagnitudes(beta, V, 3, 6, Mbarvec=np.arange(0, 2.01, 0.25), l_vec=l_vec)
sd_ = hd.createSensitivityResults(beta, V, 3, 6, method="FLCI", Mvec=np.arange(0, 0.0301, 0.0025), l_vec=l_vec)
rm = pd.DataFrame(rm); sd_ = pd.DataFrame(sd_); orig = pd.DataFrame(orig) if not isinstance(orig, pd.DataFrame) else orig
rm.to_csv(OUT / "honest_rm.csv", index=False); sd_.to_csv(OUT / "honest_sd.csv", index=False); orig.to_csv(OUT / "honest_original.csv", index=False)


def breakdown(df, col):
    bad = df[(df["lb"] <= 0) & (df["ub"] >= 0)]
    return float(bad[col].min()) if len(bad) else None


hd_info = dict(target="average of event-time coefficients e=2,3 (log points)",
               target_est=float(l_vec @ beta[3:]), target_se=float(np.sqrt(l_vec @ V[3:, 3:] @ l_vec)),
               breakdown_Mbar_relative_magnitudes=breakdown(rm, "Mbar"),
               breakdown_M_smoothness=breakdown(sd_, "M"), pre_trend_wald=pre_test,
               max_abs_pre_coef=float(np.abs(beta[pre_idx]).max()))
json.dump(dict(main=main, balance=bal_summary, honest=hd_info), open(OUT / "core_info.json", "w"), indent=1, default=float)
print(pd.DataFrame(rows).round(4).to_string()); print(con.round(4).to_string()); print(pd.DataFrame(kt).round(3).to_string())
print(es.round(4).to_string()); print(rm.round(4).to_string()); print(sd_.round(4).to_string()); print(orig)
print(json.dumps(dict(balance=bal_summary, honest=hd_info), indent=1, default=float))
