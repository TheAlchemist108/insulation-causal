"""Revision analyses, part 2: inferential heterogeneity (deviations D9-D10 in SPEC.md).

D9  Best linear projection (BLP) of the CATT on covariates, using the doubly robust
    DiD score with home-clustered sandwich standard errors. Run on the main sample
    (same-window baseline) and on cohorts 2009-2012 with the earlier (g-4, g-3) baseline.
D10 Repeated sample splitting (K splits): a causal forest fitted on half the homes ranks the
    other half; on the held-out half we estimate GATES (terciles of predicted saving),
    CLAN (covariate means, top vs bottom tercile) and a targeting operator characteristic
    and its area (AUTOC, adapted to the ATT; Yadlowsky et al. 2025). Split results are
    aggregated with the median rule of Chernozhukov et al.: per-split 97.5% intervals,
    medians of the bounds, p-value = min(1, 2 x median p).
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from econml.dml import CausalForestDML
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
sys.path.insert(0, str(Path(__file__).parent))
from prep import load, stacked, design
from drdid import nuisances, att, diff_se, SEED

OUT = Path(__file__).resolve().parents[1] / "outputs" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
d = load()
from rev_het_blp import CTB_RANK, zmat, dr_blp      # functions unchanged; moved for reuse (D12)
pct = lambda v: -(np.exp(v) - 1) * 100


# ---------------------------------------------------------------- D9 BLP
blp = []
s = stacked(d, "CWI"); X = design(s)
p, m0 = nuisances(X, s["D"], s["dY"], s["home_id"])
Z = zmat(s, "pre_level")
for lab, cols in [("main sample, IMD only", ["IMD quintile"]),
                  ("main sample, joint (same-window baseline)", list(Z.columns))]:
    blp.append(dr_blp(s, p, m0, Z[cols]).assign(model=lab))
sb = stacked(d, "CWI", base=(-4, -3)); Xb = design(sb)
pb, mb = nuisances(Xb, sb["D"], sb["dY"], sb["home_id"])
Zb = zmat(sb, "base_level")
for lab, cols in [("cohorts 2009-12, IMD only", ["IMD quintile"]),
                  ("cohorts 2009-12, joint (early baseline g-4,g-3)", list(Zb.columns))]:
    blp.append(dr_blp(sb, pb, mb, Zb[cols]).assign(model=lab))
blp = pd.concat(blp, ignore_index=True)
blp.to_csv(OUT / "blp.csv", index=False)
print(blp.round(4).to_string())

# ---------------------------------------------------------------- D10 repeated splits
HETX = lambda q: pd.concat([pd.DataFrame({"IMD": q["IMD"].astype(float), "CTB": q["CTB"].map(CTB_RANK),
                                          "AGE": q["PROP_AGE_BAND"].astype(float),
                                          "AREA": q["FLOOR_AREA_BAND"].astype(float), "pre_level": q["pre_level"]}),
                            pd.get_dummies(q["PROP_TYPE"], prefix="type").astype(float)], axis=1)
Xh = HETX(s).to_numpy()
W = pd.get_dummies(s[["REGION", "NATION", "cohort"]].astype(str)).astype(float).to_numpy()
Y, T, G = s["dY"].to_numpy(), s["D"].to_numpy(), s["home_id"].to_numpy()
homes = np.unique(G)
Z975 = stats.norm.ppf(1 - 0.025 / 2)
QGRID = np.round(np.arange(0.1, 1.01, 0.1), 2)


def att_w(D, r, p, mask, wt):
    D, r, p, wt = D[mask], r[mask], p[mask], wt[mask]
    w = (1 - D) * p / (1 - p)
    return np.sum(wt * D * r) / np.sum(wt * D) - np.sum(wt * w * r) / np.sum(wt * w)


gates, clan, autoc = [], [], []
t0 = time.time()
for k in range(K):
    rng = np.random.default_rng(SEED + k)
    inA = np.isin(G, rng.choice(homes, len(homes) // 2, replace=False)); inB = ~inA
    cf = CausalForestDML(model_y=HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=SEED),
                         model_t=HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED),
                         discrete_treatment=True, n_estimators=400, min_samples_leaf=20, max_samples=0.45,
                         cv=3, random_state=SEED + k)
    cf.fit(Y[inA], T[inA], X=Xh[inA], W=W[inA], groups=G[inA])
    sB = s[inB].reset_index(drop=True); sB.attrs = s.attrs
    S = -cf.effect(Xh[inB])                                     # priority = predicted saving
    pB, mB = nuisances(design(sB), sB["D"], sB["dY"], sB["home_id"], seed=SEED + k)
    DB, rB, gB = sB["D"].to_numpy(), sB["dY"].to_numpy() - mB, sB["home_id"].to_numpy()
    cuts = np.quantile(S[DB == 1], [1 / 3, 2 / 3])
    grp = 2 - np.digitize(S, cuts)                              # 0 = largest predicted saving
    store = {}
    for j, lab in enumerate(["G1 largest predicted saving", "G2 middle", "G3 smallest predicted saving"]):
        m = grp == j
        est, se, nt, psi = att(DB, sB["dY"], pB, mB, gB, m)
        store[j] = (est, psi, gB[m], m.sum())
        gates.append(dict(split=k, group=lab, att=est, se=se, lo=est - Z975 * se, hi=est + Z975 * se,
                          p=2 * stats.norm.sf(abs(est / se)), n_treated=nt))
    dif = store[2][0] - store[0][0]                             # smallest minus largest saving (log points)
    sdf = diff_se(store[2][1], store[0][1], store[2][2], store[0][2], store[2][3], store[0][3])
    gates.append(dict(split=k, group="G3 minus G1", att=dif, se=sdf, lo=dif - Z975 * sdf, hi=dif + Z975 * sdf,
                      p=2 * stats.norm.sf(abs(dif / sdf)), n_treated=np.nan))
    tr = DB == 1
    for var, f in [("Share IMD Q1", lambda q: (q["IMD"] == 1).astype(float)),
                   ("Share council tax A", lambda q: (q["CTB"] == "A").astype(float)),
                   ("Mean IMD quintile", lambda q: q["IMD"].astype(float)),
                   ("Mean council tax band (A=1)", lambda q: q["CTB"].map(CTB_RANK).astype(float)),
                   ("Mean baseline gas, kWh", lambda q: q["pre_kwh"].astype(float)),
                   ("Mean floor area band", lambda q: q["FLOOR_AREA_BAND"].astype(float))]:
        v = f(sB).to_numpy()
        top, bot = v[tr & (grp == 0)], v[tr & (grp == 2)]
        dd = bot.mean() - top.mean(); sdd = np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot))
        clan.append(dict(split=k, variable=var, top=top.mean(), bottom=bot.mean(), diff=dd, se=sdd,
                         lo=dd - Z975 * sdd, hi=dd + Z975 * sdd, p=2 * stats.norm.sf(abs(dd / sdd)) if sdd > 0 else 1.0))
    # AUTOC (saving scale): TOC(q) = saving among top-q priority minus overall saving
    hu, hidx = np.unique(gB, return_inverse=True)

    def autoc_of(wt):
        base = -att_w(DB, rB, pB, np.ones(len(DB), bool), wt)
        tocs = []
        for q in QGRID:
            thr = np.quantile(S[tr], 1 - q)
            tocs.append(-att_w(DB, rB, pB, S >= thr, wt) - base)
        return np.mean(tocs), tocs

    a_hat, tocs = autoc_of(np.ones(len(DB)))
    boots = []
    brng = np.random.default_rng(SEED + 1000 + k)
    for _ in range(300):
        boots.append(autoc_of(brng.exponential(1.0, len(hu))[hidx])[0])
    sa = np.std(boots, ddof=1)
    autoc.append(dict(split=k, autoc=a_hat, se=sa, lo=a_hat - Z975 * sa, hi=a_hat + Z975 * sa,
                      p=2 * stats.norm.sf(abs(a_hat / sa)), **{f"toc_{q}": t for q, t in zip(QGRID, tocs)}))
    print(f"split {k + 1}/{K} done, {time.time() - t0:.0f}s", flush=True)

gates, clan, autoc = pd.DataFrame(gates), pd.DataFrame(clan), pd.DataFrame(autoc)
gates.to_csv(OUT / "gates_splits.csv", index=False); clan.to_csv(OUT / "clan_splits.csv", index=False)
autoc.to_csv(OUT / "autoc_splits.csv", index=False)


ga =gates.groupby("group", sort=False).agg(att=("att", "median"), lo=("lo", "median"), hi=("hi", "median"),
                                           p_median=("p", "median"), n_treated=("n_treated", "median")).reset_index()
ga["p_adj"] = np.minimum(1, 2 * ga["p_median"])
ga["pct_saving"] = np.where(ga.group.str.startswith("G3 minus"), np.nan, pct(ga.att))
ga["pct_lo"] = np.where(ga.group.str.startswith("G3 minus"), np.nan, pct(ga.hi))
ga["pct_hi"] = np.where(ga.group.str.startswith("G3 minus"), np.nan, pct(ga.lo))
ca = clan.groupby("variable", sort=False).agg(top=("top", "median"), bottom=("bottom", "median"), diff=("diff", "median"),
                                             lo=("lo", "median"), hi=("hi", "median"), p_median=("p", "median")).reset_index()
ca["p_adj"] = np.minimum(1, 2 * ca["p_median"])
aa = dict(K=K, autoc_median=float(autoc.autoc.median()), lo=float(autoc.lo.median()), hi=float(autoc.hi.median()),
          p_adj=float(min(1, 2 * autoc.p.median())), units="log points of saving (positive = targeting by forest finds larger savings)",
          toc_median={str(q): float(autoc[f"toc_{q}"].median()) for q in QGRID})
ga.to_csv(OUT / "gates.csv", index=False); ca.to_csv(OUT / "clan.csv", index=False)
json.dump(aa, open(OUT / "autoc.json", "w"), indent=1)
print(ga.round(4).to_string()); print(ca.round(4).to_string()); print(json.dumps(aa, indent=1))
