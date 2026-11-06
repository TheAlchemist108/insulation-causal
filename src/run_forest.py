"""Exploratory heterogeneity: causal forest on the differenced outcome, plus held-out calibration."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import statsmodels.api as sm
sys.path.insert(0, str(Path(__file__).parent))
from prep import load, stacked
from drdid import nuisances, att, SEED

OUT = Path(__file__).resolve().parents[1] / "outputs"
d = load()
s = stacked(d, "CWI")
HET = ["IMD", "CTB", "PROP_AGE_BAND", "PROP_TYPE", "FLOOR_AREA_BAND"]
CTRL = ["REGION", "NATION", "cohort"]
ctb_rank = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G+": 7}
Xh = pd.DataFrame({"IMD": s["IMD"].astype(float), "CTB": s["CTB"].map(ctb_rank),
                   "AGE": s["PROP_AGE_BAND"].astype(float), "AREA": s["FLOOR_AREA_BAND"].astype(float),
                   "pre_level": s["pre_level"]})
Xh = pd.concat([Xh, pd.get_dummies(s["PROP_TYPE"], prefix="type").astype(float)], axis=1)
W = pd.get_dummies(s[CTRL].astype(str)).astype(float)
Y, T, G = s["dY"].to_numpy(), s["D"].to_numpy(), s["home_id"].to_numpy()


def forest():
    return CausalForestDML(
        model_y=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED),
        model_t=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED),
        discrete_treatment=True, n_estimators=1000, min_samples_leaf=20, max_samples=0.45,
        cv=5, random_state=SEED)


cf = forest().fit(Y, T, X=Xh, W=W, groups=G)
tau = cf.effect(Xh)
tr = T == 1
s["tau"] = tau

# Variable importance
imp = pd.Series(cf.feature_importances_, index=Xh.columns).sort_values(ascending=False)
imp.to_csv(OUT / "forest_importance.csv", header=["importance"])

# Conditional (partial) association among treated: CATE ~ IMD + CTB + area + age + type
# Shows whether the deprivation gradient survives holding house size and wealth proxy fixed.
Z = Xh.loc[tr, ["IMD", "CTB", "AREA", "AGE", "pre_level"]]
blp = sm.OLS(tau[tr], sm.add_constant(Z)).fit()
blp_tab = pd.DataFrame({"coef": blp.params, "note": "descriptive projection of forest CATEs (no valid SE)"})
blp_tab.to_csv(OUT / "forest_projection.csv")

# Held-out calibration: fit forest on half the homes, rank the other half by predicted effect,
# then estimate DR-DiD ATT within terciles of predicted effect on the held-out half.
rng = np.random.default_rng(SEED)
homes = np.unique(G); half = set(rng.choice(homes, len(homes) // 2, replace=False))
A = np.array([g in half for g in G]); B = ~A
cfA = forest().fit(Y[A], T[A], X=Xh[A], W=W[A], groups=G[A])
tauB = cfA.effect(Xh[B])
cuts = np.quantile(tauB[T[B] == 1], [1 / 3, 2 / 3])
terc = np.digitize(tauB, cuts)                        # 0 = largest predicted saving (most negative)
sB = s[B].reset_index(drop=True)
from prep import design
XB = design(sB)
p, m0 = nuisances(XB, sB["D"], sB["dY"], sB["home_id"])
cal = []
for k, lab in enumerate(["largest predicted saving", "middle", "smallest predicted saving"]):
    m = terc == k
    est, se, nt, _ = att(sB["D"], sB["dY"], p, m0, sB["home_id"], m)
    cal.append(dict(tercile=lab, mean_pred=float(tauB[m & (T[B] == 1)].mean()), att=est, se=se,
                    lo=est - 1.96 * se, hi=est + 1.96 * se, n_treated=nt,
                    share_IMD_Q1=float((sB.loc[m & (sB.D == 1), "IMD"] == 1).mean()),
                    share_CTB_A=float((sB.loc[m & (sB.D == 1), "CTB"] == "A").mean())))
pd.DataFrame(cal).to_csv(OUT / "forest_calibration.csv", index=False)

gate = (s[tr].groupby("IMD")["tau"].mean().rename("mean_CATE_treated").to_frame())
gate["pct_saving"] = -(np.exp(gate["mean_CATE_treated"]) - 1) * 100
gate.to_csv(OUT / "forest_gate_imd.csv")
print(imp.round(3).to_string()); print(blp_tab.round(4).to_string())
print(pd.DataFrame(cal).round(4).to_string()); print(gate.round(4).to_string())
print("ATE on treated (forest):", tau[tr].mean())
