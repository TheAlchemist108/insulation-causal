"""Revision analysis, part 3 (deviation D11 in SPEC.md): conventional two-way fixed effects (TWFE)
dynamic specifications on our data, for comparison with Penasco and Anadon (2023), who used a
TWFE-type dynamic DiD on an earlier NEED release (gas years 2005-2017) and found fading savings.

T1 "broad": all gas-heated homes, gas years 2005-2017 (valid years only, unbalanced panel);
   event-time dummies -5..+5 (endpoints binned, -1 omitted) for CWI and, separately, for LI,
   using each home's recorded install year whenever it falls; home and year fixed effects.
T2 "our units": the CWI-only cohorts 2007-2012 and never-insulated homes used in the main
   design, same years and same TWFE specification.
Outcome ln(gas). Standard errors clustered by home.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from prep import load

OUT = Path(__file__).resolve().parents[1] / "outputs" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
d = load()
YRS = range(2005, 2018)
KS = [k for k in range(-5, 6) if k != -1]

g = d[d["MAIN_HEAT_FUEL"] == 1]
long = []
for y in YRS:
    v = g[g[f"GasValFlag{y}"] == "V"]
    long.append(pd.DataFrame({"home": v["home_id"].to_numpy(), "year": y, "lny": np.log(v[f"Gcons{y}"].astype(float)).to_numpy(),
                              "cwi": v["CWI_DATE"].to_numpy(), "li": v["LI_DATE"].to_numpy()}))
L = pd.concat(long, ignore_index=True)


def dummies(L, col, prefix):
    k = (L["year"] - L[col]).clip(-5, 5)
    out = {}
    for j in KS:
        out[f"{prefix}_{j}"] = ((k == j) & L[col].notna()).astype(float).to_numpy()
    return out


def within(M, h, t, tol=1e-10, it=500):
    """Two-way demeaning by alternating projections."""
    M = M.copy()
    hi = pd.factorize(h)[0]; ti = pd.factorize(t)[0]
    nh = 1.0 / np.bincount(hi)[:, None]; nt = 1.0 / np.bincount(ti)[:, None]
    for _ in range(it):
        prev = M.copy()
        M -= (nh * np.stack([np.bincount(hi, weights=M[:, c]) for c in range(M.shape[1])], 1))[hi]
        M -= (nt * np.stack([np.bincount(ti, weights=M[:, c]) for c in range(M.shape[1])], 1))[ti]
        if np.max(np.abs(M - prev)) < tol:
            break
    return M


def twfe(L, label, with_li=True):
    regs = dummies(L, "cwi", "cwi")
    if with_li:
        regs.update(dummies(L, "li", "li"))
    names = list(regs)
    M = np.column_stack([L["lny"].to_numpy()] + [regs[n] for n in names])
    Mw = within(M, L["home"].to_numpy(), L["year"].to_numpy())
    y, X = Mw[:, 0], Mw[:, 1:]
    keep = X.std(0) > 0
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    S = pd.DataFrame(X * e[:, None]).groupby(L["home"].to_numpy()).sum().to_numpy()
    nc, n, kx = S.shape[0], len(y), X.shape[1]
    V = XtX_inv @ (S.T @ S) @ XtX_inv * nc / (nc - 1)
    se = np.sqrt(np.diag(V))
    res = pd.DataFrame({"term": names, "coef": b, "se": se})
    res = res[res.term.str.startswith("cwi_")].copy()
    res["e"] = res.term.str.replace("cwi_", "").astype(int)
    res["pct_saving"] = -(np.exp(res.coef) - 1) * 100
    res["lo"], res["hi"] = res.coef - 1.96 * res.se, res.coef + 1.96 * res.se
    res["spec"] = label
    res["n_obs"], res["n_homes"] = n, nc
    return res.sort_values("e")


# T1: broad, Penasco-Anadon-like
t1 = twfe(L, "T1 broad: all gas-heated homes, CWI and LI event dummies")
# T2: our units only (CWI-only cohorts 2007-12 with no LI in g-2..g+3, and never-insulated homes)
ours = g[((g["CWI_FLAG"] == 1) & g["CWI_DATE"].between(2007, 2012) &
          ~((g["LI_DATE"] >= g["CWI_DATE"] - 2) & (g["LI_DATE"] <= g["CWI_DATE"] + 3))) |
         ((g["CWI_FLAG"] == 0) & (g["LI_FLAG"] == 0))]["home_id"]
L2 = L[L["home"].isin(set(ours))].copy()
L2.loc[L2["li"].notna() & ~L2["cwi"].notna(), "li"] = np.nan
t2 = twfe(L2, "T2 our units: CWI-only cohorts 2007-12 + never insulated", with_li=True)
res = pd.concat([t1, t2])
res.to_csv(OUT / "twfe_event.csv", index=False)
print(res[["spec", "e", "coef", "se", "pct_saving", "n_obs", "n_homes"]].round(4).to_string())
