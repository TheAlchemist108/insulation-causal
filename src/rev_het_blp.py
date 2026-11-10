"""Best linear projection of the CATT with the doubly robust DiD score (deviations D9, D12)."""
import numpy as np
import pandas as pd
from scipy import stats

CTB_RANK = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G+": 7}


def zmat(s, base_col):
    return pd.DataFrame({"IMD quintile": s["IMD"].astype(float), "Council tax band": s["CTB"].map(CTB_RANK).astype(float),
                         "Floor area band": s["FLOOR_AREA_BAND"].astype(float), "Age band": s["PROP_AGE_BAND"].astype(float),
                         "Baseline ln gas": s[base_col].astype(float)})


def dr_blp(s, p, m0, Z):
    """Projection of the CATT on Z among treated homes; doubly robust DiD score,
    home-clustered sandwich standard errors (Semenova and Chernozhukov, 2021)."""
    D = s["D"].to_numpy(float); r = s["dY"].to_numpy() - m0; g = s["home_id"].to_numpy()
    Zc = Z - Z[D == 1].mean()                       # centre on treated means -> constant = ATT
    Zm = np.c_[np.ones(len(Zc)), Zc.to_numpy()]
    a = D - (1 - D) * p / (1 - p)
    A = (Zm * D[:, None]).T @ Zm
    b = (Zm * (a * r)[:, None]).sum(0)
    beta = np.linalg.solve(A, b)
    score = Zm * (a * r)[:, None] - (Zm * D[:, None]) * (Zm @ beta)[:, None]
    S = pd.DataFrame(score).groupby(g).sum().to_numpy()
    Ai = np.linalg.inv(A)
    V = Ai @ (S.T @ S) @ Ai
    se = np.sqrt(np.diag(V))
    names = ["Constant (ATT)"] + list(Z.columns)
    return pd.DataFrame({"term": names, "coef": beta, "se": se, "lo": beta - 1.96 * se, "hi": beta + 1.96 * se,
                         "p": 2 * stats.norm.sf(np.abs(beta / se))})
