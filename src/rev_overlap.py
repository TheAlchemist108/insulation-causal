"""Revision analysis, part 4 (deviation D4, continued): sensitivity of the main estimate to how
low propensities are handled. The main specification clips fitted propensities at 0.01, which
leaves residual weighted imbalance in pre-1930 homes (outputs/revision/balance.csv).

R1  clip at 0.001 instead of 0.01
R2  drop pre-1930 homes (age band 1; most have solid walls and almost none received CWI)
R3  trim: drop rows whose fitted propensity is below 0.01 instead of clipping them
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from prep import load, stacked, design
import drdid
from drdid import att

OUT = Path(__file__).resolve().parents[1] / "outputs" / "revision"
d = load()
pct = lambda v: -(np.exp(v) - 1) * 100


def raw_nuisances(X, D, Y, G, lo):
    """drdid.nuisances with a different clipping floor (same models, folds and seed)."""
    return drdid.nuisances(X, D, Y, G, clip=lo)


def row(label, s, p, m0, mask=None):
    est, se, nt, _ = att(s["D"], s["dY"], p, m0, s["home_id"], mask)
    return dict(spec=label, att=est, se=se, pct_saving=pct(est), pct_lo=pct(est + 1.96 * se), pct_hi=pct(est - 1.96 * se),
                n_treated=nt)


rows = []
s = stacked(d, "CWI"); X = design(s)
p1, m1 = raw_nuisances(X, s["D"], s["dY"], s["home_id"], 0.001)
rows.append(row("R1 clip propensities at 0.001", s, p1, m1))
s2 = s[s["PROP_AGE_BAND"] != 1].reset_index(drop=True); s2.attrs = s.attrs
p2, m2 = drdid.nuisances(design(s2), s2["D"], s2["dY"], s2["home_id"])
rows.append(row("R2 exclude pre-1930 homes", s2, p2, m2))
p3, m3 = raw_nuisances(X, s["D"], s["dY"], s["home_id"], 1e-6)
rows.append(row("R3 trim rows with propensity < 0.01", s, p3, m3, mask=(p3 >= 0.01)))
pd.DataFrame(rows).to_csv(OUT / "overlap_sensitivity.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())
