"""Revision analysis, part 5 (deviation D12 in SPEC.md): does the council-tax gradient survive
holding property type fixed? Among treated homes, 70% of band A homes are flats or terraces
against 7% in bands D-F, where 52% are detached. Adds property-type indicators (semi-detached
as reference) to the best linear projection of deviation D9, on the main sample and on the
2009-12 cohorts with the earlier baseline; also reports subgroup ATTs by property type."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from prep import load, stacked, design
from drdid import nuisances, att
from rev_het_blp import zmat, dr_blp

OUT = Path(__file__).resolve().parents[1] / "outputs" / "revision"
d = load()
pct = lambda v: -(np.exp(v) - 1) * 100
TYPES = {"Detached": "Detached", "Bungalow": "Bungalow", "Flat": "Flat", "terrace": "Terraced (mid or end)"}


def type_dummies(s):
    t = s["PROP_TYPE"].astype(str)
    return pd.DataFrame({"Type: detached": (t == "Detached").astype(float), "Type: bungalow": (t == "Bungalow").astype(float),
                         "Type: flat": (t == "Flat").astype(float), "Type: terraced": t.str.contains("terrace").astype(float)})


out, sub = [], []
for lab, kw, base_col in [("main sample, joint + property type", {}, "pre_level"),
                          ("cohorts 2009-12, joint + property type (early baseline)", {"base": (-4, -3)}, "base_level")]:
    s = stacked(d, "CWI", **kw)
    p, m0 = nuisances(design(s), s["D"], s["dY"], s["home_id"])
    Z = pd.concat([zmat(s, base_col), type_dummies(s)], axis=1)
    out.append(dr_blp(s, p, m0, Z).assign(model=lab))
    if not kw:
        t = s["PROP_TYPE"].astype(str)
        for name, m in [("Detached", t == "Detached"), ("Semi-detached", t == "Semi detached"),
                        ("Terraced (mid or end)", t.str.contains("terrace")), ("Flat", t == "Flat"), ("Bungalow", t == "Bungalow")]:
            est, se, nt, _ = att(s["D"], s["dY"], p, m0, s["home_id"], m.to_numpy())
            sub.append(dict(group=name, att=est, se=se, pct_saving=pct(est), pct_lo=pct(est + 1.96 * se),
                            pct_hi=pct(est - 1.96 * se), n_treated=nt))
out = pd.concat(out, ignore_index=True); out.to_csv(OUT / "blp_with_type.csv", index=False)
pd.DataFrame(sub).to_csv(OUT / "subgroups_type.csv", index=False)
print(out.round(4).to_string()); print(pd.DataFrame(sub).round(3).to_string())
