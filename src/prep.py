"""Build stacked cohort datasets from the NEED 2026 anonymised 50k file (see SPEC.md)."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "anon2026_50k.csv"
YEARS = range(2005, 2025)
COHORTS = range(2007, 2013)
CAT_COLS = ["PROP_TYPE", "PROP_AGE_BAND", "FLOOR_AREA_BAND", "CTB", "IMD", "NATION", "REGION"]


def load(path=RAW) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["home_id"] = np.arange(len(d))
    d["NATION"] = np.where(d["REGION"].str.startswith("W"), "Wales", "England")
    d["IMD"] = d["IMD_BAND_ENG"].fillna(d["IMD_BAND_WALES"]).astype("Int64")
    d["CTB"] = d["COUNCIL_TAX_BAND"].replace({"H": "G+", "I": "G+", "G": "G+"})
    return d


def _valid(d, fuel, years):
    pre = "GasValFlag" if fuel == "G" else "ElecValFlag"
    ok = np.ones(len(d), bool)
    for y in years:
        col = f"{pre}{y}"
        ok &= (d[col] == "V").to_numpy() if col in d else False
    return ok


def _y(d, fuel, y, kind):
    v = d[f"{fuel}cons{y}"].astype(float)
    return np.log(v) if kind == "log" else v


def stacked(d, treat="CWI", pre=(-2, -1), post=(2, 3), kind="log", fuel="G",
            pre_cov=True, base=None) -> pd.DataFrame:
    """One row per (home, cohort). D=1 for homes treated in that cohort.

    base: optional tuple of event years (e.g. (-4, -3)) used for an earlier baseline
    covariate ``base_level`` (mean log gas). When given, cohorts whose base years fall
    before 2005 are dropped, gas must be valid in the base years too, and ``design``
    uses ``base_level`` in place of ``pre_level`` (revision deviation D5).
    """
    other = "LI" if treat == "CWI" else "CWI"
    g_all = d[d["MAIN_HEAT_FUEL"] == 1]
    control = g_all[(g_all["CWI_FLAG"] == 0) & (g_all["LI_FLAG"] == 0)]
    out = []
    for g in COHORTS:
        yrs_pre = [g + k for k in pre]
        yrs_post = [g + k for k in post]
        yrs_base = [g + k for k in base] if base else []
        if min(yrs_pre + yrs_post + yrs_base) < 2005 or max(yrs_pre + yrs_post) > 2024:
            continue
        t = g_all[(g_all[f"{treat}_FLAG"] == 1) & (g_all[f"{treat}_DATE"] == g)]
        od = t[f"{other}_DATE"]
        t = t[~((od >= g - 2) & (od <= g + 3))]          # no other measure in window
        s = pd.concat([t.assign(D=1), control.assign(D=0)])
        s = s[_valid(s, fuel, yrs_pre + yrs_post + yrs_base)].copy()
        pre_m = np.mean([_y(s, fuel, y, kind) for y in yrs_pre], axis=0)
        post_m = np.mean([_y(s, fuel, y, kind) for y in yrs_post], axis=0)
        s["dY"] = post_m - pre_m
        s["post_m"] = post_m
        if base:
            s["base_level"] = np.mean([_y(s, "G", y, "log") for y in yrs_base], axis=0)
            s["base_kwh"] = np.mean([_y(s, "G", y, "level") for y in yrs_base], axis=0)
        s["pre_level"] = np.mean([_y(s, "G" if fuel == "G" else fuel, y, "log") for y in yrs_pre], axis=0)
        s["pre_kwh"] = np.mean([_y(s, fuel, y, "level") for y in yrs_pre], axis=0)
        s["cohort"] = g
        out.append(s)
    s = pd.concat(out, ignore_index=True)
    s.attrs["pre_cov"] = pre_cov
    s.attrs["base"] = bool(base)
    return s


def event_pair(d, e, treat="CWI", fuel="G") -> pd.DataFrame:
    """Outcome ln(y_{g+e}) - ln(y_{g-1}) for the event study (no pre-level covariate)."""
    return stacked(d, treat=treat, pre=(-1,), post=(e,), kind="log", fuel=fuel, pre_cov=False)


def design(s: pd.DataFrame, extra=("cohort",)) -> pd.DataFrame:
    X = pd.get_dummies(s[CAT_COLS + list(extra)].astype(str), drop_first=False).astype(float)
    if s.attrs.get("pre_cov", True):
        col = "base_level" if s.attrs.get("base", False) else "pre_level"
        X[col] = s[col].to_numpy()
    return X
