"""Cross-fitted doubly robust DiD (Sant'Anna & Zhao 2020; Chang 2020), clustered by home."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

SEED = 20261106


def nuisances(X, D, Y, groups, folds=5, seed=SEED):
    X = np.asarray(X); D = np.asarray(D); Y = np.asarray(Y)
    p = np.zeros(len(D)); m0 = np.zeros(len(D))
    kf = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X, D, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             early_stopping=True, random_state=seed)
        clf.fit(X[tr], D[tr]); p[te] = clf.predict_proba(X[te])[:, 1]
        c = tr[D[tr] == 0]
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                            early_stopping=True, random_state=seed)
        reg.fit(X[c], Y[c]); m0[te] = reg.predict(X[te])
    return np.clip(p, 0.01, 0.99), m0


def att(D, Y, p, m0, groups, mask=None):
    """Normalised DR-DiD ATT on units in mask; returns (att, se, n_treated)."""
    D = np.asarray(D, float); Y = np.asarray(Y); groups = np.asarray(groups)
    if mask is None:
        mask = np.ones(len(D), bool)
    D, Y, p, m0, groups = D[mask], Y[mask], p[mask], m0[mask], groups[mask]
    r = Y - m0
    w = (1 - D) * p / (1 - p)
    a1 = np.sum(D * r) / D.sum()
    a0 = np.sum(w * r) / w.sum()
    est = a1 - a0
    n = len(D)
    psi = D * (r - a1) / D.mean() - w * (r - a0) / w.mean()
    cl = np.bincount(np.unique(groups, return_inverse=True)[1], weights=psi)
    se = np.sqrt(np.sum(cl ** 2)) / n
    return est, se, int(D.sum()), psi


def diff_se(psi_a, psi_b, groups_a, groups_b, n_a, n_b):
    """SE of a difference of two subgroup ATTs (disjoint subgroups, shared control homes)."""
    ga, gb = np.asarray(groups_a), np.asarray(groups_b)
    allg = np.unique(np.concatenate([ga, gb]))
    idx = {g: i for i, g in enumerate(allg)}
    s = np.zeros(len(allg))
    np.add.at(s, [idx[g] for g in ga], psi_a / n_a)
    np.add.at(s, [idx[g] for g in gb], -psi_b / n_b)
    return np.sqrt(np.sum(s ** 2))
