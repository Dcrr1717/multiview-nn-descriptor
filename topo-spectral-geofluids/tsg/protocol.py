"""
Leak-free statistical protocol, ported from multiview-nn-descriptor
(phase11_leakfree.py, phase14_baselines.py, phase17_experiments.py) and
extended for spatio-temporally correlated samples.

Differences w.r.t. the neural-network study, all forced by the data:
* Samples (windows) from one trajectory are strongly autocorrelated, so the
  cross-validation is GROUPED: StratifiedGroupKFold with groups = control
  parameter value (Re or Ra). Every test fold holds out whole parameter
  values the classifier never saw ("leave-parameter-out").
* Uncertainty and paired tests use a STRATIFIED CLUSTER BOOTSTRAP over
  trajectories (the independent experimental units), not over windows.
Everything else is unchanged: StandardScaler + one-vs-rest logistic
regression (C = 1), weighted one-vs-rest ROC AUC ("Regime AUC", the
analogue of the Family AUC), repeated CV with percentile intervals,
paired bootstrap Delta-AUC, Wilcoxon on repeated-CV AUCs, Mann-Whitney /
Kruskal-Wallis feature tests and Benjamini-Hochberg FDR control.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def make_clf(kind="logreg"):
    if kind == "logreg":   # identical to the neural-network study
        return Pipeline([("sc", StandardScaler()),
                         ("clf", OneVsRestClassifier(
                             LogisticRegression(C=1.0, max_iter=500, random_state=42)))])
    if kind == "rf":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                        random_state=42, n_jobs=1))])
    raise ValueError(kind)


def auc_w(y, P, classes=None):
    """Weighted one-vs-rest AUC (binary: AUC of the positive-class column)."""
    classes = np.unique(y) if classes is None else classes
    try:
        if P.shape[1] == 2:
            return float(roc_auc_score(y, P[:, 1]))
        return float(roc_auc_score(y, P, multi_class="ovr", average="weighted",
                                   labels=classes))
    except ValueError:
        return np.nan


def _splitter(k, seed, grouped):
    return (StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed) if grouped
            else StratifiedKFold(n_splits=k, shuffle=True, random_state=seed))


def _fit_predict(X, y, tr, te, kind, n_cls):
    clf = make_clf(kind).fit(X[tr], y[tr])
    P = np.zeros((len(te), n_cls))
    P[:, clf.classes_] = clf.predict_proba(X[te])
    return P


def repeated_cv_auc(X, y, groups, k=4, repeats=60, kind="logreg", grouped=True):
    """Per-repeat mean of the per-fold Regime AUC (paper style).

    Returns (mean, p2.5, p97.5, per_repeat_array)."""
    X = np.asarray(X, float).reshape(len(y), -1)
    n_cls = len(np.unique(y))
    reps = []
    for rep in range(repeats):
        fa = []
        for tr, te in _splitter(k, rep, grouped).split(X, y, groups if grouped else None):
            if len(np.unique(y[te])) < n_cls:
                continue
            fa.append(auc_w(y[te], _fit_predict(X, y, tr, te, kind, n_cls)))
        reps.append(np.nanmean(fa))
    reps = np.array(reps)
    return float(reps.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)), reps


def oof_proba(X, y, groups, k=4, repeats=20, kind="logreg", grouped=True):
    """Out-of-fold class probabilities averaged over `repeats` CV partitions.
    Each window's probability only ever comes from models that did not see
    its parameter value, so the average is still leak-free."""
    X = np.asarray(X, float).reshape(len(y), -1)
    n_cls = len(np.unique(y))
    P = np.zeros((len(y), n_cls))
    for rep in range(repeats):
        for tr, te in _splitter(k, 1000 + rep, grouped).split(X, y, groups if grouped else None):
            P[te] += _fit_predict(X, y, tr, te, kind, n_cls)
    return P / repeats


def _boot_indices(y, clusters, rng):
    """Stratified cluster bootstrap: resample clusters within each class."""
    idx = []
    for c in np.unique(y):
        cl = np.unique(clusters[y == c])
        pick = rng.choice(cl, len(cl), replace=True)
        for p in pick:
            idx.append(np.flatnonzero(clusters == p))
    return np.concatenate(idx)


def bootstrap_auc(y, P, clusters, B=2000, seed=2024):
    rng = np.random.default_rng(seed)
    vals = [auc_w(y[i], P[i]) for i in (_boot_indices(y, clusters, rng) for _ in range(B))]
    vals = np.array(vals)
    return float(auc_w(y, P)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def paired_delta(y, Pa, Pb, clusters, B=2000, seed=2024):
    """Paired Delta-AUC (a - b) with stratified cluster bootstrap.
    Two-sided p = 2 min(P(d <= 0), P(d >= 0)), floored at 1/B."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        i = _boot_indices(y, clusters, rng)
        d.append(auc_w(y[i], Pa[i]) - auc_w(y[i], Pb[i]))
    d = np.array(d)
    d = d[np.isfinite(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": float(auc_w(y, Pa) - auc_w(y, Pb)), "boot_mean": float(d.mean()),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(max(p, 1.0 / len(d)))}


def group_permutation_test(X, y, groups, n_perm=200, k=4, repeats=3, seed=2024):
    """Permutation test of Regime AUC > chance: labels are permuted at the
    level of the parameter groups (so within-group dependence is kept)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    glab = np.array([np.bincount(y[groups == g]).argmax() for g in ug])
    obs = repeated_cv_auc(X, y, groups, k=k, repeats=repeats)[0]
    null = []
    for _ in range(n_perm):
        perm = dict(zip(ug, rng.permutation(glab)))
        yp = np.array([perm[g] for g in groups])
        null.append(repeated_cv_auc(X, yp, groups, k=k, repeats=repeats)[0])
    null = np.array(null)
    return {"auc_obs": obs, "null_mean": float(np.nanmean(null)),
            "null_p95": float(np.nanpercentile(null, 95)),
            "p": float((1 + np.sum(null >= obs)) / (n_perm + 1))}


def wilcoxon_repeats(a, b):
    """One-sided Wilcoxon signed-rank on paired repeated-CV AUCs (a > b).
    Descriptive only: repeats share data, so they are not independent."""
    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(a, b, alternative="greater").pvalue)


def bh(pvals, alpha=0.05):
    rej, padj, _, _ = multipletests(list(pvals), alpha=alpha, method="fdr_bh")
    return rej, padj


def feature_tests(F, names, y, units, classes, pair=None):
    """Kruskal-Wallis across regimes (epsilon^2 effect size) and, optionally,
    a Mann-Whitney test between two regimes (rank-biserial r), computed on
    per-trajectory means so that each independent trajectory counts once.
    BH-FDR within each family of tests."""
    uu = np.unique(units)
    Fm = np.array([F[units == u].mean(axis=0) for u in uu])
    ym = np.array([y[units == u][0] for u in uu])
    out = []
    for j, nm in enumerate(names):
        groups_ = [Fm[ym == c, j] for c in classes]
        try:
            H, p = kruskal(*groups_)
        except ValueError:
            H, p = 0.0, 1.0
        if not np.isfinite(p):          # constant feature (all ties): no evidence
            H, p = 0.0, 1.0
        n, kk = len(ym), len(classes)
        rec = {"feature": nm, "kw_H": float(H), "kw_p": float(p),
               "eps2": float(max(0.0, (H - kk + 1) / (n - kk)))}
        if pair is not None:
            a, b = Fm[ym == pair[0], j], Fm[ym == pair[1], j]
            try:
                U, pm = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                U, pm = len(a) * len(b) / 2, 1.0
            if not np.isfinite(pm):
                U, pm = len(a) * len(b) / 2, 1.0
            rec.update({"mw_p": float(pm), "rank_biserial": float(1 - 2 * U / (len(a) * len(b)))})
        out.append(rec)
    _, pk = bh([r["kw_p"] for r in out])
    for r, q in zip(out, pk):
        r["kw_q"] = float(q)
    if pair is not None:
        _, pm = bh([r["mw_p"] for r in out])
        for r, q in zip(out, pm):
            r["mw_q"] = float(q)
    return out
