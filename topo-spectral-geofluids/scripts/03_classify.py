"""
Regime-classification experiment and statistics for one flow (pre-registered
in PROTOCOL.md).

Endpoints
  primary : 3-class Regime AUC (laminar / transitional / turbulent)
  hard    : binary AUC transitional vs turbulent (the non-trivial contrast)
Views     : X1, X2, X3 (descriptor), B (spectral slope + simple stats),
            B_amp (trivial amplitude / control-parameter proxy),
            B+X3 and B_amp+X3 (incremental value of X3)
CV        : StratifiedGroupKFold, groups = control-parameter value
            (leave-parameter-out), k = 4, 60 repeats
Inference : stratified cluster bootstrap over trajectories (B = 2000) for
            AUC CIs and paired Delta-AUC; group-level permutation test vs
            chance; BH-FDR over each endpoint's contrast family; Wilcoxon on
            repeated-CV AUCs (descriptive); Kruskal-Wallis / Mann-Whitney
            feature tests on per-trajectory means; X1 x X2 redundancy.
Controls  : B_amp trivial baseline, leakage demonstration (naive window-level
            CV), random-forest classifier, observation-noise data sets.

Usage:  python scripts/03_classify.py kolmogorov|rb [--mode snapshot|window] [--noise s]
                                       [--features CSV --tag name] [--quick] [--no-extras]
Reads data/features_<flow>_<mode>[_noise<s>].csv, writes results/<flow>_<mode>[_noise<s>]/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsg import plots  # noqa: E402
from tsg.config import FLOWS  # noqa: E402
from tsg.decision import decide  # noqa: E402
from tsg.protocol import (bh, bootstrap_auc, feature_tests, group_permutation_test,  # noqa: E402
                          oof_proba, paired_delta, repeated_cv_auc, wilcoxon_repeats)

REGIMES = ["laminar", "transitional", "turbulent"]
CONTRASTS = [("X3", "X1"), ("X3", "X2"), ("X3", "B"), ("X3", "B_amp"), ("B+X3", "B")]


def view_matrix(df):
    cols = lambda p: [c for c in df.columns if c.startswith(p)]
    v = {"X1": cols("X1_"), "X2": cols("X2_"), "X3": cols("X3_"),
         "B": cols("B_"), "B_amp": cols("Bamp_")}
    v["B+X3"] = v["B"] + v["X3"]
    v["B_amp+X3"] = v["B_amp"] + v["X3"]
    return v


def run_endpoint(df, views, classes, R, n_oof, n_boot, n_perm, label):
    y = df["regime"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = df["param"].to_numpy()
    clusters = df["traj_id"].to_numpy()
    k = int(min(4, min(len(np.unique(groups[y == i])) for i in range(len(classes)))))
    res = {"classes": classes, "n_windows": int(len(df)), "n_traj": int(len(np.unique(clusters))),
           "n_groups": int(len(np.unique(groups))), "k": k, "views": {}}
    P, reps = {}, {}
    for name, cols in views.items():
        X = df[cols].to_numpy(float)
        m, lo, hi, rr = repeated_cv_auc(X, y, groups, k=k, repeats=R)
        P[name] = oof_proba(X, y, groups, k=k, repeats=n_oof)
        a, blo, bhi = bootstrap_auc(y, P[name], clusters, B=n_boot)
        reps[name] = rr
        res["views"][name] = {"auc_cv_mean": m, "auc_cv_interval": [lo, hi],
                              "auc_oof": a, "auc_oof_ci": [blo, bhi]}
        print(f"  [{label}] {name:9s} CV-AUC {m:.3f} [{lo:.3f},{hi:.3f}]  OOF-AUC {a:.3f} "
              f"[{blo:.3f},{bhi:.3f}]", flush=True)
    # paired contrasts + BH
    con = {}
    for a, b in CONTRASTS:
        con[f"{a}-{b}"] = paired_delta(y, P[a], P[b], clusters, B=n_boot)
    rej, padj = bh([c["p"] for c in con.values()])
    for (key, c), r_, q in zip(con.items(), rej, padj):
        c["p_bh"] = float(q)
        c["sig_bh"] = bool(r_)
        print(f"  [{label}] D({key:9s}) = {c['delta']:+.3f} CI [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
              f"p={c['p']:.4f} p_BH={q:.4f}", flush=True)
    res["contrasts"] = con
    res["wilcoxon_repeats"] = {f"X3>{b}": wilcoxon_repeats(reps["X3"], reps[b]) for b in ("X1", "X2", "B")}
    if n_perm:
        res["permutation"] = {}
        for name in ("X1", "X2", "X3", "B"):
            X = df[views[name]].to_numpy(float)
            res["permutation"][name] = group_permutation_test(X, y, groups, n_perm=n_perm, k=k)
            print(f"  [{label}] perm {name}: p={res['permutation'][name]['p']:.4f} "
                  f"(null 95% = {res['permutation'][name]['null_p95']:.3f})", flush=True)
    return res, P, reps, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flow")
    ap.add_argument("--mode", choices=["snapshot", "window"], default="snapshot")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--perm", type=int, default=None, help="override the number of permutations")
    ap.add_argument("--label", default="regime",
                    help="label column: regime (pre-registered, amended) | regime_frozen_rule | "
                         "regime_steady_first (label-sensitivity analyses)")
    ap.add_argument("--features", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-extras", action="store_true", help="skip RF / leakage / feature tests")
    a = ap.parse_args()
    t0 = time.time()
    R, n_oof, n_boot, n_perm = (10, 5, 300, 20) if a.quick else (60, 20, 2000, 200)
    n_perm = n_perm if a.perm is None else a.perm
    name = f"{a.flow}_{a.mode}" + (f"_noise{a.noise:g}" if a.noise else "")
    fpath = Path(a.features) if a.features else ROOT / "data" / f"features_{name}.csv"
    df = pd.read_csv(fpath)
    if a.label != "regime":
        df["regime"] = df[a.label]
        name += "_" + a.label.replace("regime_", "label_")
    a.tag = a.tag or name
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # classes = the regimes of the pre-registered design; samples whose dynamical
    # label is outside the design (e.g. a 'transitional' RB trajectory) are excluded
    design = [c for c in REGIMES if c in FLOWS[a.flow]["design"]]
    excluded = df[~df["regime"].isin(design)]
    df = df[df["regime"].isin(design)].reset_index(drop=True)
    views = view_matrix(df)
    present = [c for c in design if c in set(df["regime"])]
    print(f"{a.tag}: {len(df)} samples, regimes "
          f"{df.groupby('regime').traj_id.nunique().to_dict()} trajectories", flush=True)

    metrics = {"flow": a.flow, "mode": a.mode, "noise": a.noise, "tag": a.tag,
               "features": fpath.name,
               "settings": {"cv_repeats": R, "oof_repeats": n_oof, "bootstrap": n_boot,
                            "permutations": n_perm}}
    metrics["excluded_trajectories"] = sorted(excluded["traj_id"].unique().tolist())
    metrics["design"] = (df.groupby(["regime", "param"]).traj_id.nunique()
                         .reset_index().rename(columns={"traj_id": "n_traj"}).to_dict("records"))

    prim, P_prim, reps_prim, y_prim = run_endpoint(df, views, present, R, n_oof, n_boot, n_perm,
                                                   "primary")
    metrics["primary"] = prim
    hard_df = df[df["regime"].isin(["transitional", "turbulent"])].reset_index(drop=True)
    has_hard = all(hard_df[hard_df.regime == c]["param"].nunique() >= 2
                   for c in ("transitional", "turbulent"))
    P_hard = reps_hard = y_hard = None
    if has_hard:
        hard, P_hard, reps_hard, y_hard = run_endpoint(hard_df, views, ["transitional", "turbulent"],
                                                       R, n_oof, n_boot, n_perm, "hard")
        metrics["hard"] = hard

    if not a.no_extras:
        y = y_prim
        groups, clusters = df["param"].to_numpy(), df["traj_id"].to_numpy()
        k = prim["k"]
        # leakage demonstration: naive window-level CV ignores trajectories
        metrics["leakage_demo"] = {}
        for name in ("X3", "B"):
            X = df[views[name]].to_numpy(float)
            naive = repeated_cv_auc(X, y, groups, k=5, repeats=R // 3, grouped=False)[0]
            metrics["leakage_demo"][name] = {"naive_window_cv": naive,
                                             "grouped_cv": prim["views"][name]["auc_cv_mean"]}
            print(f"  [leakage] {name}: naive {naive:.3f} vs grouped "
                  f"{prim['views'][name]['auc_cv_mean']:.3f}", flush=True)
        # second classifier (non-linear)
        metrics["random_forest"] = {}
        for name in ("X1", "X2", "X3", "B", "B_amp"):
            for ep, d_, yy in (("primary", df, y_prim), ("hard", hard_df, y_hard))[:1 + has_hard]:
                X = d_[views[name]].to_numpy(float)
                m_ = repeated_cv_auc(X, yy, d_["param"].to_numpy(), k=k, repeats=max(5, R // 6),
                                     kind="rf")
                metrics["random_forest"].setdefault(name, {})[ep] = m_[0]
            print(f"  [RF] {name}: {metrics['random_forest'][name]}", flush=True)
        # feature-level non-parametric tests (per-trajectory means)
        names = views["X1"] + views["X2"] + views["B"]
        metrics["feature_tests"] = feature_tests(df[names].to_numpy(float), names,
                                                 df["regime"].to_numpy(), clusters, present,
                                                 pair=("transitional", "turbulent") if has_hard
                                                 else (present[0], present[-1]))
        # X1 x X2 redundancy (Spearman on per-trajectory means, BH)
        tm = df.groupby("traj_id")[views["X1"] + views["X2"]].mean()
        rho, pv = [], []
        for c1 in views["X1"]:
            for c2 in views["X2"]:
                r_, p_ = spearmanr(tm[c1], tm[c2])
                if np.isfinite(r_):
                    rho.append(abs(r_))
                    pv.append(p_)
        rej, _ = bh(pv)
        metrics["redundancy_x1_x2"] = {"n_pairs": len(rho), "median_abs_rho": float(np.median(rho)),
                                       "max_abs_rho": float(np.max(rho)),
                                       "frac_sig_bh": float(np.mean(rej)),
                                       "frac_abs_rho_gt_0.6": float(np.mean(np.array(rho) > 0.6))}

    metrics["decision"] = decide(metrics)
    metrics["runtime_s"] = round(time.time() - t0)
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=1)
    plots.roc_panel(P_prim, y_prim, present, P_hard, y_hard, out_dir / "fig_roc.png", a.tag)
    plots.auc_violins(reps_prim, reps_hard, out_dir / "fig_auc_repeats.png", a.tag)
    plots.delta_forest(prim["contrasts"], metrics["hard"]["contrasts"] if has_hard else None,
                       out_dir / "fig_delta_auc.png", a.tag)
    plots.embedding(df, views, out_dir / "fig_embedding.png", a.tag)
    print(json.dumps(metrics["decision"], indent=1))
    print(f"saved {out_dir} in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
