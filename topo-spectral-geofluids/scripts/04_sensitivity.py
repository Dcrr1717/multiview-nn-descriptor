"""
Sensitivity sweep (exploratory, PROTOCOL.md §10) — the analogue of
phase17_experiments.py of the network study. One descriptor choice is
varied at a time; the other view is taken from the main feature file.

  theta     in {0.2, 0.3, 0.4, 0.5, 0.6}       (X1 correlation threshold)
  blocks    in {4x4, 8x8, 16x16}               (X1 nodes; 16x16 uses all 256)
  corr      in {pearson, spearman}             (X1)
  min_pers  in {0.001, 0.01, 0.05}             (X2 persistence floor)
  x2        in {cubical sublevel, Vietoris-Rips point cloud (original desc_x2)}

Metric: repeated leave-parameter-out CV AUC of X3 (20 repeats) on the
primary and hard endpoints (snapshot mode, clean data).

Usage:  python scripts/04_sensitivity.py kolmogorov|rb [--plot-only]
"""
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsg import plots  # noqa: E402
from tsg.config import FLOWS  # noqa: E402
from tsg.descriptors import (desc_x3, normalize_window, patch_matrix,  # noqa: E402
                             x1_patch_coherence, x2_rips_pointcloud, x2_snapshot_ph)
from tsg.protocol import repeated_cv_auc  # noqa: E402

VARIANTS = ([("theta", t, {"theta": t}) for t in (0.2, 0.3, 0.4, 0.5, 0.6)] +
            [("blocks", f"{b}x{b}", {"blocks": (b, b), "max_n": b * b}) for b in (4, 8, 16)] +
            [("corr", c, {"corr": c}) for c in ("pearson", "spearman")] +
            [("min_pers", m, {"min_pers": m}) for m in (0.001, 0.01, 0.05)] +
            [("x2", x, {"x2": x}) for x in ("cubical", "rips")])


def features(args):
    path, flow, kw = args
    c = FLOWS[flow]
    W = np.load(path, mmap_mode="r")
    out = []
    for i in range(W.shape[0]):
        f = normalize_window(np.asarray(W[i, 0], dtype=np.float64))
        if "x2" in kw or "min_pers" in kw:
            if kw.get("x2") == "rips":
                out.append(x2_rips_pointcloud(patch_matrix(f, c["blocks"])))
            else:
                out.append(x2_snapshot_ph(f, c["periodic"], kw.get("min_pers", c["min_pers"])))
        else:
            out.append(x1_patch_coherence(f, theta=kw.get("theta", c["theta"]),
                                          blocks=kw.get("blocks", c["blocks"]),
                                          corr=kw.get("corr", "pearson"), max_n=kw.get("max_n", 80)))
    return out


def baseline_ref(flow):
    p = ROOT / "results" / f"{flow}_snapshot" / "metrics.json"
    if not p.exists():
        return None
    m = json.load(open(p))
    return {ep: m[ep]["views"]["B"]["auc_cv_mean"] for ep in ("primary", "hard") if ep in m}


if __name__ == "__main__":
    flow = sys.argv[1] if len(sys.argv) > 1 else "kolmogorov"
    if "--plot-only" in sys.argv:
        out = ROOT / "results" / f"{flow}_snapshot"
        plots.sensitivity(json.load(open(out / "sensitivity.json")), out / "fig_sensitivity.png", flow,
                          baseline_ref(flow))
        sys.exit(0)
    df = pd.read_csv(ROOT / "data" / f"features_{flow}_snapshot.csv")
    X1 = df[[c for c in df.columns if c.startswith("X1_")]].to_numpy(float)
    X2 = df[[c for c in df.columns if c.startswith("X2_")]].to_numpy(float)
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / flow / "*.npy")))
    tags = [Path(f).stem for f in files]
    order = {t: i for i, t in enumerate(tags)}
    key = [f"p{p:g}_r{r}" for p, r in zip(df["param"], df["traj"])]
    classes = ["laminar", "transitional", "turbulent"]
    design = [c for c in classes if c in FLOWS[flow]["design"]]
    res = []
    with Pool(4) as pool:
        for name, val, kw in VARIANTS:
            per_file = pool.map(features, [(f, flow, kw) for f in files])
            V = np.array([per_file[order[k]][w] for k, w in zip(key, df["window"])])
            x1 = X1 if ("x2" in kw or "min_pers" in kw) else V
            x2 = V if ("x2" in kw or "min_pers" in kw) else X2
            x3 = np.array([desc_x3(a, b) for a, b in zip(x1, x2)])
            rec = {"param": name, "value": str(val)}
            eps = [("primary", df["regime"].isin(design).to_numpy())]
            if df[df.regime == "transitional"]["param"].nunique() >= 2:
                eps.append(("hard", df["regime"].isin(["transitional", "turbulent"]).to_numpy()))
            for ep, mask in eps:
                cl = [c for c in classes if c in set(df["regime"][mask])]
                y = df["regime"][mask].map({c: i for i, c in enumerate(cl)}).to_numpy()
                g = df["param"][mask].to_numpy()
                k = int(min(4, min(len(np.unique(g[y == i])) for i in range(len(cl)))))
                rec[f"X3_{ep}"] = repeated_cv_auc(x3[mask], y, g, k=k, repeats=20)[0]
                rec[f"view_{ep}"] = repeated_cv_auc(V[mask], y, g, k=k, repeats=20)[0]
            res.append(rec)
            print(f"  {name:9s} {str(val):9s} " + "  ".join(f"{k} {v:.3f}" for k, v in rec.items()
                                                           if k not in ("param", "value")), flush=True)
    out = ROOT / "results" / f"{flow}_snapshot"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "sensitivity.json", "w"), indent=1)
    plots.sensitivity(res, out / "fig_sensitivity.png", flow, baseline_ref(flow))
