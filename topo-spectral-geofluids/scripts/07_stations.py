"""
Plan C1 - sparse station observations.

Features: for each snapshot sample (frame 0 of each window), the field is
read at M fixed random stations; X2s (lower-star PH on the station Delaunay
complex), BS (semivariogram + moments baseline) and BSamp (raw variance)
are computed.

Analysis: the same leak-free protocol as scripts/03_classify.py (leave-
parameter-out grouped CV, OOF probabilities, stratified cluster bootstrap,
BH over the contrast family) with views X2s, BS, BSamp and BS+X2s.

Usage:  python scripts/07_stations.py kolmogorov|rb [--M 128] [--noise 0] [--net-seed 0]
                                       [--raw-dir data/raw] [--tag NAME] [--quick]
Writes  data/features_<flow>_stations<M>[_noise<s>][_net<k>].csv and
        results/<flow>_stations<M>[...]/metrics.json
"""
import argparse
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
from tsg.config import FLOWS, SEED, label_trajectory  # noqa: E402
from tsg.descriptors import X2_NAMES  # noqa: E402
from tsg.protocol import (bh, bootstrap_auc, oof_proba, paired_delta,  # noqa: E402
                          repeated_cv_auc)
from tsg.stations import BS_NAMES, make_baseline, station_network, x2_stations  # noqa: E402

REGIMES = ["laminar", "transitional", "turbulent"]
CONTRASTS = [("X2s", "BS"), ("X2s", "BSamp"), ("BS+X2s", "BS")]
GEOM = {"kolmogorov": {"extent": (2 * np.pi, 2 * np.pi)},
        "rb": {"extent": (1.0, FLOWS["rb"]["Gamma"])}}


def network(flow, M, net_seed, shape):
    c = FLOWS[flow]
    return station_network(shape, M, c["periodic"], [SEED, 777, M, net_seed], GEOM[flow]["extent"])


def process(args):
    path, flow, M, noise, net_seed, raw_dir = args
    meta = json.load(open(path.replace(".npy", ".json")))
    W = np.load(path, mmap_mode="r")
    shape = W.shape[2:]
    (iy, ix), pos, tri = network(flow, M, net_seed, shape)
    bs = make_baseline(pos, GEOM[flow]["extent"], FLOWS[flow]["periodic"])
    rng = np.random.default_rng([SEED, meta["traj"], int(meta["param"] * 100), int(noise * 1000), M])
    rows = []
    for i in range(W.shape[0]):
        raw = np.asarray(W[i, 0], dtype=np.float64)[iy, ix]
        if noise > 0:
            raw = raw + noise * raw.std() * rng.standard_normal(raw.shape)
        v = (raw - raw.mean()) / (raw.std() + 1e-30)
        x2 = x2_stations(v, tri, FLOWS[flow]["min_pers"])
        b = bs(raw)
        row = {"flow": flow, "param": meta["param"], "traj": meta["traj"],
               "traj_id": f"{meta['param']:g}_{meta['traj']}", "window": i,
               "regime": label_trajectory(meta, flow), "M": M, "noise": noise}
        row.update({f"X2s_{n}": x for n, x in zip(X2_NAMES, x2)})
        row.update({f"BS_{n}": x for n, x in zip(BS_NAMES, b)})
        row["BSamp_log_var"] = float(np.log10(raw.var() + 1e-30))
        rows.append(row)
    return rows


def run_endpoint(df, classes, R, n_oof, n_boot, label):
    cols = lambda p: [c for c in df.columns if c.startswith(p)]
    views = {"X2s": cols("X2s_"), "BS": cols("BS_"), "BSamp": cols("BSamp_")}
    views["BS+X2s"] = views["BS"] + views["X2s"]
    y = df["regime"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    g, cl = df["param"].to_numpy(), df["traj_id"].to_numpy()
    k = int(min(4, min(len(np.unique(g[y == i])) for i in range(len(classes)))))
    res, P = {"classes": classes, "k": k, "n": int(len(df)), "views": {}}, {}
    for name, cc in views.items():
        X = df[cc].to_numpy(float)
        m, lo, hi, _ = repeated_cv_auc(X, y, g, k=k, repeats=R)
        P[name] = oof_proba(X, y, g, k=k, repeats=n_oof)
        a, blo, bhi = bootstrap_auc(y, P[name], cl, B=n_boot)
        res["views"][name] = {"auc_cv_mean": m, "auc_oof": a, "auc_oof_ci": [blo, bhi]}
        print(f"  [{label}] {name:7s} CV {m:.3f}  OOF {a:.3f} [{blo:.3f},{bhi:.3f}]", flush=True)
    con = {f"{a}-{b}": paired_delta(y, P[a], P[b], cl, B=n_boot) for a, b in CONTRASTS}
    _, padj = bh([c["p"] for c in con.values()])
    for (kk, c), q in zip(con.items(), padj):
        c["p_bh"] = float(q)
        print(f"  [{label}] D({kk}) = {c['delta']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
              f"p_BH={q:.3f}", flush=True)
    res["contrasts"] = con
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("flow", choices=list(FLOWS))
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--net-seed", type=int, default=0)
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    name = (f"{a.flow}_stations{a.M}" + (f"_noise{a.noise:g}" if a.noise else "") +
            (f"_net{a.net_seed}" if a.net_seed else "") + (f"_{a.tag}" if a.tag else ""))
    files = sorted(glob.glob(os.path.join(a.raw_dir, a.flow, "*.npy")))
    with Pool(4) as pool:
        rows = [r for rs in pool.map(process, [(f, a.flow, a.M, a.noise, a.net_seed, a.raw_dir)
                                               for f in files]) for r in rs]
    df = pd.DataFrame(rows).sort_values(["param", "traj", "window"]).reset_index(drop=True)
    df.to_csv(ROOT / "data" / f"features_{name}.csv", index=False, float_format="%.8g")
    design = [c for c in REGIMES if c in FLOWS[a.flow]["design"]]
    df = df[df["regime"].isin(design)].reset_index(drop=True)
    R, n_oof, n_boot = (10, 5, 300) if a.quick else (60, 20, 2000)
    print(f"{name}: {len(df)} samples", flush=True)
    out = {"flow": a.flow, "M": a.M, "noise": a.noise, "net_seed": a.net_seed,
           "primary": run_endpoint(df, design, R, n_oof, n_boot, "primary")}
    hard = df[df["regime"].isin(["transitional", "turbulent"])].reset_index(drop=True)
    if all(hard[hard.regime == c]["param"].nunique() >= 2 for c in ("transitional", "turbulent")):
        out["hard"] = run_endpoint(hard, ["transitional", "turbulent"], R, n_oof, n_boot, "hard")
    d = ROOT / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "metrics.json", "w"), indent=1)
    print(f"saved {d}")
