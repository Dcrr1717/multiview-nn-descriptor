"""
Extract X1, X2, X3, B and B_amp for every sample of one flow and attach the
dynamical regime label of its trajectory.

  --mode snapshot (PRIMARY)  sample = first frame of each window, a single
                             instantaneous field; purely spatial views:
                             X1 = patch-coherence graph, X2 = sublevel PH,
                             B = 7 spatial statistics.
  --mode window (SECONDARY)  sample = the 16-frame space-time window:
                             X1 = climate network (regions co-varying in
                             time), X2 = sublevel PH averaged over 4 frames,
                             B = 7 spatial + 2 temporal statistics.
  --noise s                  i.i.d. Gaussian observation noise of std s (in
                             units of the sample std) added before anything
                             is computed (robustness / decision endpoint).

Writes data/features_<flow>_<mode>[_noise<s>].csv (one row per sample;
committed, so every statistic is reproducible without re-simulating).

Usage:  python scripts/02_features.py kolmogorov|rb [--mode snapshot] [--noise 0.0]
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
from tsg.baselines import (B_NAMES, B_SNAP_NAMES, BAMP_NAMES, baseline_B,  # noqa: E402
                           baseline_Bamp, baseline_snapshot)
from tsg.config import FLOWS, SEED, label_trajectory  # noqa: E402
from tsg.descriptors import (X1_NAMES, X2_NAMES, desc_x3, normalize_window,  # noqa: E402
                             x1_functional_graph, x1_patch_coherence, x2_snapshot_ph,
                             x2_sublevel_ph)


def sample_features(raw, flow, mode, noise=0.0, rng=None, **kw):
    """raw: (T, H, W) window. Returns x1, x2, x3, b, b_amp."""
    c = FLOWS[flow]
    theta, blocks = kw.get("theta", c["theta"]), kw.get("blocks", c["blocks"])
    min_pers, corr = kw.get("min_pers", c["min_pers"]), kw.get("corr", "pearson")
    if mode == "snapshot":
        raw = raw[0]
    if noise > 0:   # observation noise, std = noise x (sample std), in physical units
        raw = raw + noise * raw.std() * rng.standard_normal(raw.shape)
    w = normalize_window(raw)
    if mode == "snapshot":
        x1 = x1_patch_coherence(w, theta=theta, blocks=blocks, corr=corr,
                                max_n=kw.get("max_n", 80))
        x2 = x2_snapshot_ph(w, periodic=c["periodic"], min_pers=min_pers)
        b = baseline_snapshot(w, c["spec_kind"], c["slope_band"])
        ba = baseline_Bamp(raw[None], c["spec_kind"])
    else:
        x1 = x1_functional_graph(w, theta=theta, blocks=blocks, corr=corr)
        x2 = x2_sublevel_ph(w, periodic=c["periodic"], n_snap=kw.get("n_snap", c["n_snap"]),
                            min_pers=min_pers)
        b = baseline_B(w, c["spec_kind"], c["slope_band"])
        ba = baseline_Bamp(raw, c["spec_kind"])
    return x1, x2, desc_x3(x1, x2), b, ba


def process(args):
    path, flow, mode, noise = args
    meta = json.load(open(path.replace(".npy", ".json")))
    W = np.load(path, mmap_mode="r")
    rng = np.random.default_rng([SEED, meta["traj"], int(meta["param"] * 100), int(noise * 1000)])
    bnames = B_SNAP_NAMES if mode == "snapshot" else B_NAMES
    rows = []
    for i in range(W.shape[0]):
        x1, x2, x3, b, ba = sample_features(np.asarray(W[i], dtype=np.float64), flow, mode,
                                            noise, rng)
        row = {"flow": flow, "param": meta["param"], "traj": meta["traj"],
               "traj_id": f"{meta['param']:g}_{meta['traj']}", "window": i,
               "regime": label_trajectory(meta, flow), "lyap": meta["lyap"],
               "lyap_sem": meta["lyap_sem"], "dist_ref": meta["dist_ref"],
               "glob_cv": meta["glob_cv"]}
        row.update({f"X1_{n}": v for n, v in zip(X1_NAMES, x1)})
        row.update({f"X2_{n}": v for n, v in zip(X2_NAMES, x2)})
        row.update({f"X3_{j:02d}": v for j, v in enumerate(x3)})
        row.update({f"B_{n}": v for n, v in zip(bnames, b)})
        row.update({f"Bamp_{n}": v for n, v in zip(BAMP_NAMES, ba)})
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("flow", choices=list(FLOWS))
    ap.add_argument("--mode", choices=["snapshot", "window"], default="snapshot")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / a.flow / "*.npy")))
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.map(process, [(f, a.flow, a.mode, a.noise) for f in files])
                for r in rs]
    df = pd.DataFrame(rows).sort_values(["param", "traj", "window"]).reset_index(drop=True)
    suffix = "" if a.noise == 0 else f"_noise{a.noise:g}"
    out = ROOT / "data" / f"features_{a.flow}_{a.mode}{suffix}.csv"
    df.to_csv(out, index=False, float_format="%.8g")
    print(f"{out}: {len(df)} samples, {df.traj_id.nunique()} trajectories")
    print(df.groupby(["param", "regime"]).size().to_string())
