"""
Generate the snapshot data set for one flow.

For every (control parameter, trajectory) pair of the design in
tsg/config.py:
  1. random initial condition (seeded), spin-up of t_spin time units;
  2. largest Lyapunov exponent over t_lyap (twin trajectory) and
     distance to the laminar/conductive state -> dynamical regime label;
  3. n_windows space-time windows of T_win snapshots every dt_sample,
     consecutive windows separated by `gap` time units.

Raw windows are stored as float32 in data/raw/<flow>/ (git-ignored,
~1 GB per flow); per-trajectory metadata in data/raw/<flow>/*.json.
Resumable: finished trajectories are skipped.

Usage:  python scripts/01_simulate.py kolmogorov|rb [--workers 4] [--ascending]
        (--ascending lets a second process work from the other end of the
        task list; finished trajectories are skipped, results are deterministic)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsg.config import FLOWS, SEED  # noqa: E402
from tsg.dynamics import lyapunov_twin  # noqa: E402
from tsg.solvers import KolmogorovFlow, RayleighBenard  # noqa: E402


def build(flow, p):
    c = FLOWS[flow]
    if flow == "kolmogorov":
        return KolmogorovFlow(p, N=c["N"], n_force=c["n_force"])
    return RayleighBenard(p, Pr=c["Pr"], Gamma=c["Gamma"], Nx=c["Nx"], Nz=c["Nz"])


def observe(flow, s, st):
    """Observed scalar field: vorticity (Kolmogorov) / temperature anomaly (RB)."""
    if flow == "kolmogorov":
        return s.ifft(st[0])
    return s.physical(st[1])


def run(task):
    flow, p, r = task
    c = FLOWS[flow]
    out = ROOT / "data" / "raw" / flow
    tag = f"p{p:g}_r{r}"
    if (out / f"{tag}.json").exists():
        return tag, "skip"
    seed = int(np.random.SeedSequence([SEED, r, int(round(float(p) * 100))]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    s = build(flow, p)
    t0 = time.time()
    if flow == "kolmogorov":
        st = (s.random_ic(rng, amp=c["ic_amp"], add_laminar=True),)
        ref = s.laminar()
    else:
        st = s.random_ic(rng, amp=c["ic_amp"])
        ref = None
    adv = (lambda x, T: (s.advance(x[0], T),)) if flow == "kolmogorov" else s.advance
    st = adv(st, c["t_spin"])
    lam, sem, st = lyapunov_twin(s, st, T=c["t_lyap"], tau=1.0, seed=seed)
    st = st if isinstance(st, tuple) else (st,)
    windows, glob = [], []
    for _ in range(c["n_windows"]):
        frames = []
        for _ in range(c["T_win"]):
            st = adv(st, c["dt_sample"])
            q = observe(flow, s, st)
            frames.append(q.astype(np.float32))
            d = s.diagnostics(st[0] if flow == "kolmogorov" else st)
            glob.append(d["D"] if flow == "kolmogorov" else d["Nu"])
        windows.append(np.stack(frames))
        st = adv(st, c["gap"])
    windows = np.stack(windows)
    if flow == "kolmogorov":
        q = observe(flow, s, st)
        dist = float(np.linalg.norm(q - ref) / np.linalg.norm(ref))
    else:
        dist = float(np.abs(observe(flow, s, st)).max())   # 0 for conduction
    glob = np.array(glob)
    meta = {"flow": flow, "param": float(p), "traj": int(r), "seed": int(seed),
            "lyap": lam, "lyap_sem": sem, "dist_ref": dist,
            "glob_mean": float(glob.mean()), "glob_cv": float(glob.std() / (abs(glob.mean()) + 1e-30)),
            "wall_s": round(time.time() - t0, 1), "shape": list(windows.shape)}
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{tag}.npy", windows)
    json.dump(meta, open(out / f"{tag}.json", "w"), indent=1)
    return tag, f"lyap={lam:+.4f}+-{sem:.4f} dist={dist:.2e} cv={meta['glob_cv']:.2e} ({meta['wall_s']}s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("flow", choices=list(FLOWS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ascending", action="store_true")
    a = ap.parse_args()
    c = FLOWS[a.flow]
    tasks = [(a.flow, p, r) for regime in c["design"] for p in c["design"][regime]
             for r in range(c["n_traj"])]
    # longest (most turbulent) runs first for better load balance
    tasks = sorted(tasks, key=lambda t: t[1] if a.ascending else -t[1])
    print(f"{a.flow}: {len(tasks)} trajectories", flush=True)
    with Pool(a.workers) as pool:
        for tag, msg in pool.imap_unordered(run, tasks):
            print(f"  {tag:14s} {msg}", flush=True)
