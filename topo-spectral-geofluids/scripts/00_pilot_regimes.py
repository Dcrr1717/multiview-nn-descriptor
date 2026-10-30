"""
Pilot bifurcation scan (descriptor-free) used to fix the regime bands.

For each control parameter value a single trajectory is integrated from a
random initial condition; we record the global observable, the distance to
the laminar/conductive state and the largest Lyapunov exponent. No spatial
descriptor is computed here: the scan only determines which parameter
values are laminar / transitional (non-chaotic, non-laminar) / turbulent
(chaotic), *before* any classification experiment is run.

Usage:  python scripts/00_pilot_regimes.py kolmogorov|rb [param ...] [--name suffix]
        (explicit params refine the scan; results go to regime_scan_<flow><suffix>.json)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsg.config import KOLMO, RB                     # noqa: E402
from tsg.dynamics import lyapunov_twin               # noqa: E402
from tsg.solvers import KolmogorovFlow, RayleighBenard  # noqa: E402


def run_kolmo(Re, t_spin=300.0, t_lyap=100.0, seed=7):
    s = KolmogorovFlow(Re, N=KOLMO["N"], n_force=KOLMO["n_force"])
    rng = np.random.default_rng(seed)
    wh = s.random_ic(rng, amp=KOLMO["ic_amp"], add_laminar=True)
    wl = s.laminar()
    D_L = Re / (2 * KOLMO["n_force"] ** 2)
    ts = []
    for _ in range(int(t_spin)):
        wh = s.advance(wh, 1.0)
        d = s.diagnostics(wh)
        ts.append([d["D"] / D_L, np.linalg.norm(s.ifft(wh) - wl) / np.linalg.norm(wl)])
    lam, sem, wh = lyapunov_twin(s, (wh,), T=t_lyap, tau=1.0, seed=seed)
    ts = np.array(ts)
    last = ts[-100:]
    return {"param": Re, "D_over_DL_mean": float(last[:, 0].mean()),
            "D_over_DL_cv": float(last[:, 0].std() / (last[:, 0].mean() + 1e-12)),
            "dist_laminar": float(last[-1, 1]), "lyap": lam, "lyap_sem": sem,
            "D_series": ts[:, 0].tolist()}


def run_rb(Ra, t_spin=300.0, t_lyap=100.0, seed=7):
    s = RayleighBenard(Ra, Pr=RB["Pr"], Gamma=RB["Gamma"], Nx=RB["Nx"], Nz=RB["Nz"])
    rng = np.random.default_rng(seed)
    st = s.random_ic(rng, amp=RB["ic_amp"])
    ts = []
    for _ in range(int(t_spin)):
        st = s.advance(st, 1.0)
        d = s.diagnostics(st)
        ts.append([d["Nu"], d["theta_rms"]])
    lam, sem, st = lyapunov_twin(s, st, T=t_lyap, tau=1.0, seed=seed)
    ts = np.array(ts)
    last = ts[-100:]
    return {"param": Ra, "Nu_mean": float(last[:, 0].mean()),
            "Nu_cv": float(last[:, 0].std() / (last[:, 0].mean() + 1e-12)),
            "theta_rms": float(last[-1, 1]), "lyap": lam, "lyap_sem": sem,
            "Nu_series": ts[:, 0].tolist()}


if __name__ == "__main__":
    args = sys.argv[1:]
    suffix = ""
    if "--name" in args:
        i = args.index("--name")
        suffix = "_" + args[i + 1]
        del args[i:i + 2]
    flow = args[0] if args else "kolmogorov"
    params = [float(x) for x in args[1:]] or (KOLMO["pilot_Re"] if flow == "kolmogorov"
                                               else RB["pilot_Ra"])
    fn = run_kolmo if flow == "kolmogorov" else run_rb
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(fn, params)
    out_dir = ROOT / "results" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / f"regime_scan_{flow}{suffix}.json", "w"), indent=1)
    for r in out:
        keys = [k for k in r if not k.endswith("series")]
        print("  ".join(f"{k}={r[k]:.4g}" for k in keys))
    print(f"done in {time.time() - t0:.0f}s")
