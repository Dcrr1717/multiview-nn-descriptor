"""
Plan C2 feasibility (exploratory) - cross-flow transfer of the chaotic vs
non-chaotic decision: train on all Kolmogorov snapshots, test on all RB
snapshots, and the reverse. No parameter value or observable is shared
between the two flows, so the amplitude proxy cannot transfer by
construction.

Usage:  python scripts/08_transfer.py      -> results/transfer.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsg.protocol import auc_w, make_clf  # noqa: E402

VIEWS = {"X1": "X1_", "X2": "X2_", "X3": "X3_", "B": "B_", "B_amp": "Bamp_"}

if __name__ == "__main__":
    K = pd.read_csv(ROOT / "data" / "features_kolmogorov_snapshot.csv")
    R = pd.read_csv(ROOT / "data" / "features_rb_snapshot.csv")
    out = {}
    for v, p in VIEWS.items():
        cols = [c for c in K.columns if c.startswith(p)]
        for name, (A, B) in {"kolmogorov->rb": (K, R), "rb->kolmogorov": (R, K)}.items():
            ya, yb = (A.regime == "turbulent").astype(int), (B.regime == "turbulent").astype(int)
            clf = make_clf().fit(A[cols].to_numpy(float), ya)
            out.setdefault(v, {})[name] = auc_w(yb.to_numpy(), clf.predict_proba(B[cols].to_numpy(float)))
        print(f"  {v:6s} " + "  ".join(f"{k} {a:.3f}" for k, a in out[v].items()))
    json.dump(out, open(ROOT / "results" / "transfer.json", "w"), indent=1)
