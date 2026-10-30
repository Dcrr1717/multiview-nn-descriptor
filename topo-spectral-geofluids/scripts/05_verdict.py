"""
Apply the pre-registered decision rule (PROTOCOL.md §9, tsg/decision.py)
across flows and endpoints; build the noise-robustness figure and the
summary tables used in REPORT.md.

Usage:  python scripts/05_verdict.py
Writes  results/checkpoint_verdict.json, results/summary_tables.md,
        results/fig_noise_robustness.png
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsg import plots  # noqa: E402
from tsg.decision import flow_criteria, verdict  # noqa: E402

FLOWS = ["kolmogorov", "rb"]
NOISES = [0, 0.25, 0.5, 1.0]
DECISION_NOISE = 0.5
VIEWS = ["X1", "X2", "X3", "B", "B_amp", "B+X3", "B_amp+X3"]


def load(flow, noise=0, mode="snapshot"):
    p = ROOT / "results" / (f"{flow}_{mode}" + (f"_noise{noise:g}" if noise else "")) / "metrics.json"
    return json.load(open(p)) if p.exists() else None


def resolve(flow):
    """First non-ceiling endpoint in the order clean-primary, clean-hard,
    noisy-primary, noisy-hard."""
    trail = []
    for noise in (0, DECISION_NOISE):
        m = load(flow, noise)
        if m is None:
            trail.append(f"noise={noise}: missing")
            continue
        for ep in ("primary", "hard"):
            if ep not in m:
                continue
            c = flow_criteria(m[ep])
            name = f"{'clean' if noise == 0 else f'noisy({noise:g})'}-{ep}"
            trail.append(f"{name}: B={c['baseline_auc']:.3f}{' (ceiling)' if c['ceiling'] else ''}")
            if not c["ceiling"]:
                return {"endpoint": name, "criteria": c, "trail": trail}
    return {"endpoint": None, "criteria": None, "trail": trail}


def table(m, ep):
    v, c = m[ep]["views"], m[ep]["contrasts"]
    rows = ["| View | Regime AUC, CV mean [95% interval] | OOF AUC [cluster-bootstrap 95% CI] |",
            "|---|---|---|"]
    for k in VIEWS:
        x = v[k]
        rows.append(f"| {k} | {x['auc_cv_mean']:.3f} [{x['auc_cv_interval'][0]:.3f}, "
                    f"{x['auc_cv_interval'][1]:.3f}] | {x['auc_oof']:.3f} "
                    f"[{x['auc_oof_ci'][0]:.3f}, {x['auc_oof_ci'][1]:.3f}] |")
    rows += ["", "| Contrast | ΔAUC | 95% CI | p | p (BH) |", "|---|---|---|---|---|"]
    for k, d in c.items():
        rows.append(f"| {k} | {d['delta']:+.3f} | [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | "
                    f"{d['p']:.4f} | {d['p_bh']:.4f}{' *' if d['sig_bh'] else ''} |")
    if "permutation" in m[ep]:
        rows += ["", "| Permutation test (vs chance) | AUC | null 95th pct | p |", "|---|---|---|---|"]
        for k, d in m[ep]["permutation"].items():
            rows.append(f"| {k} | {d['auc_obs']:.3f} | {d['null_p95']:.3f} | {d['p']:.4f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    per_flow = {f: resolve(f) for f in FLOWS}
    v, why = verdict(per_flow)
    out = {"verdict": v, "reason": why, "per_flow": per_flow}
    json.dump(out, open(ROOT / "results" / "checkpoint_verdict.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

    md = []
    for f in FLOWS:
        for noise in NOISES:
            for mode in ("snapshot", "window"):
                m = load(f, noise, mode)
                if m is None:
                    continue
                for ep in ("primary", "hard"):
                    if ep not in m:
                        continue
                    md.append(f"### {f} · {mode} · noise {noise:g} · {ep}  "
                              f"(n = {m[ep]['n_windows']} samples, {m[ep]['n_traj']} trajectories, "
                              f"{m[ep]['n_groups']} parameter values, k = {m[ep]['k']})\n")
                    md.append(table(m, ep) + "\n")
    (ROOT / "results" / "summary_tables.md").write_text("\n".join(md))

    curves = {}
    for f in FLOWS:
        for noise in NOISES:
            m = load(f, noise)
            if m is None:
                continue
            for ep in ("primary", "hard"):
                if ep not in m:
                    continue
                for k in ("X1", "X2", "X3", "B", "B_amp"):
                    x = m[ep]["views"][k]
                    curves.setdefault((f, ep), {}).setdefault(k, []).append(
                        (noise, x["auc_oof"], *x["auc_oof_ci"]))
    if curves:
        plots.noise_curves(curves, ROOT / "results" / "fig_noise_robustness.png")
