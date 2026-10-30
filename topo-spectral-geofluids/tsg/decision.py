"""
Pre-registered Go/No-Go decision rule (frozen in PROTOCOL.md before any
classification result was computed). Thresholds live here, in one place.
"""
CEILING = 0.98          # baseline OOF AUC at or above which an endpoint cannot discriminate
MIN_GAIN = 0.02         # minimum Delta-AUC(X3 - B) for a "clear" improvement
ALPHA = 0.05            # BH-adjusted significance level
REPL_TOL = 0.01         # replication: X3 >= max(X1, X2) - REPL_TOL


def flow_criteria(ep):
    """Criteria G1 (beats baseline), G2 (fusion beats each view) and the
    confound flag G4 on one endpoint's metrics block."""
    v, c = ep["views"], ep["contrasts"]
    xb, x1, x2 = c["X3-B"], c["X3-X1"], c["X3-X2"]
    sig = lambda d: d["ci"][0] > 0 and d["p_bh"] < ALPHA
    return {
        "baseline_auc": v["B"]["auc_oof"],
        "x3_auc": v["X3"]["auc_oof"],
        "ceiling": v["B"]["auc_oof"] >= CEILING,
        "G1_beats_baseline": bool(xb["delta"] >= MIN_GAIN and sig(xb)),
        "G2_fusion_beats_views": bool(sig(x1) and sig(x2)),
        "G4_beats_amplitude_proxy": bool(sig(c["X3-B_amp"])),
        "incremental_over_B": bool(sig(c["B+X3-B"])),
        "repl_direction": bool(xb["delta"] > 0 and
                               v["X3"]["auc_oof"] >= max(v["X1"]["auc_oof"], v["X2"]["auc_oof"]) - REPL_TOL),
    }


def decide(metrics):
    """Flow-level summary: criteria on every endpoint + the first informative
    (non-ceiling) endpoint in the pre-registered order primary -> hard."""
    out = {ep: flow_criteria(metrics[ep]) for ep in ("primary", "hard") if ep in metrics}
    informative = [ep for ep in ("primary", "hard") if ep in out and not out[ep]["ceiling"]]
    out["decision_endpoint"] = informative[0] if informative else None
    return out


def verdict(per_flow, primary_flow="kolmogorov", replication_flow="rb"):
    """Combine flow-level criteria (each already resolved to its decision
    endpoint, possibly on the noisy-observation data set) into GO / PLAN B / PLAN C.

    Replication status: 'replicated', 'not replicated' or 'non-informative'
    (every RB endpoint at ceiling). Only 'replicated' supports GO."""
    k, r = per_flow.get(primary_flow), per_flow.get(replication_flow)
    if k is None or k["criteria"] is None:
        return "PLAN C (NO-GO)", "every endpoint of the primary flow is at ceiling: no discrimination possible"
    g1, g2 = k["criteria"]["G1_beats_baseline"], k["criteria"]["G2_fusion_beats_views"]
    if r is None or r["criteria"] is None:
        rep_status = "non-informative"
    else:
        rep_status = "replicated" if r["criteria"]["repl_direction"] else "not replicated"
    if g1 and g2 and rep_status == "replicated":
        return "GO", "X3 clearly beats the baseline and both single views, and replicates in the second flow"
    if g1:
        fails = ([] if g2 else ["fusion does not beat both single views"]) + \
                ([] if rep_status == "replicated" else [f"replication status in the second flow: {rep_status}"])
        return "PLAN B (CONDITIONAL)", "X3 beats the baseline on the primary flow, but " + "; ".join(fails)
    return "PLAN C (NO-GO)", "X3 does not clearly beat the spectral-slope + statistics baseline on the primary flow"
