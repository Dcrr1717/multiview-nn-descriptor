"""Static figures (matplotlib). Colours: validated categorical palette for
views (fixed entity -> colour mapping) and an ordinal one-hue ramp for the
ordered regimes laminar -> transitional -> turbulent (plus marker shape)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

VIEW_COLORS = {"X3": "#2a78d6", "X1": "#eb6834", "X2": "#1baf7a", "B": "#eda100", "B_amp": "#e87ba4"}
VIEW_LABELS = {"X3": "X3 (fusion)", "X1": "X1 (graph)", "X2": "X2 (PH)",
               "B": "B (slope+stats)", "B_amp": "B_amp (amplitude)"}
REGIME_COLORS = {"laminar": "#86b6ef", "transitional": "#2a78d6", "turbulent": "#104281"}
REGIME_MARKERS = {"laminar": "o", "transitional": "s", "turbulent": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FLOW_TITLES = {"kolmogorov": "Kolmogorov flow", "rb": "Rayleigh-Benard convection"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
MAIN = ["X1", "X2", "X3", "B", "B_amp"]


def _title(flow):
    return FLOW_TITLES.get(flow.split("_")[0], flow) + ("" if "_" not in flow else
                                                         f" ({flow.split('_', 1)[1]})")


def roc_panel(P_prim, y_prim, classes, P_hard, y_hard, path, flow):
    from tsg.protocol import auc_w
    if len(classes) == 2:          # binary primary endpoint: one panel
        classes = classes[1:]
        cls_idx = [1]
    else:
        cls_idx = list(range(len(classes)))
    panels = [f"{c} vs rest" for c in classes] + (["transitional vs turbulent"] if P_hard else [])
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 0.3, 2.5), sharey=True, squeeze=False)
    axes = axes[0]
    for j, ttl in enumerate(panels):
        ax = axes[j]
        for v in MAIN:
            if j < len(classes):
                ci = cls_idx[j]
                yy, s = (y_prim == ci).astype(int), P_prim[v][:, ci]
                a = auc_w(yy, np.column_stack([1 - s, s]))
            else:
                yy, s = y_hard, P_hard[v][:, 1]
                a = auc_w(yy, P_hard[v])
            f, t, _ = roc_curve(yy, s)
            ax.plot(f, t, color=VIEW_COLORS[v], lw=2 if v == "X3" else 1.3,
                    label=f"{v} {a:.3f}", zorder=3 if v == "X3" else 2)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        ax.set_title(ttl if len(cls_idx) > 1 or j > 0 else ttl.replace("vs rest", "vs laminar"))
        ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none",
                  framealpha=0.9, handlelength=1.2, title="AUC (OOF)",
                  title_fontsize=7)
        ax.set_aspect("equal")
    axes[0].set_ylabel("True positive rate")
    fig.suptitle(f"{_title(flow)}: one-vs-rest ROC, leave-parameter-out OOF predictions", y=1.02)
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


def auc_violins(reps_prim, reps_hard, path, flow):
    panels = [(reps_prim, "Primary: Regime AUC")]
    if reps_hard is not None:
        panels.append((reps_hard, "Hard: transitional vs turbulent"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 2.6), squeeze=False)
    for ax, (reps, ttl) in zip(axes[0], panels):
        data = [np.asarray(reps[v]) for v in MAIN]
        parts = ax.violinplot(data, showextrema=False, widths=0.75)
        for b, v in zip(parts["bodies"], MAIN):
            b.set_facecolor(VIEW_COLORS[v])
            b.set_edgecolor("white")
            b.set_linewidth(2)
            b.set_alpha(0.85)
        for i, d in enumerate(data, 1):
            ax.plot([i, i], np.percentile(d, [2.5, 97.5]), color=INK, lw=1)
            ax.plot(i, np.mean(d), "o", color="white", mec=INK, ms=4, zorder=4)
        ax.set_xticks(range(1, len(MAIN) + 1))
        ax.set_xticklabels([VIEW_LABELS[v] for v in MAIN], rotation=25, ha="right")
        ax.set_title(ttl)
        lo = min(np.min(d) for d in data)
        ax.set_ylim(max(0.0, lo - 0.03), 1.005)
    axes[0][0].set_ylabel("AUC per CV repeat")
    fig.suptitle(f"{_title(flow)}: repeated leave-parameter-out CV (mean, 95% interval)", y=1.03)
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


def delta_forest(con_prim, con_hard, path, flow):
    from tsg.decision import MIN_GAIN
    keys = list(con_prim)
    panels = [(con_prim, "Primary")] + ([(con_hard, "Hard (transitional vs turbulent)")]
                                         if con_hard else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 2.3), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, (con, ttl) in zip(axes, panels):
        for i, kk in enumerate(keys):
            c = con[kk]
            yv = len(keys) - 1 - i
            ax.plot(c["ci"], [yv, yv], color=INK, lw=1.4)
            ax.plot(c["delta"], yv, "o", ms=6, color=INK if c["sig_bh"] else "white", mec=INK, zorder=3)
            ax.annotate(f"{c['delta']:+.3f}  p_BH={c['p_bh']:.3g}", (c["ci"][1], yv),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5,
                        color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axvline(MIN_GAIN, color=INK2, lw=0.8, ls="--")
        ax.set_title(ttl)
        ax.set_xlabel("Paired Delta-AUC (cluster bootstrap 95% CI)")
        xl = ax.get_xlim()
        ax.set_xlim(xl[0], xl[1] + 0.45 * (xl[1] - xl[0]))
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([k.replace("-", " - ") for k in keys[::-1]])
    fig.suptitle(f"{_title(flow)}: paired contrasts (filled = significant after BH; dashed = +{MIN_GAIN})",
                 y=1.04)
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


def embedding(df, views, path, flow):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, v in zip(axes, ("X3", "B", "B_amp")):
        X = StandardScaler().fit_transform(df[views[v]].to_numpy(float))
        Z = PCA(n_components=2, random_state=42).fit_transform(X) if X.shape[1] > 1 else \
            np.column_stack([X[:, 0], np.zeros(len(X))])
        if X.shape[1] == 2:
            Z = X
        for reg in REGIME_COLORS:
            m = (df["regime"] == reg).to_numpy()
            if m.any():
                ax.scatter(Z[m, 0], Z[m, 1], s=10, marker=REGIME_MARKERS[reg],
                           color=REGIME_COLORS[reg], edgecolor="white", linewidth=0.4, label=reg)
        ax.set_title(VIEW_LABELS[v] + (" - PCA" if X.shape[1] > 2 else ""))
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(frameon=False, loc="best", markerscale=1.3)
    fig.suptitle(f"{_title(flow)}: unsupervised 2D projection of each window (colour = regime)",
                 y=1.03)
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


def noise_curves(curves, path):
    """curves[(flow, endpoint)][view] = [(noise, auc, lo, hi), ...]"""
    keys = sorted(curves, key=lambda k: (k[0] != "kolmogorov", k[1] != "primary"))
    fig, axes = plt.subplots(1, len(keys), figsize=(2.2 * len(keys) + 0.4, 2.5), sharey=True,
                             squeeze=False)
    for ax, key in zip(axes[0], keys):
        for v in MAIN:
            if v not in curves[key]:
                continue
            pts = np.array(sorted(curves[key][v]))
            ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=VIEW_COLORS[v], alpha=0.15, lw=0)
            ax.plot(pts[:, 0], pts[:, 1], "-o", color=VIEW_COLORS[v], ms=3.5,
                    lw=2 if v == "X3" else 1.3, label=VIEW_LABELS[v], zorder=3 if v == "X3" else 2)
        ax.set_title(f"{FLOW_TITLES[key[0]].split(' ')[0]} - {key[1]}")
        ax.set_xlabel("Observation noise (x field std)")
    axes[0][0].set_ylabel("OOF AUC (95% CI)")
    axes[0][-1].legend(frameon=False, loc="lower left", fontsize=6.5)
    fig.suptitle("Robustness to observation noise (snapshot mode, leave-parameter-out)", y=1.03)
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


def sensitivity(res, path, flow):
    params = list(dict.fromkeys(r["param"] for r in res))
    fig, axes = plt.subplots(1, len(params), figsize=(1.6 * len(params) + 0.6, 2.4), sharey=True)
    for ax, p in zip(axes, params):
        rr = [r for r in res if r["param"] == p]
        x = np.arange(len(rr))
        ax.plot(x, [r["X3_primary"] for r in rr], "-o", color=VIEW_COLORS["X3"], ms=4, lw=2,
                label="X3 primary")
        ax.plot(x, [r["X3_hard"] for r in rr], "--s", color=VIEW_COLORS["X3"], ms=4, lw=1.3,
                mfc="white", label="X3 hard")
        ax.set_xticks(x)
        ax.set_xticklabels([r["value"] for r in rr], rotation=30, ha="right")
        ax.set_title(p)
        ax.set_xlim(-0.5, len(rr) - 0.5)
    axes[0].set_ylabel("Repeated-CV AUC (20 repeats)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"{_title(flow)}: sensitivity of X3 to descriptor choices (snapshot mode)", y=1.04)
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)
