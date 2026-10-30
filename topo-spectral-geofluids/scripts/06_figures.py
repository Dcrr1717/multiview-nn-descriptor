"""
Descriptive figures:
  results/fig_regime_scan.png   pilot bifurcation diagrams (descriptor-free)
  results/fig_views_<flow>.png  one snapshot per regime with its two views:
                                sublevel-set persistence diagram (X2) and the
                                patch-coherence graph adjacency (X1)
Usage:  python scripts/06_figures.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsg import plots  # noqa: E402
from tsg.config import FLOWS  # noqa: E402
from tsg.descriptors import normalize_window, patch_matrix, sublevel_persistence  # noqa: E402

plt = plots.plt


def regime_scan():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex="col")
    for j, flow in enumerate(("kolmogorov", "rb")):
        paths = sorted(glob.glob(str(ROOT / "results" / "pilot" / f"regime_scan_{flow}*.json")))
        if not paths:
            continue
        d = sorted([r for p in paths for r in json.load(open(p))], key=lambda r: r["param"])
        x = np.array([r["param"] for r in d])
        g = np.array([r["D_over_DL_mean" if flow == "kolmogorov" else "Nu_mean"] for r in d])
        lam = np.array([r["lyap"] for r in d])
        sem = np.array([r["lyap_sem"] for r in d])
        design = FLOWS[flow]["design"] or {}
        for ax in axes[:, j]:
            for reg, vals in design.items():
                ax.axvspan(min(vals) * 0.95, max(vals) * 1.05, color=plots.REGIME_COLORS[reg],
                           alpha=0.18, lw=0)
            ax.set_xscale("log")
        axes[0, j].plot(x, g, "-o", color=plots.INK, ms=3.5, lw=1.2)
        axes[0, j].set_ylabel("D / D_laminar" if flow == "kolmogorov" else "Nu")
        axes[0, j].set_title(plots.FLOW_TITLES[flow])
        axes[1, j].errorbar(x, lam, yerr=2 * sem, fmt="o", color=plots.INK, ms=3.5, lw=1, capsize=2)
        axes[1, j].axhline(0, color=plots.INK2, lw=0.8)
        axes[1, j].set_ylabel("largest Lyapunov exp. (±2 SEM)")
        axes[1, j].set_xlabel("Re" if flow == "kolmogorov" else "Ra")
        for reg in design:
            axes[0, j].plot([], [], "s", color=plots.REGIME_COLORS[reg], alpha=0.5, label=reg)
        axes[0, j].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Descriptor-free pilot scans; shaded = design bands of each regime", y=1.01)
    fig.savefig(ROOT / "results" / "fig_regime_scan.png")
    fig.savefig(ROOT / "results" / "fig_regime_scan.pdf")
    plt.close(fig)


def views(flow):
    fpath = ROOT / "data" / f"features_{flow}_snapshot.csv"
    if not fpath.exists():
        return
    df = pd.read_csv(fpath)
    c = FLOWS[flow]
    regs = [r for r in ("laminar", "transitional", "turbulent") if r in set(df.regime)]
    fig, axes = plt.subplots(3, len(regs), figsize=(2.4 * len(regs), 7.0), squeeze=False)
    for j, reg in enumerate(regs):
        row = df[df.regime == reg].iloc[0]
        W = np.load(ROOT / "data" / "raw" / flow / f"p{row.param:g}_r{int(row.traj)}.npy", mmap_mode="r")
        f = normalize_window(np.asarray(W[int(row.window), 0], dtype=np.float64))
        ax = axes[0, j]
        vmax = np.abs(f).max()
        ax.imshow(f, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{reg}\n{'Re' if flow == 'kolmogorov' else 'Ra'} = {row.param:g}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        h0, h1 = sublevel_persistence(f, c["periodic"], c["min_pers"])
        ax = axes[1, j]
        cc = __import__("gudhi").PeriodicCubicalComplex(top_dimensional_cells=f,
                                                       periodic_dimensions=list(c["periodic"]))
        cc.compute_persistence(homology_coeff_field=2, min_persistence=c["min_pers"])
        for dim, col, mk in ((0, plots.VIEW_COLORS["X2"], "o"), (1, plots.VIEW_COLORS["X1"], "^")):
            iv = cc.persistence_intervals_in_dimension(dim)
            iv = iv[np.isfinite(iv[:, 1])] if len(iv) else iv
            if len(iv):
                ax.scatter(iv[:, 0], iv[:, 1], s=9, marker=mk, color=col, edgecolor="white",
                           linewidth=0.3, label=f"H{dim} ({len(iv)})")
        lim = [f.min(), f.max()]
        ax.plot(lim, lim, color=plots.INK2, lw=0.8)
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.legend(frameon=False, loc="lower right")
        ax.set_title("X2: sublevel persistence")
        A = np.abs(np.nan_to_num(np.corrcoef(patch_matrix(f, c["blocks"]).T)))
        np.fill_diagonal(A, 0)
        ax = axes[2, j]
        ax.imshow((A > c["theta"]).astype(float), cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"X1: graph, density {row['X1_density']:.2f}")
        ax.set_xlabel("patch")
        ax.set_ylabel("patch")
        ax.grid(False)
    fig.suptitle(f"{plots.FLOW_TITLES[flow]}: one standardised snapshot per regime and its two views",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"fig_views_{flow}.png")
    fig.savefig(ROOT / "results" / f"fig_views_{flow}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    regime_scan()
    for fl in ("kolmogorov", "rb"):
        if glob.glob(str(ROOT / "data" / "raw" / fl / "*.npy")):
            views(fl)
