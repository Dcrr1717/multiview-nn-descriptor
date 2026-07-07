#!/usr/bin/env python3
"""
fig5_heatmap.py — Within-Type Robustness Heatmap
Figure 5 of the paper.

Specs:
  - 4 rows (analysis contexts) × 5 cols (descriptors)
  - AUC values as heatmap + text annotations
  - Diverging colormap centered at 0.85 (baseline threshold)
  - Output: fig5_heatmap.pdf
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino", "Times New Roman"],
    "font.size": 8, "axes.labelsize": 8, "figure.dpi": 150,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})

# AUC values from Phase 10 and Phase 10b results
# Rows: Global n=40, Within-MLP n=30, Within-CNN n=10, Y-norm (anti-confounder)
# Cols: X1, X2, X3_orig, X3_adapt, X3_rank
data = np.array([
    # Global (Family AUC)
    [0.8387, 0.8696, 0.9458, 0.9589, 0.9589],
    # Within-MLP (Family AUC)
    [0.7873, 0.8214, 0.8889, 0.9278, 0.9278],
    # Within-CNN (Family AUC)
    [1.0000, 0.7500, 1.0000, 1.0000, 1.0000],
    # Y-norm (Pairwise AUC — anti-confounder)
    [0.7501, 0.6052, 0.7060, 0.7699, 0.7699],
])

rows = ["Global\n($n{=}40$)", "Within-MLP\n($n{=}30$)",
        "Within-CNN\n($n{=}10$)", "$Y_{\\text{norm}}$\n(pairwise AUC)"]
cols = ["$X_1$", "$X_2$", "$X_3^{(\\text{c})}$",
        "$X_3^{(\\text{a})}$", "$X_3^{(\\text{r})}$"]

fig, ax = plt.subplots(figsize=(4.5, 2.4))

# Colourmap: RdYlGn clipped to [0.60, 1.00]
cmap = plt.cm.RdYlGn
norm = Normalize(vmin=0.60, vmax=1.00)
im   = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

# Text annotations
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        fg = "black" if 0.70 < v < 0.95 else "white"
        bold = "bold" if j >= 2 else "normal"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=7, color=fg, fontweight=bold)

# Best-in-row marker
for i in range(data.shape[0]):
    best_j = np.argmax(data[i])
    ax.add_patch(plt.Rectangle((best_j-0.5, i-0.5), 1, 1,
                                fill=False, edgecolor="black", lw=1.5))

ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7.5)
ax.set_title("Family AUC across analysis contexts\n(bold box = best per row)",
             fontsize=8, pad=6)

cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    fraction=0.038, pad=0.04)
cbar.set_label("AUC", fontsize=7)
cbar.ax.tick_params(labelsize=6.5)

fig.tight_layout()
fig.savefig("fig5_heatmap.pdf", format="pdf")
fig.savefig("fig5_heatmap.png")
print("  fig5_heatmap.pdf / .png saved.")
