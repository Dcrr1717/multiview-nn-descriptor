#!/usr/bin/env python3
"""
fig3_tsne.py — Family Separation via t-SNE
Figure 3 of the paper: X1 vs X2 vs X3 — Silhouette comparison.

Specs:
  - 3 panels horizontal, width = 174mm (full-width Elsevier)
  - Perplexity = 10 (appropriate for n=40)
  - Paleta Okabe-Ito para 5 familias
  - Output: fig3_tsne.pdf
"""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino", "Times New Roman"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito 5-colour palette (daltonism-safe)
PALETTE = {
    "A_Shallow": "#E69F00",
    "B_Deep":    "#56B4E9",
    "C_Residual":"#009E73",
    "D_CNNsmall":"#CC79A7",
    "E_CNNres":  "#0072B2",
}
MARKERS = {"A_Shallow":"o","B_Deep":"s","C_Residual":"^","D_CNNsmall":"D","E_CNNres":"P"}

RECORDS = Path("../outputs/fase10_results/all_records_f10.pkl")
with open(RECORDS, "rb") as f:
    records = pickle.load(f)

X1 = np.array([r["x1_g"] for r in records])
X2 = np.array([r["x2_g"] for r in records])
X3 = np.array([r["x3_g"] for r in records])
fams = np.array([r["family"] for r in records])
y    = np.array([r["acc_global"] for r in records])
le   = LabelEncoder(); y_fam = le.fit_transform(fams)

# Adaptive X3 weights (simplified here — use full version in final paper)
sc1,sc2 = StandardScaler(), StandardScaler()
X1s = sc1.fit_transform(X1); X2s = sc2.fit_transform(X2)
w1 = np.array([pearsonr(X1s[:,j],y)[0]**2 for j in range(X1s.shape[1])])
w2 = np.array([pearsonr(X2s[:,j],y)[0]**2 for j in range(X2s.shape[1])])
w1 /= w1.sum()+1e-10; w2 /= w2.sum()+1e-10
X3a = np.hstack([X1s*w1, X2s*w2])

descriptors = [("$X_1$ (Functional Graph)", X1),
               ("$X_2$ (Persistence)", X2),
               ("$X_3$ (Multiview)", X3a)]

fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.5))

for ax, (title, X) in zip(axes, descriptors):
    Xs = StandardScaler().fit_transform(X)
    emb = TSNE(n_components=2, perplexity=10, random_state=42,
               n_iter=1000, learning_rate="auto", init="pca").fit_transform(Xs)
    sil = silhouette_score(Xs, y_fam)
    for fam in np.unique(fams):
        mask = fams == fam
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=PALETTE[fam], marker=MARKERS[fam],
                   s=22, linewidths=0.3, edgecolors="white",
                   label=fam.replace("_", " "), zorder=3)
    ax.set_title(f"{title}\nSilhouette = {sil:.3f}", fontsize=7.5, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("$t$-SNE dim 1"); ax.set_ylabel("$t$-SNE dim 2")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5,
           frameon=False, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.08))

fig.suptitle("$t$-SNE projection of descriptor spaces (family colouring)",
             fontsize=8, y=1.01)
fig.tight_layout()
fig.savefig("fig3_tsne.pdf", format="pdf")
fig.savefig("fig3_tsne.png")
print("  fig3_tsne.pdf / .png saved.")
