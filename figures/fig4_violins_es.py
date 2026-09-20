# -*- coding: utf-8 -*-
"""fig4_violins_es.py — AUC familiar por pliegue (CIFAR-10), versión en español
con coma decimal. Mismo pipeline de datos que figures/fig4_violins.py del
repositorio (registros de fase 10, validación cruzada estratificada de cinco
pliegues, semilla 42); las marcas de significación se CALCULAN (Wilcoxon
unilateral pareado sobre los cinco pliegues, mínimo alcanzable p=1/32)."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr, wilcoxon
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings; warnings.filterwarnings("ignore")

ROOT = r"D:\OneDrive\Desktop\proyecto de tesis espoch"
HERE = os.path.dirname(os.path.abspath(__file__))
coma = lambda v, nd=2: f"{v:.{nd}f}".replace(".", ",")
records = pickle.load(open(os.path.join(ROOT, "outputs", "fase10_results", "all_records_f10.pkl"), "rb"))

X1 = np.array([r["x1_g"] for r in records])
X2 = np.array([r["x2_g"] for r in records])
X3 = np.array([r["x3_g"] for r in records])
y = np.array([r["acc_global"] for r in records])
fams = np.array([r["family"] for r in records])
y_fam = LabelEncoder().fit_transform(fams)

sc1, sc2 = StandardScaler(), StandardScaler()
X1s = sc1.fit_transform(X1); X2s = sc2.fit_transform(X2)
w1 = np.nan_to_num(np.array([pearsonr(X1s[:, j], y)[0]**2 for j in range(X1s.shape[1])]))
w2 = np.nan_to_num(np.array([pearsonr(X2s[:, j], y)[0]**2 for j in range(X2s.shape[1])]))
w1 /= w1.sum()+1e-10; w2 /= w2.sum()+1e-10
X3a = np.hstack([X1s*w1, X2s*w2])

DESCR = [("$X_1$", X1), ("$X_2$", X2), ("$X_3^{(c)}$", X3), ("$X_3^{(a)}$", X3a)]
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#0072B2"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_data = []
for name, Xm in DESCR:
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=500, random_state=42)))])
    aucs = cross_val_score(pipe, Xm, y_fam, cv=skf, scoring="roc_auc_ovr_weighted")
    auc_data.append(aucs)
    print(name, np.round(aucs, 4), "media", round(aucs.mean(), 4))

p1 = wilcoxon(auc_data[2], auc_data[0], alternative="greater").pvalue  # X3c vs X1
p2 = wilcoxon(auc_data[2], auc_data[1], alternative="greater").pvalue  # X3c vs X2
print(f"Wilcoxon X3c>X1: p={p1:.5f} | X3c>X2: p={p2:.5f}")

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino Linotype", "Palatino", "Times New Roman"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
})
fig, ax = plt.subplots(figsize=(3.4, 2.8))
positions = np.arange(len(DESCR))
vparts = ax.violinplot(auc_data, positions=positions, widths=0.5, showmedians=True, showextrema=False)
for i, pc in enumerate(vparts["bodies"]):
    pc.set_facecolor(COLORS[i]); pc.set_alpha(0.35)
vparts["cmedians"].set_color("black"); vparts["cmedians"].set_lw(1.2)
rng = np.random.default_rng(0)
for i, aucs in enumerate(auc_data):
    jitter = rng.uniform(-0.08, 0.08, len(aucs))
    ax.scatter(positions[i]+jitter, aucs, color=COLORS[i], s=18, zorder=3, alpha=0.85, linewidths=0.3, edgecolors="white")
ax.axhline(0.9, ls="--", lw=0.8, color="gray", alpha=0.7)
ax.text(len(DESCR)-0.5, 0.902, "AUC = 0,90", fontsize=6.5, color="gray", ha="right")

def bracket(x1, x2, yy, h, text):
    ax.plot([x1, x1, x2, x2], [yy, yy+h, yy+h, yy], lw=0.8, color="black")
    ax.text((x1+x2)/2, yy+h+0.002, text, ha="center", va="bottom", fontsize=7)

bracket(0, 2, 0.97, 0.005, stars(p1))
bracket(1, 2, 0.99, 0.005, stars(p2))
ax.set_xticks(positions); ax.set_xticklabels([d[0] for d in DESCR])
ax.set_ylabel("AUC familiar (cinco pliegues)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: coma(v)))
ax.set_ylim(0.65, 1.02)
ax.set_title("Discriminación de familias por descriptor\n(violín = distribución en los cinco pliegues)", fontsize=8)
fig.tight_layout()
out = os.path.join(HERE, "fig4_violins_es")
fig.savefig(out + ".pdf", format="pdf"); fig.savefig(out + ".png")
print("fig4_violins_es generada; estrellas:", stars(p1), stars(p2))
