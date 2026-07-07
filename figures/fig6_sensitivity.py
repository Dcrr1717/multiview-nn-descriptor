# -*- coding: utf-8 -*-
# Regenera fig_sensitivity con el barrido REAL de theta (phase17, fase16 pkl).
# Corrige el mathtext roto de la version anterior y reemplaza la serie plana
# placeholder por los valores reales por umbral, con banda de variabilidad
# sobre repeticiones de la CV.
import pickle, sys, json
import numpy as np, networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = pickle.load(open(rf"{ROOT}/outputs/fase16_cifar10_raw/all_records_f16.pkl", "rb"))
fam = np.array([r["family"] for r in R]); yf = LabelEncoder().fit_transform(fam)
print(f"n={len(R)} familias={sorted(set(fam))}", flush=True)

REPS = 60

def famauc_stats(X, k=5, reps=REPS):
    X = np.asarray(X)
    if X.ndim == 1: X = X.reshape(-1, 1)
    means = []
    for rep in range(reps):
        skf = StratifiedKFold(k, shuffle=True, random_state=rep); fa = []
        for tr, te in skf.split(X, yf):
            clf = Pipeline([("s", StandardScaler()), ("c", OneVsRestClassifier(
                LogisticRegression(C=1, max_iter=500, random_state=42)))]).fit(X[tr], yf[tr])
            fa.append(roc_auc_score(yf[te], clf.predict_proba(X[te]),
                                    multi_class="ovr", average="weighted",
                                    labels=clf.named_steps["c"].classes_))
        means.append(np.mean(fa))
    means = np.array(means)
    return float(means.mean()), float(means.std())

def d_x1(act, theta=0.40, max_n=80, corr="pearson"):
    N = min(act.shape[1], max_n); Aa = act[:, :N]
    C = np.corrcoef(Aa.T)
    C = np.atleast_2d(np.nan_to_num(C)); np.fill_diagonal(C, 0)
    G = nx.from_numpy_array((np.abs(C) > theta).astype(float))
    deg = np.array([d for _, d in G.degree()]) if G.number_of_nodes() else np.array([0.])
    try: bc = np.array(list(nx.betweenness_centrality(G).values()))
    except Exception: bc = np.zeros(len(deg))
    return np.array([nx.density(G), nx.average_clustering(G), nx.transitivity(G),
        deg.mean(), deg.std(), deg.max()/max(1,len(deg)),
        float(nx.number_connected_components(G))/max(G.number_of_nodes(),1),
        G.number_of_edges()/max(G.number_of_nodes(),1), bc.mean(), bc.std()], float)

def d_x2(act, n_pts=50, r=15):
    A = act[:n_pts]; nc = min(r, A.shape[1], A.shape[0]-1)
    if nc < 2: return np.zeros(12)
    Ar = PCA(n_components=nc, random_state=42).fit_transform(A)
    Ar = (Ar-Ar.min())/(Ar.max()-Ar.min()+1e-8)
    corr = np.nan_to_num(np.corrcoef(Ar.T)); dist = 1-np.abs(corr)
    thrs = np.linspace(0,1,30); prev = Ar.shape[0]; h0 = []
    for ki, t in enumerate(thrs[1:], 1):
        cc = nx.number_connected_components(nx.from_numpy_array((dist < t).astype(int)))
        for _ in range(max(prev-cc, 0)): h0.append([thrs[ki-1], t])
        prev = cc
    h0 = np.array(h0) if h0 else np.array([[0., .05]])
    p = h0[:,1]-h0[:,0]
    s6 = np.array([p.mean(),p.std(),p.max(),p.sum(),float(len(p)),np.percentile(p,75)])
    return np.concatenate([s6, np.zeros(6)])

def d_x3(a, theta=.40):
    x1 = d_x1(a, theta); x2 = d_x2(a)
    return np.concatenate([x1/(np.linalg.norm(x1)+1e-8), x2/(np.linalg.norm(x2)+1e-8)])

def last(r): return r["acts"][list(r["acts"])[-1]]

THETAS = [0.2, 0.3, 0.4, 0.5, 0.6]
mus, sds = [], []
for t in THETAS:
    m, s = famauc_stats([d_x3(last(r), theta=t) for r in R])
    mus.append(m); sds.append(s)
    print(f"theta={t}: FamAUC={m:.4f}  sd_reps={s:.4f}", flush=True)

json.dump({"thetas": THETAS, "auc": mus, "sd": sds, "reps": REPS},
          open(os.path.join(ROOT, "figures", "sensitivity_data.json"), "w"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino", "Times New Roman"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
})
mus = np.array(mus); sds = np.array(sds)
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.fill_between(THETAS, mus-1.96*sds, mus+1.96*sds, alpha=0.18, color="#0072B2", lw=0)
ax.plot(THETAS, mus, "-o", color="#0072B2", ms=5, lw=1.4)
ax.axvline(0.4, ls="--", lw=0.9, color="gray")
ax.text(0.405, ax.get_ylim()[0]+0.15*(ax.get_ylim()[1]-ax.get_ylim()[0]),
        r"$\theta = 0.4$ (adopted)", fontsize=7.5, color="gray")
ax.set_xlabel(r"Correlation threshold $\theta$")
ax.set_ylabel("Family AUC (OvR, KFold-5)")
ax.set_title(r"Sensitivity of $\mathbf{X}_3$ to the functional-graph threshold",
             fontsize=9)
ax.set_xticks(THETAS)
fig.tight_layout()
for out in [os.path.join(ROOT, "figures", "fig6_sensitivity")]:
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
print("fig_sensitivity regenerada (pdf/png en ambas carpetas).")
print(f"RANGO: [{mus.min():.4f}, {mus.max():.4f}] amplitud={mus.max()-mus.min():.4f}")
