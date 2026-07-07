# -*- coding: utf-8 -*-
"""
FASE 17 — Experimentos pedidos por la revisión (sobre las activaciones/pesos
re-extraídos en la Fase 16, CIFAR-10, n=40):
  [A] Sensibilidad: theta, tipo de correlación (Pearson/Spearman), tope de
      neuronas, N (ejemplos para PH), r (dim PCA), y elección de capa.
  [B] Baseline CKA (similitud representacional lineal) -> Family AUC.
  [C] Neural Persistence sobre pesos -> Family AUC.
Todo con Family AUC uno-contra-resto, CV estratificada por familia, leak-free
(X3_orig no usa la precisión).
"""
import pickle, numpy as np, networkx as nx
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

R = pickle.load(open("outputs/fase16_cifar10_raw/all_records_f16.pkl", "rb"))
fam = np.array([r["family"] for r in R]); yf = LabelEncoder().fit_transform(fam)
print(f"n={len(R)}  familias={sorted(set(fam))}")

def famauc(X, k=5, reps=60):
    X = np.asarray(X);
    if X.ndim == 1: X = X.reshape(-1, 1)
    means = []
    for rep in range(reps):
        skf = StratifiedKFold(k, shuffle=True, random_state=rep); fa = []
        for tr, te in skf.split(X, yf):
            clf = Pipeline([("s", StandardScaler()), ("c", OneVsRestClassifier(LogisticRegression(C=1, max_iter=500, random_state=42)))]).fit(X[tr], yf[tr])
            fa.append(roc_auc_score(yf[te], clf.predict_proba(X[te]), multi_class="ovr", average="weighted", labels=clf.named_steps["c"].classes_))
        means.append(np.mean(fa))
    return float(np.mean(means))

# ── descriptores parametrizables ────────────────────────────────────
def d_x1(act, theta=0.40, max_n=80, corr="pearson"):
    N = min(act.shape[1], max_n); Aa = act[:, :N]
    C = (np.corrcoef(Aa.T) if corr == "pearson" else spearmanr(Aa)[0])
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
    Ar = PCA(n_components=nc, random_state=42).fit_transform(A); Ar = (Ar-Ar.min())/(Ar.max()-Ar.min()+1e-8)
    corr = np.nan_to_num(np.corrcoef(Ar.T)); dist = 1-np.abs(corr); thrs = np.linspace(0,1,30); prev = Ar.shape[0]; h0 = []
    for ki, t in enumerate(thrs[1:], 1):
        cc = nx.number_connected_components(nx.from_numpy_array((dist < t).astype(int)))
        for _ in range(max(prev-cc, 0)): h0.append([thrs[ki-1], t])
        prev = cc
    h0 = np.array(h0) if h0 else np.array([[0., .05]])
    p = h0[:,1]-h0[:,0]; s6 = np.array([p.mean(),p.std(),p.max(),p.sum(),float(len(p)),np.percentile(p,75)])
    return np.concatenate([s6, np.zeros(6)])

def d_x3(a, theta=.40, max_n=80, corr="pearson", n_pts=50, r=15):
    x1 = d_x1(a, theta, max_n, corr); x2 = d_x2(a, n_pts, r)
    return np.concatenate([x1/(np.linalg.norm(x1)+1e-8), x2/(np.linalg.norm(x2)+1e-8)])

def last(r): return r["acts"][list(r["acts"])[-1]]

# ── [A] SENSIBILIDAD ────────────────────────────────────────────────
print("\n[A] SENSIBILIDAD (Family AUC de X3, CIFAR-10)")
ref = famauc([d_x3(last(r)) for r in R]); print(f"  referencia (theta=0.4, Pearson, cap80, N=50, r=15): {ref:.3f}")
def sweep(name, kwargs_list):
    vals = []
    for lbl, kw in kwargs_list:
        v = famauc([d_x3(last(r), **kw) for r in R]); vals.append(v)
        print(f"    {name}={lbl:8s} -> {v:.3f}")
    print(f"    >> rango {name}: [{min(vals):.3f}, {max(vals):.3f}]  (amplitud {max(vals)-min(vals):.3f})")
sweep("theta", [(str(t), {"theta": t}) for t in [0.2,0.3,0.4,0.5,0.6]])
sweep("corr",  [("Pearson",{"corr":"pearson"}),("Spearman",{"corr":"spearman"})])
sweep("cap",   [(str(c),{"max_n":c}) for c in [40,80,200]])
sweep("N",     [(str(n),{"n_pts":n}) for n in [50,100,200]])
sweep("r",     [(str(rr),{"r":rr}) for rr in [8,12,15]])

# capa: solo redes con >=3 capas guardadas
print("  --- elección de capa (redes con >=3 capas) ---")
multi = [r for r in R if len(r["acts"]) >= 3]
if multi:
    nlayers = min(len(r["acts"]) for r in multi)
    for li in range(nlayers):
        Xl = [d_x3(list(r["acts"].values())[li]) for r in multi]
        yfm = LabelEncoder().fit_transform([r["family"] for r in multi])
        # famauc local sobre subconjunto
        means=[]
        for rep in range(40):
            skf=StratifiedKFold(min(5,np.bincount(yfm).min()),shuffle=True,random_state=rep); fa=[]
            for tr,te in skf.split(np.array(Xl),yfm):
                Xa=np.array(Xl)
                clf=Pipeline([("s",StandardScaler()),("c",OneVsRestClassifier(LogisticRegression(C=1,max_iter=500,random_state=42)))]).fit(Xa[tr],yfm[tr])
                fa.append(roc_auc_score(yfm[te],clf.predict_proba(Xa[te]),multi_class="ovr",average="weighted",labels=clf.named_steps["c"].classes_))
            means.append(np.mean(fa))
        print(f"    capa {li} ({list(multi[0]['acts'])[li]}): {np.mean(means):.3f}  (n={len(multi)})")
else:
    print("    (no hay redes con >=3 capas guardadas)")

# ── [B] CKA baseline ────────────────────────────────────────────────
print("\n[B] BASELINE CKA (similitud representacional lineal) -> Family AUC")
def gram_center(X):
    X = X - X.mean(0, keepdims=True); return X @ X.T
def cka(X, Y):
    Kx, Ky = gram_center(X), gram_center(Y)
    hsic = (Kx*Ky).sum(); nx_ = np.sqrt((Kx*Kx).sum()); ny = np.sqrt((Ky*Ky).sum())
    return hsic/(nx_*ny+1e-12)
n = len(R); K = np.eye(n)
acts_last = [last(r) for r in R]
for i in range(n):
    for j in range(i+1, n):
        m = min(acts_last[i].shape[0], acts_last[j].shape[0])
        K[i,j] = K[j,i] = cka(acts_last[i][:m], acts_last[j][:m])
# Family AUC con SVM de kernel precomputado (CKA), leak-free
def famauc_kernel(K, reps=60):
    means=[]
    for rep in range(reps):
        skf=StratifiedKFold(5,shuffle=True,random_state=rep); fa=[]
        for tr,te in skf.split(np.zeros(n),yf):
            clf=OneVsRestClassifier(SVC(kernel="precomputed",C=1)).fit(K[np.ix_(tr,tr)],yf[tr])
            sc=clf.decision_function(K[np.ix_(te,tr)])
            fa.append(roc_auc_score(yf[te],sc,multi_class="ovr",average="weighted",labels=clf.classes_))
        means.append(np.mean(fa))
    return float(np.mean(means))
print(f"  CKA (kernel SVM) Family AUC: {famauc_kernel(K):.3f}")

# ── [C] Neural Persistence -> Family AUC ────────────────────────────
print("\n[C] NEURAL PERSISTENCE (pesos) -> Family AUC")
NP = np.array([r["np_desc"] for r in R])
print(f"  Neural Persistence Family AUC: {famauc(NP):.3f}")
print(f"\n  (referencia X3 = {ref:.3f})")
