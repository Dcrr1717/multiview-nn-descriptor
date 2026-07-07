"""
╔══════════════════════════════════════════════════════════════════════╗
║  FASE 14 — Controles solicitados por la revisión de pares            ║
║   #1  Línea base TRIVIAL: ¿predice la familia el solo tamaño/         ║
║       profundidad de la red (n_params, log n_params, profundidad)?    ║
║   #2  Test PAREADO de ΔAUC familiar: X3 vs X2 (bootstrap sobre redes) ║
║   Uso: python phase14_baselines.py <records.pkl> [etiqueta]          ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import sys, pickle, warnings
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
RNG = np.random.RandomState(2024)

REC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./outputs/fase10_results/all_records_f10.pkl")
LABEL = sys.argv[2] if len(sys.argv) > 2 else REC.stem
recs = pickle.load(open(REC, "rb"))
print(f"\n===== {LABEL}  (n={len(recs)}) =====")

def depth(cfg):
    if "hidden" in cfg:      return len(cfg["hidden"])           # MLP sha/dep
    if "hidden_dim" in cfg:  return int(cfg.get("n_blocks", 1))  # MLP residual
    d = len(cfg.get("channels", [1]))                            # CNN
    d += len(cfg.get("fc", [])) + int(cfg.get("n_blocks", 0))
    return d

y_fam = LabelEncoder().fit_transform(np.array([r["family"] for r in recs]))
archt = np.array([r.get("arch_type", "MLP") for r in recs])
npar  = np.array([float(r["n_params"]) for r in recs])
dep   = np.array([float(depth(r["cfg"])) for r in recs])
X1 = np.array([r["x1_g"] for r in recs]); X2 = np.array([r["x2_g"] for r in recs])
X3 = np.array([r["x3_g"] for r in recs])  # X3_orig (sin y, limpio)

# Descriptor TRIVIAL: tamaño/profundidad (la línea base más simple imaginable)
TRIV = np.column_stack([npar, np.log10(npar + 1.0), dep])

def family_auc(X, mask=None, k=5, repeats=200):
    yy = y_fam if mask is None else y_fam[mask]
    XX = X if mask is None else X[mask]
    if XX.ndim == 1: XX = XX.reshape(-1, 1)
    out = []
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rep)
        fa = []
        for tr, te in skf.split(XX, yy):
            clf = Pipeline([("sc", StandardScaler()),
                            ("clf", OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=500, random_state=42)))])
            clf.fit(XX[tr], yy[tr]); proba = clf.predict_proba(XX[te])
            cls = clf.named_steps["clf"].classes_
            try:
                a = (roc_auc_score(yy[te], proba[:, 1]) if len(cls) == 2 else
                     roc_auc_score(yy[te], proba, multi_class="ovr", average="weighted", labels=cls))
            except Exception:
                a = np.nan
            fa.append(a)
        out.append(np.nanmean(fa))
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)

# ── #1 Línea base trivial vs descriptores ────────────────────────────
print("\n[#1] FAMILY AUC — línea base trivial vs descriptores  (media [IC95])")
for name, X in [("n_params (1-d)", npar), ("profundidad (1-d)", dep),
                ("TRIVIAL [params,logparams,depth]", TRIV),
                ("X1 (grafo)", X1), ("X2 (TDA)", X2), ("X3 (fusion, sin y)", X3)]:
    m, lo, hi = family_auc(X)
    print(f"   {name:34s}  {m:.3f} [{lo:.3f}, {hi:.3f}]")

print("\n[#1b] Intra-MLP (n=30): ¿el tamaño/profundidad distingue A/B/C?")
mlp = archt == "MLP"
for name, X in [("TRIVIAL", TRIV), ("X3 (fusion)", X3)]:
    m, lo, hi = family_auc(X, mask=mlp)
    print(f"   {name:34s}  {m:.3f} [{lo:.3f}, {hi:.3f}]")

# ── #2 Test pareado de ΔAUC familiar: X3 vs X2 ───────────────────────
def oof_proba(X, k=5, seed=42):
    ncls = len(np.unique(y_fam)); P = np.zeros((len(y_fam), ncls))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y_fam):
        clf = Pipeline([("sc", StandardScaler()),
                        ("clf", OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=500, random_state=42)))])
        clf.fit(X[tr], y_fam[tr])
        cols = clf.named_steps["clf"].classes_
        P[np.ix_(te, cols)] = clf.predict_proba(X[te])   # asignación 2D correcta
    return P

def wauc(yy, P, idx):
    return roc_auc_score(yy[idx], P[idx], multi_class="ovr", average="weighted", labels=np.unique(yy))

def paired_delta(Xa, Xb, B=2000):
    Pa, Pb = oof_proba(Xa), oof_proba(Xb); n = len(y_fam); deltas = []
    for _ in range(B):
        idx = RNG.choice(n, n, replace=True)
        if len(np.unique(y_fam[idx])) < len(np.unique(y_fam)): continue
        try: deltas.append(wauc(y_fam, Pa, idx) - wauc(y_fam, Pb, idx))
        except Exception: pass
    d = np.array(deltas)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), max(p, 1.0/len(d))

print("\n[#2] Test pareado ΔFamily-AUC (bootstrap B=2000 sobre las redes)")
for a, b, an, bn in [(X3, X2, "X3", "X2"), (X3, X1, "X3", "X1"), (X3, TRIV, "X3", "TRIVIAL")]:
    dm, dl, dh, p = paired_delta(a, b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"   Δ({an}-{bn}) = {dm:+.3f}  IC95 [{dl:+.3f}, {dh:+.3f}]  p={p:.4f} {sig}")
