"""
╔══════════════════════════════════════════════════════════════════════╗
║  TESIS ESPOCH — FASE 11: RE-ANÁLISIS LEAK-FREE (n=40)                ║
║                                                                      ║
║  Corrige la fuga de información de las variantes que usan y          ║
║  (X3_adapt Pearson², X3_rank Spearman²): pesos y proyecciones Ridge  ║
║  se ajustan SOLO en el train de cada pliegue/bootstrap/LOO y se      ║
║  aplican al held-out. X1, X2, X3_orig no usan y (limpios).           ║
║                                                                      ║
║  Métricas (todas con IC 95%):                                        ║
║    Family AUC (OvR, KFold-5 estratificada repetida)                  ║
║    Pairwise AUC (bootstrap B=300 out-of-bag)                         ║
║    Kendall τ (KFold-5 repetida, out-of-fold)                         ║
║    Cohen's d (LOOCV, top vs bottom; efecto estandarizado)            ║
║    Mann-Whitney |r| (+ p) sobre predicciones out-of-fold             ║
║    BH-FDR sobre los 10 contrastes (5 Kendall + 5 MW)                 ║
║  Fuente única: outputs/fase10_results/all_records_f10.pkl            ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import json, pickle, warnings, sys
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau, mannwhitneyu
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
RNG = np.random.RandomState(2024)

REC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./outputs/fase10_results/all_records_f10.pkl")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./outputs/fase11_results")
OUT.mkdir(parents=True, exist_ok=True)
print(f"  Records: {REC}")

records = pickle.load(open(REC, "rb"))
y_g   = np.array([r["acc_global"] for r in records], float)
X1_g  = np.array([r["x1_g"] for r in records], float)
X2_g  = np.array([r["x2_g"] for r in records], float)
X3_g  = np.array([r["x3_g"] for r in records], float)
fams  = np.array([r["family"] for r in records])
archt = np.array([r.get("arch_type", "MLP") for r in records])
N = len(records)
VARIANTS = ["X1", "X2", "X3_orig", "X3_adapt", "X3_rank"]

# ── Constructor leak-free de la fusión adaptativa ─────────────────────
def fit_x3a(X1tr, X2tr, ytr, mode):
    sc1, sc2 = StandardScaler().fit(X1tr), StandardScaler().fit(X2tr)
    X1s, X2s = sc1.transform(X1tr), sc2.transform(X2tr)
    def wts(Xs):
        ws = []
        for j in range(Xs.shape[1]):
            col = Xs[:, j]
            if col.std() < 1e-8: ws.append(0.0)
            elif mode == "pearson": ws.append(pearsonr(col, ytr)[0] ** 2)
            else: ws.append(spearmanr(col, ytr)[0] ** 2)
        ws = np.array(ws); return ws / (ws.sum() + 1e-10)
    w1, w2 = wts(X1s), wts(X2s)
    ymu = ytr.mean()
    r1 = Ridge(1.0).fit(X1s, ytr - ymu); r2 = Ridge(1.0).fit(X2s, ytr - ymu)
    def transform(X1te, X2te):
        a = sc1.transform(X1te); b = sc2.transform(X2te)
        return np.column_stack([a * w1, b * w2, (a @ r1.coef_).reshape(-1, 1),
                                (b @ r2.coef_).reshape(-1, 1)])
    return transform

def build(variant, tr, te, X1, X2, X3, y):
    if variant == "X1":      return X1[tr], X1[te]
    if variant == "X2":      return X2[tr], X2[te]
    if variant == "X3_orig": return X3[tr], X3[te]
    mode = "pearson" if variant == "X3_adapt" else "spearman"
    tf = fit_x3a(X1[tr], X2[tr], y[tr], mode)
    return tf(X1[tr], X2[tr]), tf(X1[te], X2[te])

def auc_w(y_true, proba, classes):
    try:
        if len(classes) == 2: return roc_auc_score(y_true, proba[:, 1])
        return roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes)
    except Exception:
        return np.nan

def family_auc(X1, X2, X3, y, y_fam, variant, k=5, repeats=60):
    means = []
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rep)
        fa = []
        for tr, te in skf.split(X1, y_fam):
            Xtr, Xte = build(variant, tr, te, X1, X2, X3, y)
            clf = Pipeline([("sc", StandardScaler()),
                            ("clf", OneVsRestClassifier(
                                LogisticRegression(C=1.0, max_iter=500, random_state=42)))])
            clf.fit(Xtr, y_fam[tr])
            fa.append(auc_w(y_fam[te], clf.predict_proba(Xte), clf.named_steps["clf"].classes_))
        means.append(np.nanmean(fa))
    means = np.array(means)
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def oof_preds(X1, X2, X3, y, variant, k=5, seed=42):
    n = len(y); preds = np.zeros(n)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in kf.split(X1):
        Xtr, Xte = build(variant, tr, te, X1, X2, X3, y)
        pipe = Pipeline([("sc", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-3, 3, 20)))])
        pipe.fit(Xtr, y[tr]); preds[te] = pipe.predict(Xte)
    return preds

def loo_preds(X1, X2, X3, y, variant):
    n = len(y); preds = np.zeros(n)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i]); te = np.array([i])
        Xtr, Xte = build(variant, tr, te, X1, X2, X3, y)
        pipe = Pipeline([("sc", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-3, 3, 20)))])
        pipe.fit(Xtr, y[tr]); preds[i] = pipe.predict(Xte)[0]
    return preds

def kendall(X1, X2, X3, y, variant, k=5, repeats=60):
    taus = [kendalltau(oof_preds(X1, X2, X3, y, variant, k=k, seed=r), y)[0] for r in range(repeats)]
    taus = np.array(taus)
    _, pval = kendalltau(oof_preds(X1, X2, X3, y, variant, k=k, seed=42), y)
    return float(taus.mean()), float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5)), float(pval)

def pairwise_auc(X1, X2, X3, y, variant, n_boot=300):
    n = len(y); boots = []
    for _ in range(n_boot):
        idx_in = RNG.choice(n, n, replace=True)
        idx_out = np.setdiff1d(np.arange(n), idx_in)
        if len(idx_out) < 3: continue
        Xtr, Xte = build(variant, idx_in, idx_out, X1, X2, X3, y)
        sc = StandardScaler().fit(Xtr); rd = Ridge(1.0).fit(sc.transform(Xtr), y[idx_in])
        so = rd.predict(sc.transform(Xte)); yo = y[idx_out]
        ok = tot = 0
        for ii in range(len(yo)):
            for jj in range(ii + 1, len(yo)):
                if abs(yo[ii] - yo[jj]) < 0.01: continue
                tot += 1
                if (so[ii] > so[jj]) == (yo[ii] > yo[jj]): ok += 1
        if tot > 0: boots.append(ok / tot)
    boots = np.array(boots)
    return float(boots.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def cohen_mw(X1, X2, X3, y, variant, k_split):
    """Cohen's d (LOOCV) y Mann-Whitney |r| sobre predicciones out-of-sample."""
    s = loo_preds(X1, X2, X3, y, variant)            # leak-free oof scores
    order = np.argsort(y); bot = order[:k_split]; top = order[-k_split:]
    st, sb = s[top], s[bot]
    pooled = np.sqrt((st.var(ddof=1) + sb.var(ddof=1)) / 2) + 1e-10
    d = (st.mean() - sb.mean()) / pooled
    # bootstrap CI de d
    bd = []
    for _ in range(1000):
        a = st[RNG.randint(0, k_split, k_split)]; b = sb[RNG.randint(0, k_split, k_split)]
        pl = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-10
        bd.append((a.mean() - b.mean()) / pl)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    U, p = mannwhitneyu(st, sb, alternative="two-sided")
    r_abs = abs(1 - 2 * U / (k_split * k_split))
    return float(d), float(lo), float(hi), float(r_abs), float(p)

def run_block(X1, X2, X3, y, fam, label, kf=5, kfam=5):
    le = LabelEncoder(); y_fam = le.fit_transform(fam)
    n = len(y); k_split = max(2, n // 4)
    print("\n" + "=" * 92)
    print(f"  {label}  (n={n}, acc=[{y.min():.2f},{y.max():.2f}], rango={y.max()-y.min():.2f} pp, k_split={k_split})")
    print("=" * 92)
    print(f"  {'var':9s} {'FamAUC[IC]':22s} {'Pairwise[IC]':22s} {'tau[IC]':20s} {'d_LOOCV[IC]':18s} {'MW|r|':6s}")
    out = {}
    for v in VARIANTS:
        fa, fl, fh = family_auc(X1, X2, X3, y, y_fam, v, k=kfam)
        pa, pl, ph = pairwise_auc(X1, X2, X3, y, v)
        ta, tl, th, tp = kendall(X1, X2, X3, y, v, k=kf)
        dd, dl, dh, mr, mp = cohen_mw(X1, X2, X3, y, v, k_split)
        out[v] = {"family_auc": [fa, fl, fh], "pairwise_auc": [pa, pl, ph],
                  "kendall_tau": [ta, tl, th], "tau_p": tp,
                  "cohens_d": [dd, dl, dh], "mw_r": mr, "mw_p": mp}
        print(f"  {v:9s} {fa:.3f}[{fl:.3f},{fh:.3f}]  {pa:.3f}[{pl:.3f},{ph:.3f}]  "
              f"{ta:+.3f}[{tl:+.3f},{th:+.3f}]  {dd:+.2f}[{dl:+.2f},{dh:+.2f}]  {mr:.3f}")
    # BH-FDR sobre 10 contrastes (5 Kendall + 5 MW)
    pv = {f"Kendall {v}": out[v]["tau_p"] for v in VARIANTS}
    pv.update({f"MW {v}": out[v]["mw_p"] for v in VARIANTS})
    rej, padj, _, _ = multipletests(list(pv.values()), alpha=0.05, method="fdr_bh")
    rej_bonf = np.array(list(pv.values())) < (0.05 / len(pv))
    print(f"  BH-FDR: {int(rej.sum())}/{len(rej)} sig.  |  Bonferroni(0.005): {int(rej_bonf.sum())}/{len(rej)}")
    out["_bh"] = {"n_sig": int(rej.sum()), "n_total": int(len(rej)),
                  "n_bonf": int(rej_bonf.sum())}
    return out

res_full  = run_block(X1_g, X2_g, X3_g, y_g, fams, "[A] FULL n=40", kf=5, kfam=5)
m = archt == "MLP"; c = archt == "CNN"
res_mlp   = run_block(X1_g[m], X2_g[m], X3_g[m], y_g[m], fams[m], "[B] WITHIN-MLP n=30", kf=5, kfam=5)
res_cnn   = run_block(X1_g[c], X2_g[c], X3_g[c], y_g[c], fams[c], "[C] WITHIN-CNN n=10", kf=3, kfam=2)
y_norm = np.zeros(N)
for mask in [m, c]:
    mu, sg = y_g[mask].mean(), y_g[mask].std(); y_norm[mask] = (y_g[mask] - mu) / (sg + 1e-8)
res_ynorm = run_block(X1_g, X2_g, X3_g, y_norm, fams, "[D] Y_NORM n=40", kf=5, kfam=5)

rng_mlp = float(y_g[m].max() - y_g[m].min()); rng_cnn = float(y_g[c].max() - y_g[c].min())
print("\n" + "=" * 60)
print("  RANGOS DE PRECISIÓN (④)")
print(f"  MLP n=30: [{y_g[m].min():.2f},{y_g[m].max():.2f}]  rango={rng_mlp:.2f} pp")
print(f"  CNN n=10: [{y_g[c].min():.2f},{y_g[c].max():.2f}]  rango={rng_cnn:.2f} pp")
print(f"  Total n=40: rango={y_g.max()-y_g.min():.2f} pp")

summary = {"full": res_full, "within_mlp": res_mlp, "within_cnn": res_cnn, "y_norm": res_ynorm,
           "acc_ranges": {"mlp_pp": rng_mlp, "cnn_pp": rng_cnn, "total_pp": float(y_g.max() - y_g.min())}}
json.dump(summary, open(OUT / "summary_leakfree.json", "w"), indent=2)
print(f"\n  Guardado: {OUT/'summary_leakfree.json'}")
