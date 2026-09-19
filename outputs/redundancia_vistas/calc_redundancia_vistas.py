# -*- coding: utf-8 -*-
"""
Analisis de redundancia entre vistas X1 (grafo funcional, 10 rasgos) y
X2 (homologia persistente, 12 rasgos) sobre el repertorio de n=40 redes,
en los tres dominios del articulo (CIFAR-10, CIFAR-100, SVHN).

Para cada par de rasgos (i,j) se calcula la correlacion de Spearman a traves
de las n=40 redes, su p-valor, y se aplica la correccion de Benjamini-Hochberg
(q<=0.05) a la familia completa de pares del dominio.

Motivo: la afirmacion previa del articulo ("|rho|<0.15 para todo par") no
provenia de ningun calculo registrado; este script la sustituye por cifras
reales y reproducibles. Ejecutado el 2026-08-04.
"""
import io, sys, json, pickle, warnings
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = r"D:/OneDrive/Desktop/proyecto de tesis espoch/outputs"
DOMINIOS = {
    "CIFAR-10":  BASE + "/fase16_cifar10_raw/all_records_f16.pkl",
    "CIFAR-100": BASE + "/fase12_cifar100/all_records_f12.pkl",
    "SVHN":      BASE + "/fase13_svhn/all_records_f13.pkl",
}

def bh(pvals, q=0.05):
    """Benjamini-Hochberg: devuelve mascara de rechazo."""
    p = np.asarray(pvals)
    m = len(p)
    orden = np.argsort(p)
    umbral = q * (np.arange(1, m + 1) / m)
    pasa = p[orden] <= umbral
    k = np.max(np.nonzero(pasa)[0]) + 1 if pasa.any() else 0
    rechazo = np.zeros(m, bool)
    rechazo[orden[:k]] = True
    return rechazo

resultados = {}
for dom, ruta in DOMINIOS.items():
    try:
        data = pickle.load(open(ruta, "rb"))
    except Exception as e:
        print(f"[{dom}] ERROR al cargar: {e}")
        continue
    k1 = "x1_g" if "x1_g" in data[0] else "x1"
    k2 = "x2_g" if "x2_g" in data[0] else "x2"
    X1 = np.array([np.asarray(r[k1]).ravel() for r in data])
    X2 = np.array([np.asarray(r[k2]).ravel() for r in data])
    n = len(data)
    pares = []
    for i in range(X1.shape[1]):
        for j in range(X2.shape[1]):
            rho, p = spearmanr(X1[:, i], X2[:, j])
            if not np.isnan(rho):
                pares.append((i, j, float(rho), float(p)))
    absr = np.array([abs(x[2]) for x in pares])
    ps = np.array([x[3] for x in pares])
    sig = bh(ps, 0.05)
    imax = int(np.argmax(absr))
    res = {
        "n_redes": n,
        "n_pares_validos": len(pares),
        "shape_X1": list(X1.shape), "shape_X2": list(X2.shape),
        "mediana_abs_rho": round(float(np.median(absr)), 3),
        "q25_abs_rho": round(float(np.percentile(absr, 25)), 3),
        "q75_abs_rho": round(float(np.percentile(absr, 75)), 3),
        "max_abs_rho": round(float(absr.max()), 3),
        "p_del_max": float(pares[imax][3]),
        "par_del_max": [int(pares[imax][0]), int(pares[imax][1])],
        "pares_significativos_BH_q05": int(sig.sum()),
        "frac_significativos_BH": round(float(sig.mean()), 3),
        "frac_abs_rho_menor_030": round(float((absr < 0.30).mean()), 3),
        "frac_abs_rho_mayor_060": round(float((absr > 0.60).mean()), 3),
    }
    resultados[dom] = res
    print(f"[{dom}] n={n}  pares={len(pares)}")
    print(f"   mediana|rho|={res['mediana_abs_rho']}  IQR=[{res['q25_abs_rho']},{res['q75_abs_rho']}]  max={res['max_abs_rho']} (p={res['p_del_max']:.2e})")
    print(f"   significativos BH(q<=0.05): {res['pares_significativos_BH_q05']}/{len(pares)} ({res['frac_significativos_BH']*100:.0f}%)")
    print(f"   %|rho|<0.30: {res['frac_abs_rho_menor_030']*100:.0f}%   %|rho|>0.60: {res['frac_abs_rho_mayor_060']*100:.0f}%")
    print()

out = BASE + "/redundancia_vistas/resultados_redundancia.json"
json.dump(resultados, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("Guardado:", out)
