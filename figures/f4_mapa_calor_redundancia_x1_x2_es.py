# -*- coding: utf-8 -*-
"""
Figura F4 — Mapa de calor 10x12 de |rho| de Spearman entre los rasgos de la
vista X1 (grafo funcional, 10 rasgos) y los de la vista X2 (homologia
persistente, 12 rasgos), por dominio (CIFAR-10, CIFAR-100, SVHN).

Prueba: correlacion de Spearman por par de rasgos sobre las n=40 redes de cada
dominio + correccion de Benjamini-Hochberg (q<=0.05) aplicada a la familia
completa de pares del dominio (110 pares en CIFAR-10 y SVHN, 120 en CIFAR-100;
en CIFAR-10 y SVHN el rasgo "n. de clases H0" es constante y sus 10 pares no
son evaluables).

Datos: se recomputan de forma determinista (Spearman exacto, sin remuestreo ni
azar) desde los mismos tres .pkl crudos que usa
outputs/redundancia_vistas/calc_redundancia_vistas.py. El script verifica que
los estadisticos de resumen coinciden EXACTAMENTE con los ya publicados en
outputs/redundancia_vistas/resultados_redundancia.json antes de dibujar; si no
coinciden, aborta sin generar la figura.

Salidas:
  - PDF vectorial: C:/Users/dcrr1/ARTICULO_TUTORA/figures/f4_mapa_calor_redundancia_x1_x2.pdf
  - Volcado de las matrices: outputs/redundancia_vistas/matriz_redundancia_10x12.json
"""
import io
import os
import sys
import json
import pickle
import warnings

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

warnings.filterwarnings("ignore")
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Rutas ────────────────────────────────────────────────────────────────
BASE = r"D:/OneDrive/Desktop/proyecto de tesis espoch/outputs"
DOMINIOS = {
    "CIFAR-10":  BASE + "/fase16_cifar10_raw/all_records_f16.pkl",
    "CIFAR-100": BASE + "/fase12_cifar100/all_records_f12.pkl",
    "SVHN":      BASE + "/fase13_svhn/all_records_f13.pkl",
}
JSON_REF = BASE + "/redundancia_vistas/resultados_redundancia.json"
JSON_OUT = BASE + "/redundancia_vistas/matriz_redundancia_10x12.json"
DIR_FIG = os.path.dirname(os.path.abspath(__file__))
PDF_OUT = os.path.join(DIR_FIG, "f4_mapa_calor_redundancia_x1_x2_es.pdf")

# ── Nombres de los rasgos (orden exacto de desc_x1 / desc_x2 en
#    phase16_reextract.py, identico en fase12 y fase13) ────────────────────
NOM_X1 = [
    "densidad", "agrupamiento medio", "transitividad", "grado medio",
    "grado DE", "grado máx./N", "componentes/N", "aristas/N",
    "intermediación media", "intermediación DE",
]
NOM_X2 = [
    "$\\beta_0$ pers. media", "$\\beta_0$ pers. DE", "$\\beta_0$ pers. máx.", "$\\beta_0$ pers. suma",
    "$\\beta_0$ n.º clases", "$\\beta_0$ pers. p75",
    "$\\beta_1$ pers. media", "$\\beta_1$ pers. DE", "$\\beta_1$ pers. máx.", "$\\beta_1$ pers. suma",
    "$\\beta_1$ n.º clases", "$\\beta_1$ pers. p75",
]

# ── Paleta sobria, apta para daltonismo e imprimible en gris ─────────────
AZUL, NARANJA, GRIS = "#4477AA", "#EE7733", "#777777"
CMAP = LinearSegmentedColormap.from_list(
    "azul_secuencial", ["#FFFFFF", "#C6D6E6", "#8FB0CC", AZUL, "#2A5075", "#12314A"]
)
CMAP.set_bad("#E8E8E8")  # celdas no evaluables (rasgo constante)


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: devuelve mascara de rechazo (mismo codigo que
    calc_redundancia_vistas.py)."""
    p = np.asarray(pvals)
    m = len(p)
    orden = np.argsort(p)
    umbral = q * (np.arange(1, m + 1) / m)
    pasa = p[orden] <= umbral
    k = np.max(np.nonzero(pasa)[0]) + 1 if pasa.any() else 0
    rechazo = np.zeros(m, bool)
    rechazo[orden[:k]] = True
    return rechazo


# ══ 1. Recomputo determinista de la matriz por dominio ═══════════════════
res = {}
for dom, ruta in DOMINIOS.items():
    data = pickle.load(open(ruta, "rb"))
    k1 = "x1_g" if "x1_g" in data[0] else "x1"
    k2 = "x2_g" if "x2_g" in data[0] else "x2"
    X1 = np.array([np.asarray(r[k1]).ravel() for r in data])
    X2 = np.array([np.asarray(r[k2]).ravel() for r in data])
    n, d1 = X1.shape
    d2 = X2.shape[1]

    RHO = np.full((d1, d2), np.nan)
    P = np.full((d1, d2), np.nan)
    pares = []
    for i in range(d1):
        for j in range(d2):
            rho, p = spearmanr(X1[:, i], X2[:, j])
            if not np.isnan(rho):
                RHO[i, j], P[i, j] = float(rho), float(p)
                pares.append((i, j, float(rho), float(p)))

    ps = np.array([x[3] for x in pares])
    absr = np.array([abs(x[2]) for x in pares])
    sig_v = bh(ps, 0.05)
    SIG = np.zeros((d1, d2), bool)
    for (i, j, _, _), s in zip(pares, sig_v):
        SIG[i, j] = bool(s)
    imax = int(np.argmax(absr))

    res[dom] = dict(
        n=n, RHO=RHO, P=P, SIG=SIG,
        n_pares=len(pares),
        mediana=float(np.median(absr)),
        q25=float(np.percentile(absr, 25)),
        q75=float(np.percentile(absr, 75)),
        maximo=float(absr.max()),
        p_max=float(pares[imax][3]),
        rho_max=float(pares[imax][2]),
        par_max=(int(pares[imax][0]), int(pares[imax][1])),
        n_sig=int(sig_v.sum()),
        frac_sig=float(sig_v.mean()),
        frac_lt030=float((absr < 0.30).mean()),
        frac_gt060=float((absr > 0.60).mean()),
        # |rho| minimo que sobrevive a BH: umbral efectivo del dominio
        rho_min_sig=float(absr[sig_v].min()) if sig_v.any() else float("nan"),
        p_max_sig=float(ps[sig_v].max()) if sig_v.any() else float("nan"),
    )

# ══ 2. Verificacion contra el JSON ya publicado ══════════════════════════
ref = json.load(open(JSON_REF, encoding="utf-8"))
print("Verificacion contra resultados_redundancia.json")
fallo = False
for dom, r in res.items():
    e = ref[dom]
    chks = [
        ("n_redes", r["n"], e["n_redes"]),
        ("n_pares", r["n_pares"], e["n_pares_validos"]),
        ("mediana", round(r["mediana"], 3), e["mediana_abs_rho"]),
        ("q25", round(r["q25"], 3), e["q25_abs_rho"]),
        ("q75", round(r["q75"], 3), e["q75_abs_rho"]),
        ("max", round(r["maximo"], 3), e["max_abs_rho"]),
        ("n_sig_BH", r["n_sig"], e["pares_significativos_BH_q05"]),
        ("par_max", list(r["par_max"]), e["par_del_max"]),
    ]
    for nombre, got, exp in chks:
        ok = (got == exp)
        fallo |= (not ok)
        print(f"  [{'OK ' if ok else 'MAL'}] {dom:<10s} {nombre:<10s} calc={got}  json={exp}")
if fallo:
    sys.exit("ABORTADO: el recomputo no coincide con resultados_redundancia.json")
print("  -> coincidencia exacta en los tres dominios.\n")

# ══ 3. Volcado de la matriz completa (lo que el JSON de resumen no guarda) ══
volcado = {
    "descripcion": "Matrices 10x12 de rho de Spearman entre rasgos de X1 (filas) "
                   "y X2 (columnas), p-valores y rechazo BH q<=0.05 por dominio. "
                   "null = par no evaluable (rasgo constante).",
    "nombres_X1": NOM_X1, "nombres_X2": NOM_X2,
}
for dom, r in res.items():
    volcado[dom] = {
        "rho": [[None if np.isnan(v) else round(float(v), 6) for v in fila] for fila in r["RHO"]],
        "p": [[None if np.isnan(v) else float(v) for v in fila] for fila in r["P"]],
        "significativo_BH_q05": [[bool(v) for v in fila] for fila in r["SIG"]],
    }
pass  # (volcado JSON ya existe en outputs/redundancia_vistas)
print("Matriz volcada en:", JSON_OUT, "\n")

# ══ 4. Figura ════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
})

fig, axes = plt.subplots(1, 3, figsize=(6.9, 4.15), sharey=True)
fig.subplots_adjust(left=0.155, right=0.868, bottom=0.335, top=0.845, wspace=0.10)
VMAX = 0.85  # cubre el maximo global observado (0,829)

for ax, (dom, r) in zip(axes, res.items()):
    A = np.abs(r["RHO"])
    im = ax.imshow(np.ma.masked_invalid(A), cmap=CMAP, vmin=0.0, vmax=VMAX,
                   aspect="auto", interpolation="nearest")

    # Marcas BH: punto en cada par significativo (q<=0.05); color segun el
    # fondo para que siga leyendose en escala de grises.
    ii, jj = np.nonzero(r["SIG"])
    for i, j in zip(ii, jj):
        col = "white" if A[i, j] >= 0.45 else "#1A1A1A"
        ax.plot(j, i, marker="o", ms=1.9, mfc=col, mec=col, lw=0)

    # Recuadro naranja sobre el par de |rho| maximo del dominio
    i0, j0 = r["par_max"]
    ax.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False,
                           edgecolor=NARANJA, lw=1.2, zorder=5))

    ax.set_xticks(range(12))
    ax.set_xticklabels(NOM_X2, rotation=90)
    ax.set_yticks(range(10))
    ax.set_yticklabels(NOM_X1)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.35)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=1.8, width=0.5)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title(dom, pad=22)
    med = f"{r['mediana']:.3f}".replace(".", ",")
    ax.text(0.5, 1.010, f"mediana $|\\rho|$ = {med}\n{r['n_sig']}/{r['n_pares']} pares sig. BH",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8,
            color="#333333", linespacing=1.3)

axes[0].set_ylabel("Rasgo de $X_1$ (grafo funcional)", labelpad=3)
fig.text(0.52, 0.115, "Rasgo de $X_2$ (homología persistente)",
         ha="center", va="center", fontsize=9)

# Barra de color compartida + linea de referencia gris discontinua en 0,30
cax = fig.add_axes([0.900, 0.335, 0.017, 0.51])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("$|\\rho|$ de Spearman (adimensional)", labelpad=6)
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8])
cbar.ax.set_yticklabels(["0,0", "0,2", "0,4", "0,6", "0,8"], fontsize=6.4)
# Linea de referencia gris discontinua: umbral convencional |rho| = 0,30
cbar.ax.axhline(0.30, color=GRIS, ls="--", lw=0.9)
cax.yaxis.set_minor_locator(FixedLocator([0.30]))
cax.yaxis.set_minor_formatter(FixedFormatter(["0,30"]))
cax.tick_params(axis="y", which="minor", labelsize=6.2, labelcolor=GRIS,
                color=GRIS, length=1.8, width=0.5)
cbar.outline.set_visible(False)
cbar.ax.tick_params(which="major", length=1.8, width=0.5)

handles = [
    Line2D([0], [0], marker="o", ms=3, mfc="#1A1A1A", mec="#1A1A1A", lw=0,
           label="par significativo (BH, $q\\leq0{,}05$)"),
    Line2D([0], [0], marker="s", ms=6, mfc="none", mec=NARANJA, mew=1.2, lw=0,
           label="par de $|\\rho|$ máximo del dominio"),
    Line2D([0], [0], marker="s", ms=6, mfc="#E8E8E8", mec="#E8E8E8", lw=0,
           label="par no evaluable (rasgo constante)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.52, 0.005), handletextpad=0.4, columnspacing=1.6)

os.makedirs(DIR_FIG, exist_ok=True)
fig.savefig(PDF_OUT, format="pdf")
plt.close(fig)
print("Figura guardada:", PDF_OUT)

# ══ 5. Volcado de todos los numeros que muestra la figura ════════════════
print("\n" + "=" * 74)
print("VALORES QUE MUESTRA LA FIGURA")
print("=" * 74)
for dom, r in res.items():
    i0, j0 = r["par_max"]
    print(f"\n[{dom}]  n={r['n']} redes  |  pares evaluables={r['n_pares']}/120")
    print(f"  mediana |rho| = {r['mediana']:.3f}   IQR = [{r['q25']:.3f}, {r['q75']:.3f}]")
    print(f"  |rho| max = {r['maximo']:.3f}  (rho={r['rho_max']:+.3f}, p={r['p_max']:.3e})")
    print(f"     par maximo: X1[{i0}] '{NOM_X1[i0]}'  x  X2[{j0}] '{NOM_X2[j0]}'")
    print(f"  significativos BH q<=0.05: {r['n_sig']}/{r['n_pares']} ({r['frac_sig']*100:.1f}%)")
    print(f"     |rho| minimo que pasa BH = {r['rho_min_sig']:.3f}  (p mayor aceptado = {r['p_max_sig']:.3e})")
    print(f"  %|rho|<0.30: {r['frac_lt030']*100:.1f}%    %|rho|>0.60: {r['frac_gt060']*100:.1f}%")
    A = np.abs(r["RHO"])
    fin = A[np.isfinite(A)]
    print(f"  min |rho| = {fin.min():.3f}   media |rho| = {fin.mean():.3f}")
    # 3 pares mas fuertes
    idx = np.dstack(np.unravel_index(np.argsort(-np.nan_to_num(A, nan=-1), axis=None), A.shape))[0][:3]
    for k, (i, j) in enumerate(idx, 1):
        print(f"     top{k}: |rho|={A[i,j]:.3f}  X1[{i}] {NOM_X1[i]} x X2[{j}] {NOM_X2[j]}  "
              f"(rho={r['RHO'][i,j]:+.3f}, p={r['P'][i,j]:.2e}, BH={'si' if r['SIG'][i,j] else 'no'})")

tot_sig = sum(r["n_sig"] for r in res.values())
tot_par = sum(r["n_pares"] for r in res.values())
print(f"\n[GLOBAL] {tot_sig}/{tot_par} pares significativos BH "
      f"({tot_sig/tot_par*100:.1f}%) sobre los tres dominios")
todas = np.concatenate([np.abs(r["RHO"])[np.isfinite(r["RHO"])] for r in res.values()])
print(f"[GLOBAL] mediana |rho| agrupada = {np.median(todas):.3f}   "
      f"maximo global = {todas.max():.3f}   %|rho|<0.30 = {(todas<0.30).mean()*100:.1f}%")
