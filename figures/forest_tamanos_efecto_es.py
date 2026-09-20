# -*- coding: utf-8 -*-
"""
F3 - Forest plot de tamanos de efecto por descriptor y dominio.

Panel izquierdo : d de Cohen (tercil superior vs inferior de exactitud global)
                  con IC 95% bootstrap almacenado en summary_leakfree.json.
Panel derecho   : |r| de Mann-Whitney (punto, sin IC almacenado en el JSON).

Fuente EXCLUSIVA de datos (seccion "full" de cada dominio):
  outputs/fase11_results/summary_leakfree.json  (CIFAR-10)
  outputs/fase12_results/summary_leakfree.json  (CIFAR-100)
  outputs/fase13_results/summary_leakfree.json  (SVHN)
No se grafica la seccion within_cnn (IC degenerados por n=10).
No se inventa ningun valor: todo se lee del JSON.

Salida: C:/Users/dcrr1/ARTICULO_TUTORA/figures/forest_tamanos_efecto.pdf
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- utf-8 en consola Windows -------------------------------------------------
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---- rutas --------------------------------------------------------------------
BASE = r"D:/OneDrive/Desktop/proyecto de tesis espoch/outputs"
OUTDIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTDIR, exist_ok=True)
OUTPDF = os.path.join(OUTDIR, "forest_tamanos_efecto.pdf")

DOMINIOS = [
    ("CIFAR-10", os.path.join(BASE, "fase11_results", "summary_leakfree.json")),
    ("CIFAR-100", os.path.join(BASE, "fase12_results", "summary_leakfree.json")),
    ("SVHN", os.path.join(BASE, "fase13_results", "summary_leakfree.json")),
]
DESCRIPTORES = ["X1", "X2", "X3_orig", "X3_adapt", "X3_rank"]
ETIQUETAS = {
    "X1": r"$X_1$",
    "X2": r"$X_2$",
    "X3_orig": r"$X_3^{(c)}$",
    "X3_adapt": r"$X_3^{(a)}$",
    "X3_rank": r"$X_3^{(r)}$",
}

# ---- paleta sobria (daltonismo / escala de grises) ----------------------------
AZUL = "#4477AA"
NARANJA = "#EE7733"
GRIS = "#777777"
COLOR = {
    "X1": GRIS,
    "X2": AZUL,
    "X3_orig": NARANJA,
    "X3_adapt": NARANJA,
    "X3_rank": NARANJA,
}
MARCA = {"X1": "o", "X2": "s", "X3_orig": "D", "X3_adapt": "^", "X3_rank": "v"}

Q_BH = 0.05
M_PRUEBAS = 10  # 5 Kendall + 5 Mann-Whitney por dominio
ALFA_BONF = Q_BH / M_PRUEBAS  # 0.005


def umbral_bh(pvals, q=Q_BH):
    """Devuelve el p-valor critico de Benjamini-Hochberg (0.0 si nada es significativo)."""
    p = np.sort(np.asarray(pvals, dtype=float))
    m = p.size
    ranks = np.arange(1, m + 1)
    ok = p <= (ranks / m) * q
    return float(p[ok][-1]) if ok.any() else 0.0


# ---- carga --------------------------------------------------------------------
datos = []   # filas del forest plot
resumen = {}  # para el reporte por consola
for dom, ruta in DOMINIOS:
    with open(ruta, "r", encoding="utf-8") as fh:
        full = json.load(fh)["full"]
    pvals = [full[k]["tau_p"] for k in DESCRIPTORES] + [full[k]["mw_p"] for k in DESCRIPTORES]
    p_crit = umbral_bh(pvals)
    resumen[dom] = {"p_crit_bh": p_crit, "_bh": full["_bh"], "filas": []}
    for k in DESCRIPTORES:
        v = full[k]
        d, lo, hi = v["cohens_d"]
        fila = dict(
            dominio=dom, desc=k, d=float(d), lo=float(lo), hi=float(hi),
            mw_r=float(v["mw_r"]), mw_p=float(v["mw_p"]),
            sig_bh=bool(v["mw_p"] <= p_crit),
            sig_bonf=bool(v["mw_p"] <= ALFA_BONF),
        )
        datos.append(fila)
        resumen[dom]["filas"].append(fila)

# ---- posiciones verticales (dominios separados por un hueco) ------------------
ypos, yticks, ylabels, sep, grupos = [], [], [], [], []
y = 0.0
for gi, (dom, _) in enumerate(DOMINIOS):
    y0 = y
    for k in DESCRIPTORES:
        ypos.append(y)
        yticks.append(y)
        ylabels.append(ETIQUETAS[k])
        y += 1.0
    grupos.append((dom, y0, y - 1.0))
    if gi < len(DOMINIOS) - 1:
        sep.append(y - 0.5 + 0.25)
        y += 1.1
ypos = np.array(ypos)

# ---- figura -------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
})
fig, (axd, axr) = plt.subplots(
    1, 2, figsize=(6.5, 4.4), sharey=True,
    gridspec_kw={"width_ratios": [1.85, 1.0], "wspace": 0.10},
)

# --- panel A: d de Cohen con IC 95% bootstrap ---
for yy, f in zip(ypos, datos):
    c = COLOR[f["desc"]]
    axd.plot([f["lo"], f["hi"]], [yy, yy], color=c, lw=1.3, solid_capstyle="butt", zorder=2)
    for x in (f["lo"], f["hi"]):
        axd.plot([x, x], [yy - 0.17, yy + 0.17], color=c, lw=1.0, zorder=2)
    axd.plot(f["d"], yy, marker=MARCA[f["desc"]], ms=4.6, color=c,
             mfc=c, mec=c, ls="none", zorder=3)

y_ref = ypos[2]  # fila central del primer grupo (zona sin datos)
axd.axvline(0.0, color=GRIS, ls="--", lw=0.8, zorder=1)
axd.axvline(0.8, color=GRIS, ls=":", lw=0.9, zorder=1)
axd.text(-0.12, y_ref, u"d = 0", color=GRIS, fontsize=7,
         rotation=90, ha="center", va="center")
axd.text(0.68, y_ref, u"d = 0,8 (grande)", color=GRIS, fontsize=7,
         rotation=90, ha="center", va="center")
axd.set_xlabel(u"d de Cohen (adimensional)")
axd.set_xlim(-0.75, 6.85)
axd.set_xticks([0, 1, 2, 3, 4, 5, 6])

# --- panel B: |r| de Mann-Whitney (punto, sin IC) ---
for yy, f in zip(ypos, datos):
    c = COLOR[f["desc"]]
    relleno = c if f["sig_bh"] else "white"
    axr.plot(f["mw_r"], yy, marker=MARCA[f["desc"]], ms=4.6, color=c,
             mfc=relleno, mec=c, mew=1.0, ls="none", zorder=3)

axr.axvline(0.5, color=GRIS, ls=":", lw=0.9, zorder=1)
axr.text(0.455, y_ref, u"|r| = 0,5", color=GRIS, fontsize=7,
         rotation=90, ha="center", va="center")
axr.set_xlabel(u"|r| de Mann-Whitney (0-1)")
axr.set_xlim(0.0, 1.12)
axr.set_xticks([0.0, 0.5, 1.0])
axr.set_xticklabels(["0,0", "0,5", "1,0"])

# --- ejes comunes ---
for ax in (axd, axr):
    ax.set_ylim(ypos.max() + 1.05, ypos.min() - 0.75)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for ys in sep:
        ax.axhline(ys, color="#DDDDDD", lw=0.7, zorder=0)

axd.set_yticks(yticks)
axd.set_yticklabels(ylabels)
axr.spines["left"].set_visible(False)

# etiquetas de dominio a la izquierda de cada grupo
for dom, y0, y1 in grupos:
    axd.text(-0.30, (y0 + y1) / 2.0, dom, transform=axd.get_yaxis_transform(),
             rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")

# leyenda: relleno = significativo BH
h_sig = plt.Line2D([], [], marker="o", ls="none", ms=4.6, color="#333333",
                   mfc="#333333", mec="#333333")
h_ns = plt.Line2D([], [], marker="o", ls="none", ms=4.6, color="#333333",
                  mfc="white", mec="#333333", mew=1.0)
fig.legend([h_sig, h_ns],
           [u"Mann-Whitney significativo (BH, q $\\leq$ 0,05)",
            u"no significativo"],
           loc="lower center", bbox_to_anchor=(0.55, 0.005), ncol=2,
           frameon=False, handletextpad=0.4, columnspacing=1.6)

fig.tight_layout()
fig.subplots_adjust(bottom=0.185, left=0.16, wspace=0.10)
fig.savefig(OUTPDF, format="pdf", bbox_inches="tight")
plt.close(fig)

# ---- reporte por consola (valores exactos graficados) -------------------------
print("PDF guardado en:", OUTPDF)
print()
hdr = "{:<10} {:<11} {:>7} {:>8} {:>8} {:>7} {:>11} {:>4} {:>5}"
print(hdr.format("dominio", "descriptor", "d", "lo95", "hi95", "|r|MW", "p_MW", "BH", "Bonf"))
for dom, _ in DOMINIOS:
    info = resumen[dom]
    for f in info["filas"]:
        print(hdr.format(dom, f["desc"], "%.3f" % f["d"], "%.3f" % f["lo"], "%.3f" % f["hi"],
                         "%.3f" % f["mw_r"], "%.3e" % f["mw_p"],
                         "si" if f["sig_bh"] else "NO",
                         "si" if f["sig_bonf"] else "NO"))
    print("  -> BH p_critico = %.3e ; bloque _bh del JSON = %s" % (info["p_crit_bh"], info["_bh"]))

ds = [f["d"] for f in datos]
los = [f["lo"] for f in datos]
print()
print("n filas graficadas = %d (5 descriptores x 3 dominios)" % len(datos))
print("d de Cohen: min %.3f (%s/%s), max %.3f (%s/%s), mediana %.3f" % (
    min(ds), datos[int(np.argmin(ds))]["dominio"], datos[int(np.argmin(ds))]["desc"],
    max(ds), datos[int(np.argmax(ds))]["dominio"], datos[int(np.argmax(ds))]["desc"],
    float(np.median(ds))))
print("IC 95 %% que excluyen 0 (lo95 > 0): %d de %d" % (sum(1 for l in los if l > 0), len(los)))
print("IC 95 %% con lo95 > 0,8 (efecto grande garantizado): %d de %d" %
      (sum(1 for l in los if l > 0.8), len(los)))
print("|r| MW: min %.3f, max %.3f, mediana %.3f" % (
    min(f["mw_r"] for f in datos), max(f["mw_r"] for f in datos),
    float(np.median([f["mw_r"] for f in datos]))))
print("MW significativos BH: %d/%d ; Bonferroni (p <= %.4f): %d/%d" % (
    sum(1 for f in datos if f["sig_bh"]), len(datos), ALFA_BONF,
    sum(1 for f in datos if f["sig_bonf"]), len(datos)))
