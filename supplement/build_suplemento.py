# -*- coding: utf-8 -*-
"""Genera ARTICULO_TESIS_V9/SUPLEMENTO/{suplemento.tex, fig_solapamiento.pdf}
a partir de los JSON de resultados (sin recomputar nada)."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(r"C:\Users\dcrr1\OneDrive\Desktop\ARTICULO_TESIS_V9\SUPLEMENTO"); OUT.mkdir(exist_ok=True)
O = Path(r"D:\OneDrive\Desktop\proyecto de tesis espoch\outputs")
coma = lambda v, nd=3: (f"{v:.{nd}f}".replace(".", ",")).replace("-", "$-$")
def pfmt(p):
    if p < 1e-4: return "$<$0,0001"
    return coma(p, 4)
def ci(t, nd=3):
    if max(abs(x) for x in t) > 50: return "n.\\,d.\\textsuperscript{*}"
    return f"{coma(t[0],nd)} [{coma(t[1],2)}; {coma(t[2],2)}]"

# ---------- tablas por dominio ----------
DOM = [("CIFAR-10", "fase11_results"), ("CIFAR-100", "fase12_results"), ("SVHN", "fase13_results")]
CTX = [("full", "Población completa ($n{=}40$)"), ("within_mlp", "Solo perceptrones multicapa ($n{=}30$)"),
       ("within_cnn", "Solo redes convolucionales ($n{=}10$)"), ("y_norm", "Exactitud normalizada por tipo de red ($Y_{\\mathrm{norm}}$)")]
DESC = [("X1", "$X_1$ (grafo funcional)"), ("X2", "$X_2$ (homología persistente)"),
        ("X3_orig", "$X_3^{(\\mathrm{c})}$ (pesos iguales)"), ("X3_adapt", "$X_3^{(\\mathrm{a})}$ (adaptativa)"),
        ("X3_rank", "$X_3^{(\\mathrm{r})}$ (adaptativa, ponderación por rangos)")]
tabs = []
for dname, folder in DOM:
    d = json.load(open(O / folder / "summary_leakfree.json"))
    ar = d["acc_ranges"]
    tabs.append(f"\\subsection*{{{dname}}}\n"
                f"Rango de exactitud de prueba en el dominio: {coma(ar['total_pp'],1)} puntos porcentuales en total "
                f"({coma(ar['mlp_pp'],1)} entre perceptrones multicapa y {coma(ar['cnn_pp'],1)} entre redes convolucionales).\n")
    for ck, clabel in CTX:
        c = d[ck]; bh = c["_bh"]
        rows = []
        for dk, dlabel in DESC:
            r = c[dk]
            rows.append(f"{dlabel} & {ci(r['family_auc'])} & {ci(r['pairwise_auc'])} & {ci(r['kendall_tau'])} & {pfmt(r['tau_p'])} & "
                        f"{ci(r['cohens_d'],2)} & {coma(r['mw_r'],2)} & {pfmt(r['mw_p'])} \\\\")
        tabs.append(
            "\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
            f"\\caption{{{dname}, {clabel.replace('$n{=}','$n{=}')}. Media [IC 95\\,\\%] sobre la validación cruzada repetida "
            "(AUC familiar y $\\tau$) y sobre 300 remuestreos bootstrap (AUC por pares); $d$ de Cohen y prueba de Mann--Whitney "
            f"entre familias extremas. Contrastes significativos tras Benjamini--Hochberg: {bh['n_sig']}/{bh['n_total']} "
            f"(Bonferroni: {bh['n_bonf']}).}}\n"
            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l c c c c c c c}\\toprule\n"
            "Descriptor & AUC familiar & AUC por pares & $\\tau$ de Kendall & $p_\\tau$ & $d$ de Cohen & $r_{\\mathrm{MW}}$ & $p_{\\mathrm{MW}}$ \\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")
TABLAS = "\n".join(tabs)

# ---------- solapamiento ----------
red = json.load(open(O / "redundancia_vistas" / "resultados_redundancia.json"))
mat = json.load(open(O / "redundancia_vistas" / "matriz_redundancia_10x12.json"))
rows = []
for dn in ["CIFAR-10", "CIFAR-100", "SVHN"]:
    r = red[dn]
    rows.append(f"{dn} & {r['n_pares_validos']} & {coma(r['mediana_abs_rho'])} & [{coma(r['q25_abs_rho'],2)}; {coma(r['q75_abs_rho'],2)}] & "
                f"{coma(r['max_abs_rho'])} & {r['pares_significativos_BH_q05']} ({coma(100*r['frac_significativos_BH'],1)}\\,\\%) & "
                f"{coma(100*r['frac_abs_rho_menor_030'],1)}\\,\\% & {coma(100*r['frac_abs_rho_mayor_060'],1)}\\,\\% \\\\")
TAB_SOLAP = ("\\begin{table}[H]\\centering\\small\n\\caption{Solapamiento entre las vistas $X_1$ y $X_2$ por dominio. "
             "Correlación de Spearman entre cada par de rasgos sobre las $n{=}40$ redes; los pares con algún rasgo constante no son evaluables.}\n"
             "\\label{tab:solap}\\resizebox{\\linewidth}{!}{\\begin{tabular}{l c c c c c c c}\\toprule\n"
             "Dominio & Pares & Mediana $|\\rho|$ & RIC & Máx.\\ $|\\rho|$ & Signif.\\ BH ($q{\\le}0{,}05$) & $|\\rho|<0{,}3$ & $|\\rho|>0{,}6$ \\\\\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")

plt.rcParams.update({"font.family": "serif", "font.size": 7, "savefig.dpi": 600, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
for ax, dn in zip(axes, ["CIFAR-10", "CIFAR-100", "SVHN"]):
    rho = np.array([[np.nan if v is None else v for v in row] for row in mat[dn]["rho"]], float)
    sig = np.array([[bool(v) if v is not None else False for v in row] for row in mat[dn]["significativo_BH_q05"]])
    im = ax.imshow(np.abs(rho), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            if np.isnan(rho[i, j]): ax.text(j, i, "–", ha="center", va="center", fontsize=5, color="grey"); continue
            ax.text(j, i, coma(abs(rho[i, j]), 2).replace("$-$", ""), ha="center", va="center", fontsize=4.3,
                    color="white" if abs(rho[i, j]) > 0.55 else "black", fontweight="bold" if sig[i, j] else "normal")
            if sig[i, j]: ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor="black", lw=0.5))
    ax.set_title(dn, fontsize=8); ax.set_xticks(range(12)); ax.set_xticklabels(mat["nombres_X2"], rotation=90, fontsize=5.5)
    ax.set_yticks(range(10)); ax.set_yticklabels(mat["nombres_X1"], fontsize=5.5)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02); cb.set_label("$|\\rho|$ de Spearman", fontsize=7)
cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: coma(v, 1))); cb.ax.tick_params(labelsize=6)
fig.savefig(OUT / "fig_solapamiento.pdf")
print("figura ok")

# ---------- barrido de sensibilidad (valores del registro run_phase17_v2.log) ----------
SWEEP = r"""
\begin{table}[H]\centering\small
\caption{Barrido de sensibilidad del AUC familiar de $X_3$ en CIFAR-10 ($n{=}40$; validación cruzada estratificada de cinco particiones repetida 60 veces, regresión logística uno contra el resto). Referencia: $\theta{=}0{,}4$, correlación de Pearson, tope de 80 neuronas, $N{=}50$ ejemplos, $r{=}15$ componentes.}
\label{tab:barrido}
\begin{tabular}{l l c}\toprule
Parámetro & Valores & AUC familiar \\\midrule
\multirow{5}{*}{Umbral $\theta$} & 0,2 & 0,871 \\ & 0,3 & 0,871 \\ & \textbf{0,4 (referencia)} & \textbf{0,862} \\ & 0,5 & 0,841 \\ & 0,6 & 0,856 \\
\midrule
\multirow{2}{*}{Tipo de correlación} & Pearson (referencia) & 0,862 \\ & Spearman & 0,832 \\
\midrule
\multirow{3}{*}{Tope de neuronas} & 40 & 0,812 \\ & 80 (referencia) & 0,862 \\ & 200 & 0,871 \\
\midrule
\multirow{3}{*}{Ejemplos $N$ para la homología} & 50 (referencia) & 0,862 \\ & 100 & 0,862 \\ & 200 & 0,862 \\
\midrule
\multirow{3}{*}{Componentes PCA $r$} & 8 & 0,862 \\ & 12 & 0,862 \\ & 15 (referencia) & 0,862 \\
\midrule
\multirow{3}{*}{Capa analizada (redes con $\ge 3$ capas, $n{=}29$)} & primera (\texttt{relu\_1}) & 0,891 \\ & intermedia (\texttt{relu\_4}) & 0,869 \\ & última (\texttt{relu\_7}) & 0,819 \\
\bottomrule\end{tabular}\end{table}
"""

TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-tabla]{babel}
\usepackage[margin=2.3cm]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools,bm}
\usepackage{booktabs,multirow,float,graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage[spanish,capitalize,nameinlink]{cleveref}
\crefname{table}{Tabla}{Tablas}\Crefname{table}{Tabla}{Tablas}
\crefname{section}{sección}{secciones}\Crefname{section}{Sección}{Secciones}
\graphicspath{{./}}
\theoremstyle{plain}
\newtheorem{proposition}{Proposición}
\newtheorem{theorem}{Teorema}
\newtheorem{corollary}{Corolario}
\theoremstyle{remark}
\newtheorem{remark}{Observación}
\newcommand{\X}[1]{\mathbf{X}_{#1}}
\newcommand{\Xth}{\mathbf{X}_3}
\newcommand{\Dgm}[1]{\mathrm{Dgm}_{#1}}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
\newcommand{\R}{\mathbb{R}}
\title{Material suplementario\\[4pt]\large Descriptor multivista de activaciones neuronales, mediante la fusión de grafos funcionales y homología persistente}
\author{César Daniel Reinoso Reinoso \and Amalia Isabel Escudero Villa\\[2pt]\normalsize Escuela Superior Politécnica de Chimborazo (ESPOCH), Riobamba, Ecuador}
\date{Septiembre de 2026 \\[2pt] \small Repositorio: \url{https://github.com/Dcrr1717/multiview-nn-descriptor}}
\begin{document}
\maketitle

\section*{Propósito}
Este documento acompaña al artículo enviado a la \emph{Revista Bases de la Ciencia} y reúne el material que, por el límite de extensión de la revista, se remitió al repositorio: los enunciados y las demostraciones de las propiedades matemáticas de los descriptores (\cref{sec:prop}), el estudio de solapamiento entre las dos vistas (\cref{sec:solap}), el barrido de sensibilidad sobre el umbral $\theta$ y los demás parámetros fijados a priori (\cref{sec:barrido}) y las tablas completas de resultados por dominio y por contexto de análisis (\cref{sec:tablas}). Todos los valores proceden de los archivos de resultados del repositorio (\texttt{outputs/}) y se escriben con coma decimal, como en el artículo.

\section{Notación}
Sea $A \in \R^{N\times d}$ la matriz de activaciones de una capa oculta, con $N$ ejemplos de evaluación en las filas y $d$ neuronas en las columnas. La vista de grafo funcional $\X{1}\in\R^{10}$ se obtiene de la matriz de correlaciones de Pearson $C$ entre las primeras $m=\min(d,80)$ neuronas, umbralizada con $E_{ij}=\mathbf{1}[|C_{ij}|>\theta]$, $\theta=0{,}4$, y resumida mediante diez medidas de grafo,
\begin{equation}
  \X{1} = \bigl[\rho,\;\bar{C},\;\mathcal{T},\;\bar{k},\;\sigma_k,\;k_{\max}/|V|,\;n_{\mathrm{cc}}/|V|,\;|E|/|V|,\;\bar{c}_{\mathrm{B}},\;\sigma_{c_{\mathrm{B}}}\bigr]^\top,
  \label{eq:x1}
\end{equation}
es decir, densidad, agrupamiento medio, transitividad, media, desviación típica y máximo normalizado del grado, fracción de componentes conexas, razón aristas/nodos y media y desviación típica de la centralidad de intermediación. La vista de homología persistente $\X{2}\in\R^{12}$ trata las filas de $A[:50]$, reducidas por PCA a $r=\min(15,d,N-1)$ componentes y reescaladas a $[0,1]$, como una nube de puntos $P$, construye la filtración de Vietoris--Rips $\{\mathcal{VR}(P,\epsilon)\}_{\epsilon\ge 0}$ y resume cada diagrama de persistencia $\Dgm{k}(P)=\{(b_j^k,d_j^k)\}_j$, $k\in\{0,1\}$, con persistencias $p_j^k=d_j^k-b_j^k$, mediante seis estadísticos,
\begin{equation}
  \sigma_6(\Dgm{k}) = \bigl[\bar{p}_k,\;\operatorname{std}(p_k),\;\max(p_k),\;{\textstyle\sum} p_k,\;|\Dgm{k}|,\;P_{75}(p_k)\bigr],
  \qquad \X{2}=[\sigma_6(\Dgm{0}),\,\sigma_6(\Dgm{1})].
  \label{eq:x2block}
\end{equation}
La variante de control del descriptor fusionado, $\Xth^{(\mathrm{c})}\in\R^{22}$, concatena $\X{1}$ y $\X{2}$ divididas por su norma euclídea en cada red; la variante adaptativa $\Xth^{(\mathrm{a})}\in\R^{24}$ añade la ponderación por correlación con la exactitud y las dos predicciones Ridge descritas en el artículo.

\section{Propiedades matemáticas de los descriptores}
\label{sec:prop}

\begin{proposition}[Invariancia por permutación]
\label{prop:invariance}
Sea $\pi \in S_d$ una permutación cualquiera de las columnas de $A$ y sea $A^\pi$ la matriz permutada. Restringido a las mismas $m=\min(d,80)$ neuronas, $\X{1}(A^\pi_{\,\cdot,\pi(1..m)}) = \X{1}(A_{\,\cdot,1..m})$, de modo que reetiquetar las neuronas deja invariante el descriptor de grafo. Lo mismo vale para $\X{2}$.
\end{proposition}

\begin{proof}
La matriz de correlaciones de $A^\pi$ es $C^\pi = P^\top C P$, donde $P$ es la matriz de permutación asociada a $\pi$; la umbralización componente a componente conmuta con la conjugación por $P$, de modo que el grafo $G_\theta(A^\pi)$ es isomorfo a $G_\theta(A)$ mediante $\pi$. Las diez componentes de~\eqref{eq:x1} (densidad, agrupamiento medio, transitividad, momentos del grado, fracción de componentes conexas, razón aristas/nodos y momentos de la centralidad de intermediación) son invariantes de grafo, es decir, toman el mismo valor sobre grafos isomorfos, pues se definen mediante conteos y promedios sobre vértices y aristas sin referencia al etiquetado. Para $\X{2}$, la nube de puntos $\{a_{i\cdot}\}\subset\R^m$ es un conjunto no ordenado y la filtración de Vietoris--Rips depende solo de las distancias entre puntos, que no cambian al reordenar las coordenadas, pues una permutación de coordenadas es una isometría de $\R^m$ con la métrica euclídea.
\end{proof}

\begin{remark}[Alcance de la invariancia bajo truncamiento]
La invariancia exacta requiere $d\le 80$, debido a que las capas más anchas se truncan a sus primeras 80 columnas por índice, regla que depende del orden y restringe la garantía a la subpoblación seleccionada. Como el orden de neuronas empleado es fijo y reproducible, la selección de las primeras 80 columnas es idéntica para todas las redes, de modo que la comparación entre descriptores no se ve afectada.
\end{remark}

\begin{proposition}[Homología de grado cero mediante componentes conexas]
\label{prop:h0}
Sea $P=\{p_1,\dots,p_m\}$ una nube finita y sea $G_\epsilon = (P, \{\,\{p_i,p_j\} : \norm{p_i-p_j}_2 \le \epsilon\,\})$ su grafo de vecindad a escala $\epsilon$. Entonces $\beta_0\bigl(\mathcal{VR}(P,\epsilon)\bigr) = \beta_0(G_\epsilon)$; en particular, el número de clases de grado cero vivas a escala $\epsilon$ es igual al número de componentes conexas de $G_\epsilon$, y $\Dgm{0}$ queda determinado por los valores de $\epsilon$ en los que dos componentes se fusionan.
\end{proposition}

\begin{proof}
El grafo $G_\epsilon$ es exactamente el $1$-esqueleto de $\mathcal{VR}(P,\epsilon)$. La homología en dimensión cero de un complejo simplicial depende solo de su $1$-esqueleto, pues $H_0$ se define como el cociente $Z_0/B_0$ y los bordes $B_0$ están generados por las aristas; dos vértices son homólogos si y solo si están unidos por un camino de aristas. Por tanto, las clases de $H_0(\mathcal{VR}(P,\epsilon))$ están en biyección con las componentes conexas de $G_\epsilon$. Al crecer $\epsilon$, cada clase nace en $\epsilon=0$ (todo punto es una componente) y muere en el menor $\epsilon$ en el que su componente se fusiona con otra más antigua; esos valores de fusión son las muertes registradas en $\Dgm{0}$.
\end{proof}

Los diagramas completos son estables respecto de la distancia \emph{cuello de botella} $d_B(D,D') = \inf_{\gamma}\sup_{x\in D}\norm{x-\gamma(x)}_\infty$, con $\gamma$ recorriendo las biyecciones que emparejan puntos, incluso contra la diagonal.

\begin{theorem}[Estabilidad de los diagramas de persistencia; \citealt{cohensteiner2007stability}, forma para Vietoris--Rips en \citealt{chazal2014persistence}]
\label{thm:stability}
Sean $P, Q \subset \R^r$ nubes de puntos finitas y sea $d_H(P,Q)$ su distancia de Hausdorff. Entonces, para toda dimensión $k$,
\[
  d_B\bigl(\Dgm{k}(P),\, \Dgm{k}(Q)\bigr) \;\leq\; 2\, d_H(P, Q).
\]
\end{theorem}

\begin{corollary}
\label{cor:stability}
La persistencia máxima $\max_j p_j^k$ es estable, de modo que si $d_H(P,Q)\le\delta$, entonces $\bigl|\max_j p_j^k(P) - \max_j p_j^k(Q)\bigr| \le 4\delta$.
\end{corollary}

\begin{proof}
Sea $\gamma$ una biyección que realiza $d_B \le 2\delta$. El punto $x=(b,d)$ que alcanza la persistencia máxima en $\Dgm{k}(P)$ se empareja con $\gamma(x)=(b',d')$ en $\Dgm{k}(Q)$ (o con la diagonal) con $|b-b'|\le 2\delta$ y $|d-d'|\le 2\delta$, de donde la persistencia de $\gamma(x)$ difiere de la de $x$ en a lo sumo $4\delta$; como el máximo en $Q$ domina la persistencia de $\gamma(x)$, se sigue $\max p^k(Q) \ge \max p^k(P) - 4\delta$, y por simetría se obtiene la otra desigualdad.
\end{proof}

\begin{remark}[Estadísticos fuera de la garantía y variante de respaldo]
Dos estadísticos de~\eqref{eq:x2block}, el conteo $|\Dgm{k}|$ y la persistencia total, cambian abruptamente cuando un rasgo de vida corta aparece o desaparece, por lo que quedan fuera de esta garantía; su estabilidad se comprobó empíricamente con el repertorio replicado del artículo (AUC familiar de 0,929 con 20 arquitecturas reentrenadas con otras semillas). Como alternativa cuando Ripser no está disponible, una variante $\X{2}^*$ basada en la \cref{prop:h0} recorre 30 escalas equiespaciadas y recupera $\Dgm{0}$ hasta la resolución del muestreo de escalas, con los rasgos de grado uno fijados en cero. Todos los resultados principales del artículo usan el descriptor $\X{2}$ completo basado en Ripser; el barrido de la \cref{sec:barrido} usa la variante $\X{2}^*$ (véase la nota de esa sección).
\end{remark}

\section{Solapamiento entre las vistas}
\label{sec:solap}

\begin{proposition}[Solapamiento parcial entre las vistas (empírico)]
\label{prop:overlap}
En los $n{=}40$ modelos del repertorio de CIFAR-10, la correlación de Spearman entre pares de rasgos de~$\X{1}$ y~$\X{2}$ presenta una mediana de $|\rho|{=}0{,}235$, un rango intercuartílico de $[0{,}11;\,0{,}47]$ y un máximo de $0{,}829$ ($p{=}4{,}0{\times}10^{-11}$); tras la corrección de Benjamini--Hochberg ($q{\le}0{,}05$), 41 de los 110 pares evaluables muestran una asociación significativa.
\end{proposition}

Los 110 pares evaluables resultan de cruzar los diez rasgos de $\X{1}$ con los doce de $\X{2}$, descontados aquellos en que algún rasgo es constante (en CIFAR-10 y SVHN el número de clases de grado cero es constante, por lo que sus diez pares no se evalúan; en CIFAR-100 los 120 pares son evaluables). La \cref{tab:solap} extiende el resumen a los tres dominios y la \cref{fig:solap} muestra las tres matrices completas. Ambas vistas comparten información parcial, esperable al derivar de las mismas activaciones, aunque la mayoría de los pares presenta asociación débil ($|\rho|<0{,}3$ en más de la mitad de los pares en los tres dominios); si las vistas fueran redundantes, la fusión no superaría a la mejor de ellas, extremo que el artículo descarta en los tres dominios.

""" + TAB_SOLAP + r"""
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{fig_solapamiento.pdf}
\caption{Valor absoluto de la correlación de Spearman entre cada rasgo de $X_1$ (filas) y de $X_2$ (columnas) en los tres dominios. Los recuadros negros y las cifras en negrita marcan los pares significativos tras Benjamini--Hochberg ($q\le 0{,}05$); el guion marca los pares no evaluables. Datos: \texttt{outputs/redundancia\_vistas/matriz\_redundancia\_10x12.json}.}
\label{fig:solap}
\end{figure}

\section{Barrido de sensibilidad}
\label{sec:barrido}

El umbral $\theta{=}0{,}4$, el tipo de correlación, el tope de 80 neuronas, el número $N{=}50$ de ejemplos para la homología y el rango $r{=}15$ de la PCA se fijaron a priori. Para medir cuánto dependen los resultados de esas elecciones se recalcularon los descriptores de las 40 redes de CIFAR-10 variando un parámetro cada vez (\cref{tab:barrido}). El barrido se ejecutó con el guion \texttt{phase17\_experiments.py} sobre las activaciones re-extraídas de la última capa oculta (\texttt{outputs/fase16\_cifar10\_raw}), con la fusión de pesos iguales y con la variante de respaldo $\X{2}^*$ (grado cero por componentes conexas sobre la distancia de correlación, sin Ripser); por ello sus valores absolutos son inferiores al 0,949 del artículo y no deben compararse con él. Lo que interesa del barrido es la amplitud: el umbral $\theta$ mueve el AUC familiar en 0,030 dentro de $\{0{,}2;\ldots;0{,}6\}$, y el valor adoptado de $0{,}4$ no es el mejor del rango, lo que confirma que no se eligió para favorecer al método; el número de ejemplos y el rango de la PCA no alteran el resultado; el tope de neuronas es el parámetro más sensible (amplitud 0,059) y la capa analizada también influye, con mejor separación en la primera capa oculta.

""" + SWEEP + r"""

\section{Comprobaciones de robustez}
\label{sec:robustez}

El artículo resume cuatro comprobaciones adicionales sobre el repertorio de CIFAR-10; sus valores exactos, obtenidos con \texttt{phase15\_reviewer.py} (registro en \texttt{supplement/phase15\_robustez.log}), son los siguientes. Primera, estabilidad de la ponderación: la correlación de Spearman entre los pesos $w^{(v)}$ ajustados con toda la muestra y los ajustados dentro de cada pliegue fue, en promedio, de 0,970 en CIFAR-10, 0,956 en CIFAR-100 y 0,934 en SVHN. Segunda, aporte de las dos proyecciones Ridge: la variante adaptativa alcanzó un AUC familiar de 0,947 [0,91; 0,98] con ellas (24 dimensiones) y de 0,943 [0,90; 0,97] sin ellas (22 dimensiones), un aporte de $+0{,}004$. Tercera, AUC uno contra el resto por familia para $X_3^{(\mathrm{c})}$: A (ReLU) 0,944; B (BatchNorm y GELU) 0,890; C (residual) 0,963; D (CNN pequeña) 0,999; E (CNN residual) 1,000. Cuarta, línea base de metadatos enriquecida (número de parámetros, su logaritmo, profundidad y anchura): 0,854 [0,81; 0,89] frente a 0,949 [0,92; 0,97] de $X_3^{(\mathrm{c})}$, una ventaja de $+0{,}095$. Además, los pesos aprendidos en distintos dominios no se parecen entre sí (Spearman de $-0{,}04$ a $0{,}40$ entre pares de dominios), lo que refuerza la elección de la variante de control sin ponderación para los resultados principales.

\section{Figuras complementarias}
\label{sec:figuras}

Las cuatro figuras siguientes acompañaban a versiones anteriores del artículo y se remiten aquí por el límite de extensión de la revista; sus cifras aparecen en el texto del artículo.

\begin{figure}[H]\centering
\includegraphics[width=0.55\linewidth]{fig4_violins_es.pdf}
\caption{AUC familiar en los cinco pliegues de validación cruzada (CIFAR-10). Cada punto es un pliegue, los violines son densidades de núcleo y las barras marcan las medianas; el contraste de Wilcoxon unilateral sobre los cinco pliegues pareados arrojó $p{=}0{,}031$ para $X_3^{(c)}$ frente a $X_1$ (\textasteriskcentered) y $p{=}0{,}062$ frente a $X_2$ (n.\,s.).}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{fig_correccion_benjamini_hochberg.pdf}
\caption{Valores $p$ de la batería de diez pruebas por dominio frente al umbral escalonado de Benjamini--Hochberg ($q\,k/m$, $q{=}0{,}05$) y a la cota de Bonferroni ($\alpha/m{=}0{,}005$): 29 de las 30 pruebas superan la corrección y 28 resisten Bonferroni.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{forest_tamanos_efecto.pdf}
\caption{Tamaños de efecto por descriptor y dominio entre los cuartos extremos de exactitud: $d$ de Cohen con IC 95\,\% bootstrap (izquierda) y $|r|$ de Mann--Whitney (derecha); el relleno marca los contrastes significativos tras Benjamini--Hochberg. $X_3^{(r)}$ es la variante adaptativa con ponderación por rangos (Spearman al cuadrado), que solo interviene en la batería.}
\end{figure}

\section{Tablas completas por dominio}
\label{sec:tablas}

Para cada dominio se reportan los cuatro contextos de análisis del artículo (población completa, solo perceptrones multicapa, solo redes convolucionales y exactitud normalizada por tipo de red) y los cinco descriptores evaluados, incluida la variante adaptativa con ponderación por rangos $X_3^{(\mathrm{r})}$, que no se discute en el artículo. Todos los componentes que dependen de la exactitud se reajustaron dentro de cada partición de entrenamiento (sin fuga de información). Fuente: \texttt{outputs/fase11\_results}, \texttt{fase12\_results} y \texttt{fase13\_results/summary\_leakfree.json}. La marca n.\,d.\textsuperscript{*} señala un $d$ de Cohen no definido: en el subconjunto de diez redes convolucionales una de las familias extremas tiene varianza prácticamente nula en la exactitud, de modo que el cociente se dispara sin significado; en ese contexto se reportan solo el AUC y las pruebas de rangos.

""" + TABLAS + r"""

\section*{Referencias}
\begin{list}{}{\setlength{\leftmargin}{1.2em}\setlength{\itemindent}{-1.2em}}
\item Ballester, R., Arnal Clemente, X., Casacuberta, C., Madadi, M., Corneanu, C. A. y Escalera, S. (2024). Predicting the generalization gap in neural networks using topological data analysis. \emph{Neurocomputing}, 596, 127787.
\item Bauer, U. (2021). Ripser: efficient computation of Vietoris--Rips persistence barcodes. \emph{Journal of Applied and Computational Topology}, 5, 391--423.
\item Chazal, F., de Silva, V. y Oudot, S. (2014). Persistence stability for geometric complexes. \emph{Geometriae Dedicata}, 173, 193--214.
\item Cohen-Steiner, D., Edelsbrunner, H. y Harer, J. (2007). Stability of persistence diagrams. \emph{Discrete \& Computational Geometry}, 37, 103--120.
\item Edelsbrunner, H., Letscher, D. y Zomorodian, A. (2002). Topological persistence and simplification. \emph{Discrete \& Computational Geometry}, 28, 511--533.
\item Zhang, B., Dong, Z., Zhang, J. y Lin, H. (2023). Functional network: A novel framework for interpretability of deep neural networks. \emph{Neurocomputing}, 519, 94--103.
\end{list}
\end{document}
"""
TEX = TEX.replace("\\citealt{cohensteiner2007stability}", "Cohen-Steiner et al., 2007").replace("\\citealt{chazal2014persistence}", "Chazal et al., 2014")
(OUT / "suplemento.tex").write_text(TEX, encoding="utf-8")
print("tex ok", OUT / "suplemento.tex")
