# Descriptor multivista de activaciones neuronales

Reference implementation and reproducibility code for the paper

> **Descriptor multivista de activaciones neuronales, mediante la fusión de grafos funcionales y homología persistente**
> (*Multiview descriptor of neural activations, through the fusion of functional graphs and persistent homology*)
> César Daniel Reinoso Reinoso, Amalia Isabel Escudero Villa
> Escuela Superior Politécnica de Chimborazo (ESPOCH), Riobamba, Ecuador
> *Revista Bases de la Ciencia* (2026) — DOI: *pending*

## What this is

We characterise a **trained** neural network by two complementary, quantitative
views of its hidden-neuron activations and fuse them into a single descriptor
**X₃** (22-dimensional in the accuracy-free control variant used for the main
results, 24-dimensional in the adaptive variant):

- **X₁ — functional graph** (ℝ¹⁰): ten classical graph-theoretic invariants of
  the thresholded correlation graph of the neuron activations.
- **X₂ — persistent homology** (ℝ¹²): twelve summary statistics of the
  Vietoris–Rips persistence of the neuron point cloud.
- **X₃ = φ(X₁, X₂)**: the fusion, with an accuracy-free equal-weight variant
  `X3^(c)` (ℝ²², main results) and an accuracy-weighted adaptive variant
  `X3^(a)` (ℝ²⁴, adds Pearson² weights and two Ridge projections).

Across a zoo of **40 networks** (30 MLP + 10 CNN, 5 families) trained on
CIFAR-10, CIFAR-100 and SVHN, X₃ discriminates architectural families with a
weighted one-vs-rest **Family AUC of 0.949 / 0.957 / 0.972**, under a
leakage-free cross-validation protocol. See the paper for the full analysis.

## Supplementary material

**[`SUPLEMENTO.pdf`](SUPLEMENTO.pdf)** (Spanish) contains the material the paper
defers to this repository because of the journal's page limit:

1. statements **and proofs** of the mathematical properties of the descriptors
   (permutation invariance of X₁/X₂, degree-0 homology via connected
   components, bottleneck stability and the stability of the maximum
   persistence);
2. the **overlap study** between the two views (Spearman correlation of every
   X₁×X₂ feature pair, per domain, with Benjamini–Hochberg control; data in
   `outputs/redundancia_vistas/`);
3. the **sensitivity sweep** over θ, correlation type, neuron cap, number of
   examples, PCA rank and layer (log in
   `outputs/fase16_cifar10_raw/run_phase17_v2.log`, produced by
   `phase17_experiments.py`);
4. the **robustness checks** (fold-weight stability, Ridge-projection contribution,
   per-family AUC, enriched metadata baseline; log in `supplement/phase15_robustez.log`)
   and the four **complementary figures** (fold violins, Benjamini–Hochberg,
   effect-size forest, overlap heat map);
5. the **complete result tables per domain and per analysis context**
   (full population, MLP-only, CNN-only, type-normalised accuracy; source
   JSONs in `outputs/fase1{1,2,3}_results/summary_leakfree.json`).

The PDF is regenerated from those files with `supplement/build_suplemento.py`
followed by `pdflatex supplement/suplemento.tex`.

## Repository layout

```
.
├── phase16_reextract.py     # builds the 5-family zoo (n=40): trains + extracts activations
├── phase11_leakfree.py      # leak-free X3 analysis protocol (the core methodology)
├── phase12_cifar100.py      # multidomain replication (CIFAR-100)
├── phase13_svhn.py          # multidomain replication (SVHN)
├── phase14_baselines.py     # trivial / metadata baselines
├── phase17_experiments.py   # sensitivity sweep, CKA baseline, neural persistence
├── figures/                 # figure-generation scripts (fig2, fig3, fig4, fig5, fig6)
├── SUPLEMENTO.pdf           # supplementary material (proofs, overlap study, sensitivity sweep, full tables)
├── supplement/              # LaTeX source + builder script of SUPLEMENTO.pdf
├── data/
│   ├── descriptors_cifar10.csv   # per-model X1/X2/X3 features + family + accuracy (n=40)
│   └── model_configs.csv         # per-model architecture config, params, accuracy
└── outputs/                 # precomputed model records (see "Reproducing" below)
    ├── fase10_results/all_records_f10.pkl   # descriptor records (used by the analysis + figures)
    ├── fase16_cifar10_raw/all_records_f16.pkl  # raw activations (used by the sensitivity sweep)
    ├── fase1{1,2,3}_results/summary_leakfree.json  # per-domain result summaries (CIFAR-10/100, SVHN)
    └── redundancia_vistas/  # X1×X2 overlap study (script + JSON matrices)
```

## Reproducing the results

**Quick path — reproduce every statistic and figure without retraining.**
The precomputed model records under `outputs/` (and the CSVs under `data/`) are
enough to recompute the descriptors, the Family AUC, the significance tests and
all figures:

```bash
pip install -r requirements.txt

# core leak-free analysis (Family AUC, tests, BH-FDR) on the CIFAR-10 zoo
python phase11_leakfree.py

# baselines, sensitivity sweep, CKA and neural-persistence comparisons
python phase14_baselines.py
python phase17_experiments.py

# figures (written next to each script inside figures/)
cd figures && python fig4_violins.py && python fig6_sensitivity.py
```

**Full path — rebuild the zoo from scratch.**
`phase16_reextract.py` trains the 40 networks and extracts their activations;
`phase12_cifar100.py` and `phase13_svhn.py` replicate on the other two domains.
These download CIFAR-10 / CIFAR-100 / SVHN automatically through `torchvision`
and require a GPU for reasonable runtimes. All runs use random seed **2024**.

> The `.pkl` records are plain NumPy/Python objects. If you prefer not to
> `pickle.load` third-party files, the human-readable `data/*.csv` exports carry
> the descriptors and configurations needed to reproduce every statistical result.

## Requirements

See `requirements.txt`. Core stack: PyTorch + torchvision, scikit-learn, SciPy,
statsmodels, NetworkX, NumPy, Matplotlib.

## Citation

```bibtex
@article{reinoso2026multiview,
  author  = {Reinoso Reinoso, C{\'e}sar Daniel and Escudero Villa, Amalia Isabel},
  title   = {Descriptor multivista de activaciones neuronales, mediante la
             fusi{'o}n de grafos funcionales y homolog{'i}a persistente},
  journal = {Revista Bases de la Ciencia},
  year    = {2026},
  note    = {DOI pending}
}
```

## License

Released under the [MIT License](LICENSE). © 2026 César Daniel Reinoso Reinoso
([ORCID 0000-0002-1233-517X](https://orcid.org/0000-0002-1233-517X)) and
Amalia Isabel Escudero Villa
([ORCID 0000-0003-4506-932X](https://orcid.org/0000-0003-4506-932X)), ESPOCH.
