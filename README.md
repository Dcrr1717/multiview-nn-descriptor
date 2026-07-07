# A Multiview Descriptor for Neural Architecture Discrimination

Reference implementation and reproducibility code for the paper

> **A Multiview Descriptor for Neural Architecture Discrimination via Functional Graphs and Persistent Homology**
> César Daniel Reinoso Reinoso, Amalia Isabel Escudero Villa
> Escuela Superior Politécnica de Chimborazo (ESPOCH), Riobamba, Ecuador
> *Revista Bases de la Ciencia* (2026) — DOI: *pending*

## What this is

We characterise a **trained** neural network by two complementary, quantitative
views of its hidden-neuron activations and fuse them into a single 24-dimensional
descriptor **X₃**:

- **X₁ — functional graph** (ℝ¹⁰): ten classical graph-theoretic invariants of
  the thresholded correlation graph of the neuron activations.
- **X₂ — persistent homology** (ℝ¹²): twelve summary statistics of the
  Vietoris–Rips persistence of the neuron point cloud.
- **X₃ = φ(X₁, X₂)** (ℝ²⁴): the fusion, with an accuracy-free equal-weight
  variant `X3^(c)` and an accuracy-weighted adaptive variant `X3^(a)`.

Across a zoo of **40 networks** (30 MLP + 10 CNN, 5 families) trained on
CIFAR-10, CIFAR-100 and SVHN, X₃ discriminates architectural families with a
weighted one-vs-rest **Family AUC of 0.949 / 0.957 / 0.972**, under a
leakage-free cross-validation protocol. See the paper for the full analysis.

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
├── data/
│   ├── descriptors_cifar10.csv   # per-model X1/X2/X3 features + family + accuracy (n=40)
│   └── model_configs.csv         # per-model architecture config, params, accuracy
└── outputs/                 # precomputed model records (see "Reproducing" below)
    ├── fase10_results/all_records_f10.pkl   # descriptor records (used by the analysis + figures)
    └── fase16_cifar10_raw/all_records_f16.pkl  # raw activations (used by the sensitivity sweep)
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
  title   = {A Multiview Descriptor for Neural Architecture Discrimination
             via Functional Graphs and Persistent Homology},
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
