# topo-spectral-geofluids

Multiview **topological–spectral descriptor** (functional graph ⊕ persistent
homology) ported from neural-network activations
([Dcrr1717/multiview-nn-descriptor](https://github.com/Dcrr1717/multiview-nn-descriptor))
to **geophysical-type fluid fields**, plus the Go/No-Go checkpoint for the
Springer *GEM — International Journal on Geomathematics* topical collection
*Computational Methods and Techniques Applicable to Geophysical Fluids*
(submission deadline 15 Dec 2026; checkpoint 1 Nov 2026).

* **Verdict and evidence:** [`REPORT.md`](REPORT.md)
* **Pre-registered protocol (frozen before any classification):** [`PROTOCOL.md`](PROTOCOL.md)

## What is in here

| Path | Content |
|---|---|
| `tsg/solvers.py` | 2D pseudo-spectral solvers (IF-RK4, 2/3 de-aliasing): **Kolmogorov flow** (periodic, `sin(4y)` forcing) and **Rayleigh–Bénard convection** (free-slip, sine series via odd extension) |
| `tsg/dynamics.py` | descriptor-free regime labelling: twin-trajectory Lyapunov exponent |
| `tsg/descriptors.py` | X1 (functional graph, ℝ¹⁰), X2 (sublevel-set persistent homology, ℝ¹²), X3 (fusion, ℝ²²) — same invariants / summaries / fusion as the network paper |
| `tsg/baselines.py` | B (energy-spectrum slope + simple statistics) and B_amp (amplitude confound proxy) |
| `tsg/protocol.py` | leak-free statistics: leave-parameter-out grouped CV, cluster bootstrap, paired ΔAUC, permutation, Wilcoxon, Kruskal–Wallis / Mann–Whitney, BH-FDR |
| `tsg/decision.py` | the pre-registered Go/No-Go rule |
| `tsg/config.py` | every simulation / descriptor parameter and the regime design |
| `scripts/00_pilot_regimes.py` | descriptor-free bifurcation scans → regime bands |
| `scripts/01_simulate.py` | data set generation (raw windows, git-ignored) |
| `scripts/02_features.py` | X1/X2/X3/B/B_amp per sample → `data/features_*.csv` (committed) |
| `scripts/03_classify.py` | the experiment + statistics → `results/<flow>_<mode>[_noise]/` |
| `scripts/04_sensitivity.py` | sensitivity sweep (θ, patches, correlation, persistence floor, Rips X2) |
| `scripts/05_verdict.py` | applies the decision rule → `results/checkpoint_verdict.json` |
| `scripts/06_figures.py` | regime-scan and "two views" figures |
| `tests/` | solver validation (Taylor–Green decay, laminar fixed point, energy budget, RB linear growth rate and sub-critical decay) and descriptor invariances |

## Mapping from the network descriptor

| neural network (paper) | fluid field (here) |
|---|---|
| hidden neuron (unit) | coarse patch of the mesh (8 × 8 = 64 nodes) |
| input example (sample) | pixel of the patch (snapshot mode) / time instant (window mode) |
| X1: correlation graph of neurons | coherence graph of mesh regions |
| X2: Vietoris–Rips PH of the activation point cloud | PH of the sublevel-set filtration of the field (cubical); Rips variant kept for sensitivity |
| X3 = [X1/‖X1‖, X2/‖X2‖] | identical |
| Family AUC (5 architecture families) | Regime AUC (laminar / transitional / turbulent) |
| trivial baseline: n_params, depth | B_amp: raw field amplitude (∝ Re/Ra) |
| leak-free CV over networks | leave-parameter-out grouped CV + trajectory-level bootstrap |

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest -q tests/
```

**Quick path** (≈ 1 h on 4 cores) — the committed `data/features_*.csv` are
enough to recompute every statistic, verdict and most figures:

```bash
python scripts/03_classify.py kolmogorov
python scripts/03_classify.py rb
python scripts/05_verdict.py
```

**Full path** (≈ 5 h on 4 CPU cores, no GPU) — regenerate the simulations:

```bash
python scripts/00_pilot_regimes.py kolmogorov          # descriptor-free regime scan
python scripts/00_pilot_regimes.py rb
python scripts/01_simulate.py kolmogorov               # data/raw/kolmogorov (~0.6 GB)
python scripts/01_simulate.py rb
for f in kolmogorov rb; do
  python scripts/02_features.py $f                      # primary: snapshot mode
  python scripts/02_features.py $f --mode window        # secondary: space-time window
  for s in 0.25 0.5 1.0; do python scripts/02_features.py $f --noise $s; done
done
for f in kolmogorov rb; do
  python scripts/03_classify.py $f
  python scripts/03_classify.py $f --mode window --perm 0
  for s in 0.25 0.5 1.0; do python scripts/03_classify.py $f --noise $s --perm 0 --no-extras; done
  python scripts/04_sensitivity.py $f
done
python scripts/05_verdict.py
python scripts/06_figures.py
```

All randomness is seeded from `SEED = 2024` (as in the network study).

## License

MIT (same as the parent project). © 2026 César Daniel Reinoso Reinoso and
Amalia Isabel Escudero Villa, ESPOCH.
