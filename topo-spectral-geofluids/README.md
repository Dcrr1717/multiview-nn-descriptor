# topo-spectral-geofluids

Multiview **topological–spectral descriptor** (functional graph ⊕ persistent
homology) ported from neural-network activations
([Dcrr1717/multiview-nn-descriptor](https://github.com/Dcrr1717/multiview-nn-descriptor))
to **geophysical-type fluid fields**, plus the Go/No-Go checkpoint for the
Springer *GEM — International Journal on Geomathematics* topical collection
*Computational Methods and Techniques Applicable to Geophysical Fluids*
(submission deadline 15 Dec 2026; checkpoint 1 Nov 2026).

> **Checkpoint verdict (30 Oct 2026): PLAN C (NO-GO).** X3 is a valid regime
> signature (AUC 0.95–0.99), but on the pre-registered decision endpoint it
> beats neither the spectral-slope + statistics baseline (ΔAUC +0.012,
> CI [−0.012, +0.053]) nor its best single view. Evidence, diagnosis and
> Plan B/C: [`REPORT.md`](REPORT.md). An exploratory follow-up (§7) found no
> advantage for the descriptor with sparse station data or cross-flow
> transfer either. The recommended deliverable is now a methods /
> negative-result note.

* **Verdict and evidence:** [`REPORT.md`](REPORT.md)
* **Pre-registered protocol (frozen before any classification):** [`PROTOCOL.md`](PROTOCOL.md)
* **Machine-readable verdict:** [`results/checkpoint_verdict.json`](results/checkpoint_verdict.json);
  all tables: [`results/summary_tables.md`](results/summary_tables.md)

This folder is self-contained and meant to become its own repository
(`topo-spectral-geofluids`). To extract it with its history:
`git subtree split --prefix topo-spectral-geofluids -b topo-spectral-geofluids`,
then push that branch to the new repository. Commit hashes cited in
`PROTOCOL.md` / `REPORT.md` (e.g. the protocol freeze `b6718d4`) refer to the
history in `Dcrr1717/multiview-nn-descriptor`. After the split the same
commits keep their messages and order but get new hashes.

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
| `tsg/stations.py`, `scripts/07_stations.py` | Plan C1 feasibility (exploratory): station networks, lower-star PH on the Delaunay complex, semivariogram baseline |
| `scripts/08_transfer.py` | Plan C2 feasibility (exploratory): cross-flow transfer |
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

(The noise, window and label-sensitivity variants use the flags shown below;
every run writes `results/<flow>_<mode>[_noise<s>][_label_<rule>]/metrics.json`
plus its figures.)

**Full path** (≈ 6–7 h on 4 CPU cores, no GPU) — regenerate the simulations:

```bash
python scripts/00_pilot_regimes.py kolmogorov          # descriptor-free regime scan
python scripts/00_pilot_regimes.py rb
python scripts/01_simulate.py kolmogorov               # data/raw/kolmogorov (~0.6 GB)
python scripts/01_simulate.py rb                       # data/raw/rb (~0.2 GB)
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
  python scripts/03_classify.py $f --label regime_steady_first --perm 0 --no-extras   # label sensitivity
  python scripts/03_classify.py $f --label regime_frozen_rule --perm 0 --no-extras
done
python scripts/03_classify.py kolmogorov --noise 0.5 --label regime_frozen_rule --perm 0 --no-extras
python scripts/05_verdict.py
python scripts/06_figures.py
# exploratory follow-up (REPORT.md section 7)
for f in kolmogorov rb; do for M in 64 128 256; do python scripts/07_stations.py $f --M $M --quick; done; done
python scripts/08_transfer.py
```

All randomness is seeded from `SEED = 2024` (as in the network study).

## License

MIT (same as the parent project). © 2026 César Daniel Reinoso Reinoso and
Amalia Isabel Escudero Villa, ESPOCH.
