# Pre-registered protocol — Go/No-Go checkpoint

**Status: FROZEN** (amendments, all made before any descriptor was computed,
are listed in §13). This file, `tsg/decision.py` and `tsg/config.py` were
committed *before* any descriptor, baseline or classifier was computed on the
experimental data (see the git history: the protocol commit precedes the
commit that adds `data/features_*.csv` and `results/<flow>_*`). The only
data seen before freezing were the descriptor-free pilot bifurcation scans
in `results/pilot/` (global observables and Lyapunov exponents).

Any deviation from this protocol is reported in `REPORT.md` under
*Deviations*.

## 1. Question

Does the multiview descriptor **X3** (fusion of a functional-graph view X1 and a
persistent-homology view X2), ported from
[multiview-nn-descriptor](https://github.com/Dcrr1717/multiview-nn-descriptor),
separate the dynamical regimes of a geophysical-type flow (laminar /
transitional / turbulent) **better than each view alone and better than a
spectral-slope + simple-statistics baseline**, under a leak-free protocol?

## 2. Data

| | Kolmogorov flow (primary) | Rayleigh–Bénard convection (replication) |
|---|---|---|
| Equations | 2D NS, vorticity form, force `sin(4y) x̂`, `ν = 1/Re`, no drag | 2D Boussinesq, free-slip isothermal plates, free-fall units, `Pr = 1` |
| Domain / grid | `[0, 2π)²` periodic, 128² (all Re) | `Γ = 2`, periodic in x, 128 × 64 (all Ra) |
| Numerics | pseudo-spectral, 2/3 de-aliasing, IF-RK4, CFL 0.5 | same; sine series in z via odd extension |
| Observed field | vorticity ω | temperature anomaly θ |
| Control parameter | Re | Ra |
| Design (regime band → values) | laminar {5, 6, 7, 8}; transitional {12, 15, 18, 22}; turbulent {40, 55, 70, 100} | laminar {5·10⁴, 10⁵, 2·10⁵, 3·10⁵}; turbulent {4·10⁵, 5·10⁵, 7·10⁵, 10⁶} (§2.1) |
| Trajectories | 4 independent random ICs per value (48 total) | 4 per value |
| Sampling | spin-up 300, Lyapunov 100, then 12 windows of 16 frames (Δt = 0.5) separated by 12 time units | same (free-fall units) |

Regime bands were chosen from the pilot scans (`results/pilot/`): for
Kolmogorov flow the laminar state is stable up to Re ≈ 10, non-laminar
**steady** coherent states exist for 11 ≤ Re ≤ 25, Re = 30 is time-dependent
but not measurably chaotic, and the flow is chaotic (λ₁ > 0) for Re ≥ 40.
Values next to a bifurcation (Re = 10, 25–35) are excluded to keep labels
unambiguous.

### 2.1 Rayleigh–Bénard design

Fixed from `results/pilot/regime_scan_rb*.json` (tables in §11) before any RB
feature was computed. With free-slip plates, Pr = 1 and Γ = 2 the pilot shows
steady convection rolls up to Ra = 3.5·10⁵ and an abrupt transition to chaotic
shear-bursting convection (λ₁ > 0, Nu bursting between ≈ 1 and ≈ 14) from
Ra = 4·10⁵ on — the convectively driven shear of free-slip 2D convection
(Goluskin et al. 2014, *J. Fluid Mech.* 759). **No time-periodic window
exists on this branch**, so RB is a **two-regime** replication
(laminar = steady convection, turbulent = chaotic). Its primary endpoint is
the binary laminar-vs-turbulent AUC and it has no *hard* endpoint. A RB
trajectory whose dynamical label is *transitional* (if any) is outside the
design: it is excluded from the analysis and listed in `metrics.json`.

### 2.2 Labels (descriptor-free)

Each trajectory is labelled from its own dynamics (`tsg.config.label_trajectory`):
**turbulent** if the largest Lyapunov exponent (twin trajectory, 100 time units;
300 for RB) satisfies λ₁ − 2·SEM > 0 [*amended* to λ̂₁ > 0.01, see §13]; otherwise **laminar** if Kolmogorov: relative distance
to the laminar state < 10⁻³ / RB: CV of Nu over the sampling period < 10⁻³;
otherwise **transitional**. If a trajectory's dynamical label differs from its
band (multistability), the dynamical label is used and the count is reported.

## 3. Samples and descriptors

**Primary — snapshot mode.** A sample is one instantaneous field (frame 0 of each
window): 12 samples per trajectory. Everything is computed on the field
standardised to zero mean / unit variance, so no view can read the absolute
amplitude (a trivial proxy of Re/Ra).

* **X1 ∈ ℝ¹⁰** — functional graph: the field is cut into 8 × 8 = 64 patches
  (nodes, analogue of the 80-neuron cap); each patch's pixels in local
  coordinates are the samples; edge if |Pearson r| > θ = 0.40; the ten
  invariants of the original `desc_x1` (density, average clustering,
  transitivity, degree mean/std/max-fraction, components fraction,
  edges per node, betweenness mean/std).
* **X2 ∈ ℝ¹²** — persistent homology of the sublevel-set filtration (cubical
  complex; periodic in x and y for Kolmogorov, in x only for RB); finite H0 and
  H1 pairs with persistence ≥ 0.01σ; the original `sig6` summary (mean, std,
  max, sum, count, 75th percentile of persistence) for H0 and H1.
* **X3 ∈ ℝ²²** — the original label-free fusion `desc_x3`:
  `[X1/‖X1‖, X2/‖X2‖]`.

**Secondary — window mode** (not used for the decision). A sample is the whole
16-frame window: X1 = graph of regions co-varying in time (climate network),
X2 = sublevel PH averaged over 4 frames, B adds two temporal statistics.

## 4. Baselines

* **B** (snapshot ℝ⁷): log–log slope of the energy spectrum (Kolmogorov:
  isotropic kinetic-energy spectrum from ω, fit over 5 ≤ k ≤ 30; RB: horizontal
  θ-variance spectrum, 2 ≤ kₓ-index ≤ 30), spectral centroid, skewness,
  kurtosis, peak/rms, ⟨|∇q|²⟩/⟨q²⟩, gradient anisotropy — on the same
  standardised field.
* **B_amp** (ℝ²): log₁₀ variance and log₁₀ spectral energy of the **raw**
  field. Confound control only (the analogue of the `n_params` baseline).

## 5. Classifier and cross-validation

Identical to the network study: `StandardScaler` + one-vs-rest
`LogisticRegression(C=1, max_iter=500)`; weighted one-vs-rest ROC AUC
("Regime AUC", analogue of the Family AUC).
CV: `StratifiedGroupKFold`, **groups = control-parameter value**
(leave-parameter-out: test folds contain only Re/Ra values never seen in
training), k = 4, 60 repetitions (mean and 2.5–97.5 % interval, as in the
paper). Out-of-fold probabilities are averaged over 20 partitions (still
leak-free) and used for the bootstrap and ROC curves.

## 6. Endpoints

* **Primary**: Regime AUC over the design regimes (Kolmogorov: 3-class; RB:
  binary laminar vs turbulent).
* **Hard** (Kolmogorov only): binary AUC, transitional vs turbulent.

## 7. Inference

* AUC 95 % CI: stratified **cluster bootstrap over trajectories** (B = 2000).
* Paired ΔAUC with the same bootstrap for the contrast family
  {X3−X1, X3−X2, X3−B, X3−B_amp, (B+X3)−B}; two-sided p; **Benjamini–Hochberg**
  within each endpoint (5 contrasts).
* Permutation test vs chance: labels permuted at the control-parameter level
  (200 permutations).
* Wilcoxon signed-rank on the 60 repeated-CV AUCs (descriptive, as Fig. 4 of
  the paper; repeats are not independent).
* Feature level: Kruskal–Wallis (ε²) across regimes and Mann–Whitney
  (rank-biserial r) transitional vs turbulent on **per-trajectory means**,
  BH-FDR.
* View redundancy: Spearman |ρ| for every X1 × X2 pair (per-trajectory means),
  BH-FDR — as in the supplement of the paper.

## 8. Confound controls

1. Amplitude removed from all views (standardised field); **B_amp** quantifies
   how much a trivial Re/Ra proxy achieves; X3−B_amp and B_amp+X3 reported.
2. **Leakage control**: grouped leave-parameter-out CV + trajectory-level
   bootstrap; the naive window-level CV is reported only to show the inflation.
3. **Incremental value**: (B+X3) vs B.
4. Same grid, same observable and same sampling for every parameter value.
5. Second classifier (random forest) — robustness only.

## 9. Decision rule (implemented in `tsg/decision.py`)

**Decision endpoint (per flow)** — the first endpoint, in this order, on which
the baseline B is *not* at ceiling (B OOF-AUC < 0.98):
clean-primary → clean-hard → noisy-primary → noisy-hard (RB: clean-primary →
noisy-primary), where *noisy* means i.i.d. Gaussian observation noise with
std = 0.5 × field std (SNR ≈ 6 dB), in snapshot mode. If all are at ceiling the
flow is non-informative.

Criteria on the decision endpoint:

* **G1 — beats the baseline**: ΔAUC(X3 − B) ≥ +0.02, bootstrap CI lower bound > 0,
  BH-adjusted p < 0.05.
* **G2 — fusion beats each view**: ΔAUC(X3 − X1) and ΔAUC(X3 − X2) both with CI
  lower bound > 0 and BH-adjusted p < 0.05.
* **R — replication (RB)**: ΔAUC(X3 − B) > 0 and X3 ≥ max(X1, X2) − 0.01 on the RB
  decision endpoint.
* **G4 — amplitude confound** (reported, not gating): ΔAUC(X3 − B_amp) CI > 0.

**Verdict**

* **GO**: G1 and G2 on Kolmogorov flow, and R on RB.
* **PLAN B (conditional)**: G1 on Kolmogorov flow, but G2 fails or the RB
  replication is *not replicated* or *non-informative* (every RB endpoint at
  ceiling).
* **PLAN C (NO-GO)**: G1 fails on Kolmogorov flow (including the case where
  every endpoint is at ceiling).

## 10. Secondary / exploratory (never change the verdict)

Window mode; noise curve (σ = 0.25, 0.5, 1.0); random forest; sensitivity sweep
(θ ∈ {0.2, …, 0.6}, patches 4×4 / 8×8 / 16×16, Spearman vs Pearson,
persistence floor, Vietoris–Rips point-cloud X2 of the original paper).

### Plan B and Plan C (what changes if the verdict is not GO)

* **Plan B** — the paper is reframed around what does work: e.g. the single
  dominant view (sublevel-set PH or region-coherence graph) vs the baseline,
  with fusion as an ablation; or a narrower single-flow contribution with the
  replication listed as a limitation; or a learned (leak-free) fusion weight.
* **Plan C** — do not submit the regime-classification claim. Pivot options:
  (i) noisy / sparse observations, where spectral slopes degrade and PH has
  stability guarantees; (ii) early-warning of transition (regression on Re or
  λ₁ instead of classification); (iii) a methods/negative-result note.

## 11. Appendix — pilot tables

One trajectory per value (seeded random IC), spin-up 300, statistics over the
last 100 time units, λ₁ from a 100-unit twin trajectory (block SEM). Source:
`results/pilot/regime_scan_*.json`.

**Kolmogorov flow (n = 4, 128²)**

| Re | D/D_lam (last 100) | CV of D | dist. to laminar | λ₁ ± SEM |
|---|---|---|---|---|
| 6 | 1.000 | 1.4e-16 | 1.1e-07 | -0.064 ± 0.049 |
| 8 | 1.000 | 1.6e-15 | 2.9e-08 | -0.064 ± 0.037 |
| 10 | 0.985 | 7.5e-04 | 1.2e-01 | -0.029 ± 0.028 |
| 11 | 0.766 | 1.4e-06 | 4.8e-01 | -0.030 ± 0.029 |
| 12 | 0.611 | 4.0e-09 | 6.2e-01 | -0.029 ± 0.029 |
| 13 | 0.501 | 1.7e-10 | 7.1e-01 | -0.029 ± 0.029 |
| 14 | 0.420 | 4.8e-06 | 7.6e-01 | -0.029 ± 0.028 |
| 16 | 0.343 | 5.4e-05 | 8.1e-01 | -0.023 ± 0.025 |
| 18 | 0.280 | 1.6e-09 | 8.5e-01 | -0.024 ± 0.028 |
| 20 | 0.233 | 5.9e-10 | 8.8e-01 | -0.026 ± 0.025 |
| 22 | 0.198 | 8.0e-09 | 9.0e-01 | -0.026 ± 0.026 |
| 25 | 0.159 | 2.5e-05 | 9.2e-01 | -0.035 ± 0.027 |
| 30 | 0.121 | 6.4e-02 | 9.3e-01 | -0.016 ± 0.026 |
| 40 | 0.104 | 3.1e-01 | 9.3e-01 | +0.111 ± 0.015 |
| 60 | 0.059 | 3.9e-01 | 9.6e-01 | +0.149 ± 0.027 |
| 100 | 0.027 | 3.1e-01 | 9.9e-01 | +0.291 ± 0.040 |

**Rayleigh–Bénard (free-slip, Pr = 1, Γ = 2, 128 × 64)**

| Ra | Nu (last 100) | CV of Nu | λ₁ ± SEM |
|---|---|---|---|
| 2e+03 | 2.53 | 0.0e+00 | -0.039 ± 0.039 |
| 5e+03 | 3.80 | 1.3e-16 | -0.033 ± 0.034 |
| 1e+04 | 4.99 | 1.6e-16 | -0.030 ± 0.030 |
| 2e+04 | 6.50 | 1.5e-16 | -0.026 ± 0.027 |
| 5e+04 | 9.14 | 4.9e-12 | -0.018 ± 0.021 |
| 1e+05 | 11.74 | 4.9e-09 | -0.019 ± 0.025 |
| 2e+05 | 15.01 | 8.1e-07 | -0.021 ± 0.031 |
| 2.5e+05 | 16.24 | 2.4e-06 | -0.014 ± 0.022 |
| 3e+05 | 17.30 | 5.8e-06 | -0.013 ± 0.021 |
| 3.5e+05 | 18.25 | 1.3e-05 | -0.010 ± 0.017 |
| 4e+05 | 3.62 | 9.1e-01 | +0.065 ± 0.044 |
| 5e+05 | 5.28 | 7.1e-01 | +0.106 ± 0.035 |
| 7e+05 | 3.99 | 9.0e-01 | +0.116 ± 0.048 |
| 1e+06 | 1.94 | 6.6e-01 | +0.122 ± 0.042 |

## 12. Design change made before freezing (descriptor-free)

The first draft used the 16-frame space-time window as the primary sample.
The Kolmogorov pilot showed that every transitional state in the design band
(11 ≤ Re ≤ 25) is **steady**, so any temporal statistic (and the time-correlation
graph) would separate transitional from turbulent trivially. The primary
analysis was therefore moved to **single snapshots** with purely spatial views
and a purely spatial baseline; the window analysis is kept as secondary. This
change used only the pilot dynamics, never descriptor or classifier output.


## 13. Amendments after freezing (all before any descriptor was computed)

**Amendment 1 — RB Lyapunov window 100 → 300 time units** (data generation of
both flows restarted at commit 40834b6 after an integrator fix; this amendment
restarts RB only). The RB turbulent state is intermittent (long quiescent shear
phases between bursts), so the block SEM of a 100-unit twin-trajectory estimate
is too large to resolve λ₁ > 0: the pilot gives λ₁ = 0.065 ± 0.044 at Ra = 4·10⁵,
and the first RB data trajectory (Ra = 10⁶) gave 0.067 ± 0.034. Under the
frozen rule (λ₁ − 2·SEM > 0) these chaotic trajectories would be labelled
*transitional* and excluded. The criterion is unchanged; only the estimation
window grows, which shrinks the SEM by ≈ √3. Only dynamical diagnostics of one
trajectory were seen; no descriptor, baseline or classifier output existed.

**Amendment 2 — chaos criterion λ₁ − 2·SEM > 0 → λ̂₁ > 0.01 (both flows).**
Even with the 300-unit window, the first three RB trajectories at Ra = 10⁶ gave
λ̂₁ = 0.056 ± 0.023, 0.057 ± 0.035 and 0.048 ± 0.030: all clearly positive, with
irregular Nu bursts (CV ≈ 1), yet two of three fail the 2-SEM test. The block
SEM is not a valid uncertainty for intermittent chaos (block means are
heavy-tailed: blocks with and without a burst), so the test is replaced by a
point-estimate threshold placed in the gap observed in the pilots: every
non-chaotic pilot state (steady, laminar, time-dependent Re = 30) has
λ̂₁ ≤ −0.010 and every chaotic one λ̂₁ ≥ +0.048. Labels are recomputed from the
stored trajectory metadata (no re-simulation). The frozen-rule label is kept
in the feature files (`regime_frozen_rule`) and the number of trajectories
whose label differs is reported. Again, no descriptor, baseline or
classifier output existed when this amendment was made.
