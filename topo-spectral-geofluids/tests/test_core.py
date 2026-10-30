"""Fast correctness checks: python -m pytest -q tests/"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsg.baselines import baseline_B, spectral_slope, spectrum  # noqa: E402
from tsg.descriptors import (desc_x3, graph_invariants, normalize_window,  # noqa: E402
                             sublevel_persistence, x1_functional_graph, x2_sublevel_ph)
from tsg.solvers import KolmogorovFlow, RayleighBenard  # noqa: E402


def test_taylor_green_decay():
    """Taylor-Green vortex is an exact NS solution decaying as exp(-2 nu t)."""
    s = KolmogorovFlow(Re=50, N=32)
    s.F_hat[:] = 0
    w0 = 2 * np.cos(s.X) * np.cos(s.Y)
    w = s.ifft(s.advance(s.fft(w0), 2.0))
    assert np.abs(w - w0 * np.exp(-2 * s.nu * 2.0)).max() < 1e-10


def test_kolmogorov_laminar_fixed_point():
    s = KolmogorovFlow(Re=5, N=32)
    wl = s.laminar()
    w = s.ifft(s.advance(s.fft(wl), 5.0))
    assert np.abs(w - wl).max() / np.abs(wl).max() < 1e-5


def test_kolmogorov_energy_budget():
    """dE/dt = I - D (power input minus viscous dissipation)."""
    s = KolmogorovFlow(Re=30, N=64)
    wh = s.random_ic(np.random.default_rng(0), amp=1.0, add_laminar=True)
    dt = 1e-3
    d0 = s.diagnostics(wh)
    wh1 = s.advance(wh, dt)
    d1 = s.diagnostics(wh1)
    dEdt = (d1["E"] - d0["E"]) / dt
    rate = 0.5 * (d0["I"] - d0["D"] + d1["I"] - d1["D"])
    assert abs(dEdt - rate) < 1e-3 * max(1.0, abs(rate))


def test_rb_linear_growth_rate():
    """Free-slip onset mode (kx = pi, m = 1): growth rate vs linear theory."""
    Ra, Pr = 2000.0, 1.0
    r = RayleighBenard(Ra, Pr, Gamma=2.0, Nx=32, Nz=16)
    th = np.sin(np.pi * r.Z) * np.cos(np.pi * r.X) * 1e-8
    st = (np.zeros_like(r.fft(th)), r.odd(r.fft(th)))
    st = r.advance(st, 8.0)
    a0 = np.abs(r.ifft(st[1])).max()
    st = r.advance(st, 4.0)
    a1 = np.abs(r.ifft(st[1])).max()
    k2, a2 = np.pi ** 2, 2 * np.pi ** 2
    b, c = (Pr + 1) * a2, Pr * a2 * a2 - Ra * Pr * k2 / a2
    theory = (-b + np.sqrt(b * b - 4 * c)) / 2 / np.sqrt(Ra * Pr)
    assert abs(np.log(a1 / a0) / 4.0 - theory) / theory < 0.01


def test_rb_subcritical_decay():
    r = RayleighBenard(600.0, 1.0, Gamma=2.0, Nx=32, Nz=16)   # Ra_c(k=pi) = 8 pi^4 ~ 779
    st = r.random_ic(np.random.default_rng(1), amp=1e-3)
    a0 = np.abs(r.ifft(st[1])).max()
    st = r.advance(st, 20.0)
    assert np.abs(r.ifft(st[1])).max() < a0


def test_graph_invariants_permutation_invariant():
    """X1 is invariant to relabelling the regions (Proposition in the paper's supplement)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 30))
    A[:, 1] = A[:, 0] + 0.1 * rng.standard_normal(40)
    p = rng.permutation(30)
    assert np.allclose(graph_invariants(A), graph_invariants(A[:, p]))


def test_x1_steady_window_is_empty_graph():
    w = np.repeat(np.cos(4 * np.linspace(0, 2 * np.pi, 64, endpoint=False))[None, :, None], 16, 0)
    w = np.repeat(w, 64, axis=2)
    x1 = x1_functional_graph(normalize_window(w), blocks=(8, 8))
    assert x1[0] == 0 and x1[7] == 0


def test_sublevel_ph_counts_extrema():
    """Two Gaussian wells on a torus -> one finite H0 pair: the shallower well
    (depth 0.5) dies at the saddle between the wells (value ~ -0.066)."""
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    X, Y = np.meshgrid(x, x)
    f = -np.exp(-((X - 2) ** 2 + (Y - 2) ** 2)) - 0.5 * np.exp(-((X - 4.5) ** 2 + (Y - 4.5) ** 2))
    h0, _ = sublevel_persistence(f, (True, True), min_pers=0.01)
    assert len(h0) == 1 and 0.40 < h0[0] < 0.47


def test_x2_and_x3_shapes():
    rng = np.random.default_rng(0)
    w = normalize_window(rng.standard_normal((8, 32, 32)))
    x2 = x2_sublevel_ph(w, n_snap=2)
    x1 = x1_functional_graph(w, blocks=(4, 4))
    x3 = desc_x3(x1, x2)
    assert x2.shape == (12,) and x3.shape == (22,)
    assert np.isclose(np.linalg.norm(x3[:10]), 1) and np.isclose(np.linalg.norm(x3[10:]), 1)


def test_spectral_slope_power_law():
    """A field with E(k) ~ k^-3 must give slope ~ -3."""
    N = 128
    rng = np.random.default_rng(0)
    k = np.fft.fftfreq(N, 1 / N)
    KY, KX = np.meshgrid(k, np.fft.rfftfreq(N, 1 / N), indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    # E(k) ~ k^-3  <=>  |w_k|^2 ~ k^2 E(k) / (2 pi k) ~ k^-2  <=>  |w_k| ~ k^-1
    amp = np.where(K > 0, 1.0 / np.where(K > 0, K, 1), 0) * np.exp(2j * np.pi * rng.random(K.shape))
    w = np.fft.irfft2(amp, s=(N, N))
    kk, S = spectrum(w, "periodic_vorticity")
    assert abs(spectral_slope(kk, S, (5, 30)) + 3) < 0.3
    assert baseline_B(normalize_window(np.stack([w, w])), "periodic_vorticity", (5, 30)).shape == (9,)


def test_patch_matrix_columns_are_patches():
    from tsg.descriptors import patch_matrix
    f = np.arange(16 * 32, dtype=float).reshape(16, 32)
    P = patch_matrix(f, blocks=(2, 4))          # patches of 8 x 8 pixels
    assert P.shape == (64, 8)
    assert np.array_equal(P[:, 1], f[0:8, 8:16].ravel())     # row-block 0, col-block 1
    assert np.array_equal(P[:, 4], f[8:16, 0:8].ravel())     # row-block 1, col-block 0


def test_patch_coherence_orders_laminar_above_noise():
    from tsg.descriptors import x1_patch_coherence
    x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    lam = np.repeat(np.cos(4 * x)[:, None], 128, axis=1) + 0.01 * np.cos(x)[None, :]
    noise = np.random.default_rng(0).standard_normal((128, 128))
    d_lam = x1_patch_coherence(normalize_window(lam))[0]
    d_noise = x1_patch_coherence(normalize_window(noise))[0]
    assert d_lam > 0.5 and d_noise < 0.05
