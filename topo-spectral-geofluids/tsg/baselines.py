"""
Baselines against which X3 is tested.

B  - "spectral slope + simple statistics", computed on the SAME
          amplitude-normalised field the descriptors see. Snapshot mode
          (primary) uses the 7 spatial features, window mode (secondary)
          adds the 2 temporal ones:
            spec_slope   log-log slope of the energy (variance) spectrum
            spec_centroid  energy-weighted mean wavenumber
            skew, kurt     spatial skewness / kurtosis of the field
            peak_rms       max|q| / rms(q)
            grad_ratio     <|grad q|^2> / <q^2>  (inverse Taylor-like scale^2)
            anisotropy     <q_x^2> / (<q_x^2> + <q_y^2>)
            var_cv_t       temporal CV of the snapshot variance in the window
            lag1_corr_t    mean correlation between consecutive snapshots
          Snapshot statistics are averaged over the window.

B_amp (R^2) - trivial control-parameter proxy computed on the RAW field:
          log10 of the mean field variance and log10 of the mean spectral
          energy. Used only as a confound control (analogue of the
          n_params / depth baseline of the neural-network study).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

B_SNAP_NAMES = ["spec_slope", "spec_centroid", "skew", "kurt", "peak_rms",
                "grad_ratio", "anisotropy"]
B_NAMES = B_SNAP_NAMES + ["var_cv_t", "lag1_corr_t"]
BAMP_NAMES = ["log_var_raw", "log_energy_raw"]


def spectrum(field, kind):
    """Return (k, S(k)).

    kind = "periodic_vorticity": isotropic shell spectrum of the kinetic
           energy implied by a vorticity field on a 2*pi-periodic box,
           E(k) = sum_shell |w_k|^2 / (2 k^2).
    kind = "rb_scalar": horizontal (x) variance spectrum of a scalar field,
           averaged over the rows z (x is the only periodic direction).
    """
    if kind == "periodic_vorticity":
        N = field.shape[0]
        wh = np.fft.rfft2(field) / field.size
        k = np.fft.fftfreq(N, 1.0 / N)
        kr = np.fft.rfftfreq(N, 1.0 / N)
        KY, KX = np.meshgrid(k, kr, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        w = np.full(KX.shape, 2.0)
        w[:, 0] = 1.0
        if N % 2 == 0:
            w[:, -1] = 1.0
        e = w * np.abs(wh) ** 2 / (2 * np.where(K2 > 0, K2, np.inf))
        shell = np.rint(np.sqrt(K2)).astype(int)
        kmax = N // 3
        S = np.bincount(shell.ravel(), e.ravel(), minlength=kmax + 1)[: kmax + 1]
        return np.arange(kmax + 1), S
    if kind == "rb_scalar":
        Nx = field.shape[1]
        fh = np.fft.rfft(field, axis=1) / Nx
        S = (np.abs(fh) ** 2).mean(axis=0)
        S[1:] *= 2
        kmax = Nx // 3
        return np.arange(kmax + 1), S[: kmax + 1]
    raise ValueError(kind)


def spectral_slope(k, S, band, floor=1e-12):
    """Least-squares slope of log S vs log k over the band; S floored at
    `floor` x total so that spectra with empty bands (steady laminar
    states) give a finite slope."""
    m = (k >= band[0]) & (k <= band[1])
    s = np.maximum(S[m], floor * S.sum() + 1e-300)
    return float(np.polyfit(np.log(k[m]), np.log(s), 1)[0])


def baseline_snapshot(q, kind, band):
    """The 7 spatial features of one (normalised) snapshot."""
    k, S = spectrum(q, kind)
    gy, gx = np.gradient(q)
    v = q.var() + 1e-30
    return np.nan_to_num(np.array([
        spectral_slope(k, S, band),
        float((k[1:] * S[1:]).sum() / (S[1:].sum() + 1e-300)),
        float(skew(q.ravel())), float(kurtosis(q.ravel())),
        float(np.abs(q).max() / np.sqrt(v)),
        float((gx ** 2 + gy ** 2).mean() / v),
        float((gx ** 2).mean() / ((gx ** 2).mean() + (gy ** 2).mean() + 1e-30))]))


def baseline_B(window, kind, band):
    """Window mode: spatial features averaged over the window + 2 temporal ones."""
    feats = np.mean([baseline_snapshot(q, kind, band) for q in window], axis=0)
    var_t = window.reshape(window.shape[0], -1).var(axis=1)
    var_cv = var_t.std() / (var_t.mean() + 1e-30)
    flat = window.reshape(window.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    num = (flat[1:] * flat[:-1]).sum(axis=1)
    den = np.sqrt((flat[1:] ** 2).sum(axis=1) * (flat[:-1] ** 2).sum(axis=1)) + 1e-30
    lag1 = float(np.mean(num / den))
    return np.concatenate([feats, [var_cv, lag1]])


def baseline_Bamp(raw_window, kind):
    var = float(np.mean([q.var() for q in raw_window]))
    en = float(np.mean([spectrum(q, kind)[1].sum() for q in raw_window]))
    return np.array([np.log10(var + 1e-30), np.log10(en + 1e-30)])
