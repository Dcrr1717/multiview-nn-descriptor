"""
Sparse, irregular "station" observations of a 2D field (Plan C1).

A station network is a fixed set of M grid points drawn uniformly at random
(seeded). A sample is the vector of field values at the stations at one
instant, so no Fourier spectrum can be computed and only scattered values
are available, as with weather stations, tide gauges or floats.

* X2s (R^12) - persistent homology of the lower-star (sublevel) filtration of
  the station values on the Delaunay triangulation of the stations
  (periodic directions handled by tiling), summarised with the original
  `sig6` for H0 and H1.
* BS  (R^7)  - station baseline, the analogue of the spectral baseline B:
  log-log slope of the empirical semivariogram at short lags (the analogue of
  the spectral slope), normalised short-lag semivariance, skewness, kurtosis,
  peak/rms, directional anisotropy of the short-lag semivariogram, and the
  fraction of stations above +1 sigma.
* BSamp (R^1) - log10 variance of the raw station values (amplitude proxy).
"""
from __future__ import annotations

import gudhi
import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import kurtosis, skew

from tsg.descriptors import sig6

BS_NAMES = ["vario_slope", "vario_short", "skew", "kurt", "peak_rms", "vario_aniso",
            "frac_above_1sd"]


def station_network(shape, M, periodic, seed, extent):
    """Random station network on a grid of `shape` = (H, W).

    extent = (Ly, Lx) physical size; periodic = (bool, bool) per axis.
    Returns grid indices (iy, ix), physical positions (M, 2) as (y, x), and
    the Delaunay triangles of the (tiled, where periodic) network expressed
    in station ids.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    flat = rng.choice(H * W, size=M, replace=False)
    iy, ix = np.unravel_index(flat, shape)
    pos = np.column_stack([iy * extent[0] / H, ix * extent[1] / W])
    shifts_y = [-1, 0, 1] if periodic[0] else [0]
    shifts_x = [-1, 0, 1] if periodic[1] else [0]
    tiles, ids = [], []
    for sy in shifts_y:
        for sx in shifts_x:
            tiles.append(pos + np.array([sy * extent[0], sx * extent[1]]))
            ids.append(np.arange(M))
    P, I = np.vstack(tiles), np.concatenate(ids)
    tri = Delaunay(P).simplices
    centre = np.zeros(len(P), bool)
    c0 = (shifts_y.index(0) * len(shifts_x) + shifts_x.index(0)) * M
    centre[c0:c0 + M] = True
    keep = tri[centre[tri].any(axis=1)]
    simp = np.unique(np.sort(I[keep], axis=1), axis=0)
    simp = simp[(simp[:, 0] != simp[:, 1]) & (simp[:, 1] != simp[:, 2])]
    return (iy, ix), pos, simp


def lower_star_persistence(values, triangles, min_pers=0.01):
    st = gudhi.SimplexTree()
    for i, v in enumerate(values):
        st.insert([i], filtration=float(v))
    for t in triangles:
        a, b, c = (int(x) for x in t)
        for e in ((a, b), (a, c), (b, c)):
            st.insert(list(e), filtration=float(max(values[e[0]], values[e[1]])))
        st.insert([a, b, c], filtration=float(max(values[a], values[b], values[c])))
    st.compute_persistence(homology_coeff_field=2, min_persistence=min_pers)
    out = []
    for dim in (0, 1):
        iv = st.persistence_intervals_in_dimension(dim)
        iv = iv[np.isfinite(iv[:, 1])] if len(iv) else np.zeros((0, 2))
        out.append(iv[:, 1] - iv[:, 0] if len(iv) else np.zeros(0))
    return out


def x2_stations(values, triangles, min_pers=0.01):
    h0, h1 = lower_star_persistence(values, triangles, min_pers)
    return np.concatenate([sig6(h0), sig6(h1)])


def _pairs(pos, extent, periodic):
    d = pos[:, None, :] - pos[None, :, :]
    for ax in (0, 1):
        if periodic[ax]:
            L = extent[ax]
            d[..., ax] -= L * np.round(d[..., ax] / L)
    iu = np.triu_indices(len(pos), 1)
    return d[..., 0][iu], d[..., 1][iu], iu


def make_baseline(pos, extent, periodic, n_bins=12):
    """Precompute the pair geometry once per network; returns f(values) -> BS."""
    dy, dx, iu = _pairs(pos, extent, periodic)
    h = np.hypot(dy, dx)
    hmax = 0.5 * min(extent)
    edges = np.linspace(0, hmax, n_bins + 1)
    b = np.digitize(h, edges) - 1
    ok = (b >= 0) & (b < n_bins)
    short = b == 1                                 # second lag bin (first is near-empty)
    along_x = np.abs(dx) > np.abs(dy)
    centers = 0.5 * (edges[1:] + edges[:-1])

    def f(values):
        v = (values - values.mean()) / (values.std() + 1e-30)
        sq = 0.5 * (v[iu[0]] - v[iu[1]]) ** 2
        gam = np.bincount(b[ok], sq[ok], minlength=n_bins) / np.maximum(
            np.bincount(b[ok], minlength=n_bins), 1)
        lags = slice(1, 4)                        # short-lag range for the slope
        g = np.maximum(gam[lags], 1e-12)
        slope = float(np.polyfit(np.log(centers[lags]), np.log(g), 1)[0])
        gx = sq[short & along_x].mean() if (short & along_x).any() else 0.0
        gy = sq[short & ~along_x].mean() if (short & ~along_x).any() else 0.0
        return np.nan_to_num(np.array([
            slope, float(gam[1]), float(skew(v)), float(kurtosis(v)),
            float(np.abs(v).max()), float(gx / (gx + gy + 1e-30)), float((v > 1).mean())]))
    return f
