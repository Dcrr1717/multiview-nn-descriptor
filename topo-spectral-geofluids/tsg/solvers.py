"""
Pseudo-spectral solvers for two canonical 2D geophysical-fluid flows.

Both solvers integrate the vorticity(-temperature) formulation with an
integrating-factor fourth-order Runge-Kutta scheme (IF-RK4): the linear
diffusion operators are integrated exactly in Fourier space and the
nonlinear (advection, buoyancy, forcing) terms explicitly. Products are
de-aliased with the 2/3 rule. Time steps are chosen adaptively from a CFL
condition (re-evaluated every 20 steps) and every output interval is hit
exactly, so snapshots are sampled on a uniform clock.

1. KolmogorovFlow
   Doubly periodic box [0, 2*pi)^2, body force f = (sin(n y), 0):

       d_t w + u . grad w = nu lap w - alpha w - n cos(n y),   nu = 1/Re

   (Chandler & Kerswell 2013, J. Fluid Mech. 722; Platt, Sirovich &
   Fitzmaurice 1991, Phys. Fluids A 3). Laminar fixed point:
   w_L = -(Re/n) cos(n y).

2. RayleighBenard
   Boussinesq convection between free-slip, isothermal plates, periodic in
   x, in free-fall units (length H, velocity sqrt(g a dT H)):

       d_t w + u . grad w = sqrt(Pr/Ra) lap w + d_x theta
       d_t theta + u . grad theta = w + lap theta / sqrt(Ra Pr)

   with T = 1 - z + theta. Free-slip/isothermal walls make psi, w and theta
   pure sine series in z; we use the equivalent odd extension to a
   z-periodic box of height 2 and project onto the odd subspace after every
   step, so a plain 2D FFT is exact for these boundary conditions.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft


def _rk4_if(state, rhs, E, E2, dt):
    """One integrating-factor RK4 step for a tuple of spectral fields.

    `rhs(state) -> tuple` returns the explicit (nonlinear + forcing) tendency;
    E = exp(L dt), E2 = exp(L dt / 2) are per-field integrating factors.
    """
    k1 = rhs(state)
    s2 = tuple(e2 * (s + 0.5 * dt * k) for s, k, e2 in zip(state, k1, E2))
    k2 = rhs(s2)
    s3 = tuple(e2 * s + 0.5 * dt * k for s, k, e2 in zip(state, k2, E2))
    k3 = rhs(s3)
    s4 = tuple(e * s + dt * e2 * k for s, k, e, e2 in zip(state, k3, E, E2))
    k4 = rhs(s4)
    return tuple(e * s + dt / 6.0 * (e * a + 2.0 * e2 * (b + c) + d)
                 for s, a, b, c, d, e, e2 in zip(state, k1, k2, k3, k4, E, E2))


class KolmogorovFlow:
    """2D Kolmogorov flow in a 2*pi-periodic box (vorticity form)."""

    def __init__(self, Re: float, N: int = 128, n_force: int = 4,
                 alpha: float = 0.0, cfl: float = 0.5, dt_max: float = 0.05):
        self.Re, self.N, self.n, self.alpha = float(Re), int(N), int(n_force), float(alpha)
        self.nu = 1.0 / self.Re
        self.cfl, self.dt_max = cfl, dt_max
        self.dx = 2 * np.pi / N
        k = sfft.fftfreq(N, 1.0 / N)                 # integer wavenumbers
        kr = sfft.rfftfreq(N, 1.0 / N)
        self.ky, self.kx = np.meshgrid(k, kr, indexing="ij")   # arrays (Ny, Nx/2+1)
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2inv = np.where(self.k2 > 0, 1.0 / np.where(self.k2 > 0, self.k2, 1), 0.0)
        kmax = N // 3                                 # 2/3 de-aliasing
        self.dealias = (np.abs(self.kx) <= kmax) & (np.abs(self.ky) <= kmax)
        self.L = -self.nu * self.k2 - self.alpha
        y = np.arange(N) * self.dx
        self.Y, self.X = np.meshgrid(y, y, indexing="ij")
        self.F_hat = self.fft(-self.n * np.cos(self.n * self.Y))
        self._cache = {}

    # ── transforms ───────────────────────────────────────────────────
    def fft(self, f):
        return sfft.rfft2(f)

    def ifft(self, fh):
        return sfft.irfft2(fh, s=(self.N, self.N))

    def laminar(self):
        return -(self.Re / self.n) * np.cos(self.n * self.Y) if self.alpha == 0 else \
            self.ifft(self.F_hat / (self.nu * self.k2 + self.alpha + 1e-300) * (self.k2 > 0))

    def velocity(self, wh):
        psih = wh * self.k2inv
        return self.ifft(1j * self.ky * psih), self.ifft(-1j * self.kx * psih)

    def _rhs(self, state):
        (wh,) = state
        u, v = self.velocity(wh)
        wx, wy = self.ifft(1j * self.kx * wh), self.ifft(1j * self.ky * wh)
        nl = -self.fft(u * wx + v * wy)
        return (self.dealias * nl + self.F_hat,)

    def _factors(self, dt):
        key = round(dt, 14)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = ((np.exp(self.L * dt),), (np.exp(self.L * dt / 2),))
        return self._cache[key]

    def cfl_dt(self, wh):
        u, v = self.velocity(wh)
        umax = max(np.abs(u).max(), np.abs(v).max(), 1e-8)
        return min(self.cfl * self.dx / umax, self.dt_max)

    def advance(self, wh, T, recheck=20):
        """Integrate the spectral vorticity `wh` exactly by time T. The CFL
        time step is re-evaluated every `recheck` steps and the last step
        lands exactly on T."""
        t = 0.0
        while T - t > 1e-12:
            rem = T - t
            n = max(1, int(np.ceil(rem / self.cfl_dt(wh))))
            dt = rem / n
            E, E2 = self._factors(dt)
            m = min(n, recheck)
            for _ in range(m):
                (wh,) = _rk4_if((wh,), self._rhs, E, E2, dt)
            t = T if m == n else t + m * dt
        return wh

    # ── diagnostics ──────────────────────────────────────────────────
    def diagnostics(self, wh):
        w = self.ifft(wh)
        u, v = self.velocity(wh)
        E = 0.5 * np.mean(u ** 2 + v ** 2)
        Z = 0.5 * np.mean(w ** 2)
        D = self.nu * np.mean(w ** 2)                        # viscous dissipation
        I = np.mean(u * np.sin(self.n * self.Y))             # power input
        return {"E": E, "Z": Z, "D": D, "I": I}

    def random_ic(self, rng, amp=1.0, kmin=1, kmax=8, add_laminar=False):
        """Random-phase vorticity with flat spectrum in [kmin, kmax], rms = amp."""
        kk = np.sqrt(self.k2)
        ph = np.exp(2j * np.pi * rng.random(self.k2.shape))
        mag = ((kk >= kmin) & (kk <= kmax)).astype(float) * rng.standard_normal(self.k2.shape)
        w = self.ifft(mag * ph)
        w *= amp / (w.std() + 1e-30)
        if add_laminar:
            w = w + self.laminar()
        return self.fft(w)


class RayleighBenard:
    """2D free-slip Rayleigh-Benard convection (Boussinesq, free-fall units)."""

    def __init__(self, Ra: float, Pr: float = 1.0, Gamma: float = 2.0,
                 Nx: int = 128, Nz: int = 64, cfl: float = 0.5, dt_max: float = 0.02):
        self.Ra, self.Pr, self.Gamma = float(Ra), float(Pr), float(Gamma)
        self.Nx, self.Nz = int(Nx), int(Nz)        # Nz = points on the physical [0, 1)
        self.Nzf = 2 * self.Nz                      # odd-extended periodic height 2
        self.cfl, self.dt_max = cfl, dt_max
        self.dx, self.dz = Gamma / Nx, 1.0 / Nz
        self.nu = np.sqrt(Pr / Ra)
        self.kappa = 1.0 / np.sqrt(Ra * Pr)
        kz = sfft.fftfreq(self.Nzf, 1.0 / self.Nzf) * np.pi        # period 2
        kx = sfft.rfftfreq(self.Nx, 1.0 / self.Nx) * (2 * np.pi / Gamma)
        self.kz, self.kx = np.meshgrid(kz, kx, indexing="ij")
        self.k2 = self.kx ** 2 + self.kz ** 2
        self.k2inv = np.where(self.k2 > 0, 1.0 / np.where(self.k2 > 0, self.k2, 1), 0.0)
        mx, mz = self.Nx // 3, self.Nzf // 3
        self.dealias = (np.abs(self.kx) <= mx * 2 * np.pi / Gamma) & (np.abs(self.kz) <= mz * np.pi)
        self.Lw, self.Lt = -self.nu * self.k2, -self.kappa * self.k2
        self.neg = (-np.arange(self.Nzf)) % self.Nzf                 # index of -kz
        z = np.arange(self.Nzf) * (2.0 / self.Nzf)
        x = np.arange(self.Nx) * self.dx
        self.Z, self.X = np.meshgrid(z, x, indexing="ij")
        self._cache = {}

    def fft(self, f):
        return sfft.rfft2(f)

    def ifft(self, fh):
        return sfft.irfft2(fh, s=(self.Nzf, self.Nx))

    def odd(self, fh):
        """Project a spectral field onto the z-odd (pure sine) subspace."""
        return 0.5 * (fh - fh[self.neg, :])

    def velocity(self, wh):
        psih = wh * self.k2inv
        return self.ifft(1j * self.kz * psih), self.ifft(-1j * self.kx * psih)  # u, w

    def _rhs(self, state):
        wh, th = state
        u, w = self.velocity(wh)
        ax = lambda fh: self.ifft(1j * self.kx * fh)
        az = lambda fh: self.ifft(1j * self.kz * fh)
        nl_w = -self.fft(u * ax(wh) + w * az(wh))
        nl_t = -self.fft(u * ax(th) + w * az(th))
        return (self.dealias * nl_w + 1j * self.kx * th,
                self.dealias * nl_t + self.fft(w))

    def _factors(self, dt):
        key = round(dt, 14)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = ((np.exp(self.Lw * dt), np.exp(self.Lt * dt)),
                                (np.exp(self.Lw * dt / 2), np.exp(self.Lt * dt / 2)))
        return self._cache[key]

    def cfl_dt(self, wh):
        u, w = self.velocity(wh)
        h = min(self.dx, self.dz)
        return min(self.cfl * h / max(np.abs(u).max(), np.abs(w).max(), 1e-8), self.dt_max)

    def advance(self, state, T, recheck=20):
        """Integrate (w, theta) exactly by time T, CFL re-evaluated every
        `recheck` steps (see KolmogorovFlow.advance)."""
        wh, th = state
        t = 0.0
        while T - t > 1e-12:
            rem = T - t
            n = max(1, int(np.ceil(rem / self.cfl_dt(wh))))
            dt = rem / n
            E, E2 = self._factors(dt)
            m = min(n, recheck)
            for _ in range(m):
                wh, th = _rk4_if((wh, th), self._rhs, E, E2, dt)
                wh, th = self.odd(wh), self.odd(th)
            t = T if m == n else t + m * dt
        return wh, th

    def physical(self, fh):
        """Field on the physical domain z in [0, 1): array (Nz, Nx)."""
        return self.ifft(fh)[: self.Nz]

    def diagnostics(self, state):
        wh, th = state
        u, w = self.velocity(wh)
        th_ = self.ifft(th)
        half = slice(0, self.Nz)
        E = 0.5 * np.mean(u[half] ** 2 + w[half] ** 2)
        Nu = 1.0 + np.mean(w[half] * th_[half]) / self.kappa
        return {"E": E, "Nu": Nu, "theta_rms": th_[half].std()}

    def random_ic(self, rng, amp=1e-2):
        th = amp * rng.standard_normal((self.Nzf, self.Nx))
        th = self.odd(self.fft(th)) * self.dealias
        return (np.zeros_like(th), th)
