"""Single source of truth for every simulation / descriptor parameter.

The regime bands below were fixed from the descriptor-free pilot scans in
results/pilot/ (bifurcation diagnostics only) BEFORE any descriptor or
classifier was computed on the experimental data; see PROTOCOL.md.
"""

SEED = 2024  # global seed, same as the neural-network study

_COMMON = {
    "n_traj": 4,          # independent random initial conditions per parameter value
    "t_spin": 300.0,      # spin-up (time units) before labelling / sampling
    "t_lyap": 100.0,      # twin-trajectory Lyapunov estimate (labelling)
    "n_windows": 12,      # samples per trajectory
    "T_win": 16,          # frames per window (window mode); snapshot mode uses frame 0
    "dt_sample": 0.5,     # time between frames
    "gap": 12.0,          # time between consecutive windows
    "blocks": (8, 8),     # 64 regions = graph nodes (analogue of the 80-neuron cap)
    "theta": 0.40,        # correlation threshold of the functional graph (paper value)
    "min_pers": 0.01,     # persistence floor, in units of the field std
    "n_snap": 4,          # snapshots per window used for X2 in window mode
}

FLOWS = {
    # ── Kolmogorov flow (doubly periodic, n = 4 forcing, nu = 1/Re) ─────────
    "kolmogorov": dict(_COMMON, **{
        "N": 128,             # grid points per direction (same grid for every Re)
        "n_force": 4,
        "ic_amp": 1.0,        # rms of the random perturbation added to the laminar state
        "pilot": [6, 8, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25, 30, 40, 60, 100],
        "design": {"laminar": [5, 6, 7, 8],
                   "transitional": [12, 15, 18, 22],
                   "turbulent": [40, 55, 70, 100]},
        "periodic": (True, True),
        "spec_kind": "periodic_vorticity",
        "slope_band": (5, 30),     # above the forcing wavenumber, below the 2/3 cut-off (42)
    }),
    # ── Rayleigh-Benard convection (free-slip, periodic in x, Pr = 1) ───────
    "rb": dict(_COMMON, **{
        "Pr": 1.0,
        "Gamma": 2.0,
        "Nx": 128,
        "Nz": 64,             # points on the physical height [0, 1)
        "ic_amp": 1e-2,
        "pilot": [2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5],
        # pilot: steady rolls up to Ra = 3.5e5, chaotic shear-bursting from 4e5 on,
        # no time-periodic window -> two regimes (PROTOCOL.md 2.1)
        "design": {"laminar": [5e4, 1e5, 2e5, 3e5],
                   "turbulent": [4e5, 5e5, 7e5, 1e6]},
        # Amendment 1 (PROTOCOL.md 13): the intermittent bursting state needs a
        # longer twin-trajectory window to resolve lambda_1 > 0 (block SEM)
        "t_lyap": 300.0,
        "periodic": (False, True),  # (z, x): walls in z, periodic in x
        "spec_kind": "rb_scalar",
        "slope_band": (2, 30),
    }),
}

# backwards-compatible aliases used by scripts/00_pilot_regimes.py
KOLMO = dict(FLOWS["kolmogorov"], pilot_Re=FLOWS["kolmogorov"]["pilot"])
RB = dict(FLOWS["rb"], pilot_Ra=FLOWS["rb"]["pilot"])


LYAP_CHAOS = 0.01   # Amendment 2 (PROTOCOL.md 13): point-estimate threshold for chaos


def label_trajectory(meta, flow, rule="amended"):
    """Descriptor-free regime label of one trajectory.

    turbulent    : chaotic, largest Lyapunov exponent lambda_1 > 0.01
                   (point estimate; Amendment 2). Sensitivity variants:
                   rule="frozen" applies the original lambda_1 - 2 SEM > 0;
                   rule="steady_first" labels a trajectory that is steady over
                   the sampling period (CV of the global observable < 1e-3)
                   as non-turbulent whatever lambda_1 (weakly unstable steady
                   states, e.g. RB rolls at Ra = 4e5 before they burst);
    laminar      : Kolmogorov - converged to the laminar fixed point
                   (relative distance < 1e-3);
                   RB - steady convection (CV of Nu over the sampling
                   period < 1e-3);
    transitional : everything else (non-laminar coherent states: steady,
                   travelling, periodic or quasi-periodic).
    """
    chaotic = (meta["lyap"] - 2 * meta["lyap_sem"] > 0 if rule == "frozen"
               else meta["lyap"] > LYAP_CHAOS)
    if rule == "steady_first" and meta["glob_cv"] < 1e-3:
        chaotic = False
    if chaotic:
        return "turbulent"
    if flow == "kolmogorov" and meta["dist_ref"] < 1e-3:
        return "laminar"
    if flow == "rb" and meta["glob_cv"] < 1e-3:
        return "laminar"
    return "transitional"
