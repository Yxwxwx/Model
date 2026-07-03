"""Reproduce Colbert–Miller JCP 96, 1982 (1992) — Section II.B, Figs. 1–2.

Fig. 1: V_c large (box not limiting) → pure Δx discretisation error.
Fig. 2: Δx small (discretisation not limiting) → pure V_c boundary error.
"""

import numpy as np
from dvr import SincDVR
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATES = [0, 2, 4, 6, 8]


def make_grid(dx: float, v_cut: float):
    """Truncated uniform grid per Eq. (2.5): keep x_i = i·Δx where V(x_i) ≤ V_c."""
    i_max = int(np.floor(np.sqrt(2.0 * v_cut) / dx))
    if i_max < 2:
        i_max = 2
    xs = np.arange(-i_max, i_max + 1) * dx
    dvr = SincDVR(xs[0], xs[-1], len(xs), mass=1.0)
    return xs, dvr


def fractional_errors_dx(v_cut: float, dx_vals):
    """Compute |E_DVR/E_exact - 1| for each Δx at fixed V_c."""
    errs = {n: [] for n in STATES}
    for dx in dx_vals:
        xs, dvr = make_grid(dx, v_cut)
        V = 0.5 * xs * xs
        n_states = min(max(STATES) + 1, len(xs))
        result = dvr.solve(V, n_states)
        for n in STATES:
            if n < len(result.energies):
                errs[n].append(abs(result.energies[n] / (n + 0.5) - 1.0))
            else:
                errs[n].append(np.nan)
    return errs


def fractional_errors_vc_step(dx: float, i_max_list):
    """Compute |E_DVR/E_exact - 1| for each grid size i_max (step-function data).

    Returns dict {n: list_of_errors} and list of V_c thresholds (upper edge of each step).
    The error is constant for V_c ∈ [(i·dx)²/2, ((i+1)·dx)²/2).
    """
    thresholds = []
    errs = {n: [] for n in STATES}
    for i_max in i_max_list:
        xs = np.arange(-i_max, i_max + 1) * dx
        vc_upper = ((i_max + 1) * dx) ** 2 / 2  # next threshold
        thresholds.append(vc_upper)
        if len(xs) < max(STATES) + 1:
            for n in STATES:
                errs[n].append(np.nan)
            continue
        dvr = SincDVR(xs[0], xs[-1], len(xs), mass=1.0)
        V = 0.5 * xs * xs
        result = dvr.solve(V.tolist(), max(STATES) + 1)
        for n in STATES:
            if n < len(result.energies):
                errs[n].append(abs(result.energies[n] / (n + 0.5) - 1.0))
            else:
                errs[n].append(np.nan)
    return errs, thresholds


def plot_fig1(v_cut: float = 20.0):
    """Paper Fig. 1: fractional error vs Δx at large, fixed V_c."""
    dx_vals = np.linspace(0.5, 2.5, 100)
    print(f"Fig. 1: V_c = {v_cut:.0f}, Δx ∈ [{dx_vals[0]:.2f}, {dx_vals[-1]:.2f}]")
    errs = fractional_errors_dx(v_cut, dx_vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    for n in STATES:
        y = np.array(errs[n])
        ax.semilogy(dx_vals[~np.isnan(y)], y[~np.isnan(y)], label=f"$n = {n}$")

    ax.set_xlabel(r"$\Delta x$", fontsize=13)
    ax.set_ylabel(r"$|E_{\rm DVR}/E_{\rm exact} - 1|$", fontsize=12)
    ax.set_title(
        f"Colbert–Miller Fig. 1 — harmonic oscillator ($V_c = {v_cut:.0f}$)",
        fontsize=13,
    )
    ax.legend(fontsize=11, framealpha=0.7)
    ax.set_xlim(0.45, 2.55)
    ax.set_ylim(5e-8, 2)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("fig1_harmonic_dx.png", dpi=150)
    print("  → fig1_harmonic_dx.png")


def plot_fig2(dx: float = 0.1):
    """Paper Fig. 2: fractional error vs V_c at small, fixed Δx.

    V_c is continuous — the grid expands at discrete thresholds where i_max jumps.
    We sample at the _midpoint_ V_c of each constant-grid step and connect with
    lines, producing smooth curves that match the paper.
    """
    i_max_vals = np.arange(2, int(np.sqrt(2 * 18) / dx) + 2)
    print(f"Fig. 2: Δx = {dx:.2f}, i_max ∈ [{i_max_vals[0]}, {i_max_vals[-1]}]")
    errs, vc_thresh = fractional_errors_vc_step(dx, i_max_vals)

    # Midpoint V_c of each step: vc_mid[i] = (vc_thresh[i-1] + vc_thresh[i]) / 2
    vc_mid = np.array(vc_thresh)
    vc_mid[:-1] = 0.5 * (np.array(vc_thresh[:-1]) + np.array(vc_thresh[1:]))
    # First element: lower bound is (i_max[0] * dx)^2 / 2
    vc_mid[0] = 0.5 * ((i_max_vals[0] * dx) ** 2 / 2 + vc_thresh[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    for n in STATES:
        y = np.array(errs[n])
        mask = ~np.isnan(y)
        ax.semilogy(vc_mid[mask], y[mask], "-", label=f"$n = {n}$")

    ax.set_xlabel(r"$V_c$  ($\hbar\omega$)", fontsize=13)
    ax.set_ylabel(r"$|E_{\rm DVR}/E_{\rm exact} - 1|$", fontsize=12)
    ax.set_title(
        f"Colbert–Miller Fig. 2 — harmonic oscillator ($\\Delta x = {dx:.2f}$)",
        fontsize=13,
    )
    ax.legend(fontsize=11, framealpha=0.7)
    ax.set_xlim(1.5, 18.5)
    ax.set_ylim(5e-8, 2)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("fig2_harmonic_vc.png", dpi=150)
    print("  → fig2_harmonic_vc.png")


if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
