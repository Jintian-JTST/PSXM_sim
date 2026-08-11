"""chi2_scan.py -- formalize and validate the "minimized chi2" shield method.

The shield solve is a genuine chi-square minimization, not an exact null.
This script makes that explicit by separating:

  OUTPUT (unknowns)    : shield_n discrete current elements, located ON the
                         shield ring (radius = tpl.shield_radius).
  INPUT (constraints)  : B=0 samples on 2 rings offset radially from the
                         shield (at gap_mm and gap_mm/2, so the two rings
                         never coincide anywhere in the scan), with
                         n_between points in each azimuthal gap between
                         adjacent shield elements on each ring (shared
                         layout: shield_common.shield_zero_solver).

    chi2 = sum over all sample points of (Bx_pred^2 + By_pred^2)

Because the number of sample points (input) is deliberately made larger
than the number of shield current elements (output), lstsq is not solving
an exact system -- it is minimizing chi2 over an over-determined one. This
script scans (gap_mm, n_between) and reports, for each combination:

  1. redundancy factor        n_equations / n_unknowns
  2. achieved chi2            rms residual field at the samples, relative
                               to the typical field there (should be small
                               but is NOT exactly zero -- proof it's a true
                               least-squares fit, not a tautology)
  3. leakage suppression       at the 0.419 m nearest-beam benchmark --
                               the quantity the sampling layout is
                               supposed to control

so the (gap_mm, n_between, shield_n) choice used in shield_common.py can be
picked from data instead of by eye, and "does increasing sampling points
help?" has a quantitative answer.

Run:  python chi2_scan.py     (prints a table, saves figures/chi2_parameter_scan.png)
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from shield_common import shield_zero_solver

MU0 = 4e-7 * np.pi
I_COIL = 1000.0
R_BENCH_MM = 419.0      # nearest-beam benchmark


def quad_currents():
    return np.array([0, I_COIL, I_COIL, 0, -I_COIL, -I_COIL], float)


def build_shield_solver(tpl, gap_mm, n_between, outer_mm):
    """OUTPUT = tpl's shield_n current elements (on the shield ring).
    INPUT = B=0 samples on (shield+gap_mm) and (shield+outer_mm) rings
    (shared layout, see shield_common.shield_zero_solver), n_between
    points per azimuthal gap between shield elements."""
    solver = shield_zero_solver(tpl, gap_mm=gap_mm, outer_mm=outer_mm, n_between=n_between)
    KM = solver.coefficient_matrix() @ tpl.group_matrix()
    K6, Ksh = KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:]
    return K6, Ksh


def ring_meanB(coils, rho, n=96):
    vals = []
    for a in np.linspace(0.017, 2 * np.pi + 0.017, n, endpoint=False):
        try:
            vals.append(coils.B_magnitude(rho * np.cos(a), rho * np.sin(a)))
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else np.nan


def run_case(shield_n, gap_mm, n_between):
    """outer_mm = gap_mm/2 (rather than a fixed constant): the two B=0
    rings then always sit at two DISTINCT radii for any gap_mm in the
    scan (a fixed outer_mm would coincide with gap_mm at one point in the
    grid, making the two rings redundant instead of adding constraints)."""
    tpl = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=shield_n)
    I_c = quad_currents()
    K6, Ksh = build_shield_solver(tpl, gap_mm, n_between, outer_mm=gap_mm / 2.0)

    n_eq, n_unk = Ksh.shape
    X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
    I_s = (-X) @ I_c

    # achieved chi2: residual field at every sample point after the solve.
    # This is NOT identically zero -- that's the point: it's a minimized
    # (not exactly satisfied) chi2 because n_eq > n_unk.
    resid = Ksh @ I_s + K6 @ I_c
    chi2 = float(np.sum(resid ** 2))
    B_scale = float(np.max(np.abs(K6 @ I_c)))
    chi2_rel = float(np.sqrt(chi2 / n_eq) / B_scale)

    # leakage suppression at the benchmark distance
    shielded = PSXMCoils(currents=I_c, shield=True, shield_radius=tpl.shield_radius,
                         shield_n=shield_n, shield_currents=I_s)
    unshielded = PSXMCoils(currents=I_c)
    bu, bs = ring_meanB(unshielded, R_BENCH_MM), ring_meanB(shielded, R_BENCH_MM)
    suppression = float(bu / bs) if bs > 0 else np.inf

    return dict(shield_n=shield_n, gap_mm=gap_mm, n_between=n_between,
                n_eq=n_eq, n_unk=n_unk, redundancy=n_eq / n_unk,
                chi2_rel=chi2_rel, suppression=suppression)


def main():
    shield_n = 200
    gaps = (1.0, 2.0, 5.0, 10.0, 20.0)
    n_betweens = (1, 3, 6)

    rows = []
    print(f"{'gap_mm':>7} {'n_between':>9} {'redund.':>8} {'chi2_rel':>10} "
          f"{'suppress@0.419m':>16}")
    for nb in n_betweens:
        for g in gaps:
            r = run_case(shield_n, g, nb)
            rows.append(r)
            print(f"{g:7.1f} {nb:9d} {r['redundancy']:8.1f} {r['chi2_rel']:10.2e} "
                  f"{r['suppression']:16.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for nb in n_betweens:
        rs = [r for r in rows if r["n_between"] == nb]
        ax[0].semilogy([r["gap_mm"] for r in rs], [r["suppression"] for r in rs],
                       "o-", label=f"n_between={nb}")
        ax[1].semilogy([r["gap_mm"] for r in rs], [r["chi2_rel"] for r in rs],
                       "o-", label=f"n_between={nb}")
    ax[0].set_xlabel("sample gap (mm)")
    ax[0].set_ylabel("leakage suppression at 0.419 m")
    ax[0].set_title("does the design outcome depend on the sampling layout?")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    ax[1].set_xlabel("sample gap (mm)")
    ax[1].set_ylabel("rms residual B / typical B  (achieved chi2)")
    ax[1].set_title("achieved chi2 (small but nonzero => real least squares)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("figures/chi2_parameter_scan.png", dpi=150)
    print("\nsaved figures/chi2_parameter_scan.png")


if __name__ == "__main__":
    main()
