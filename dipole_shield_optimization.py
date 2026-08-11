"""dipole_shield_optimization.py -- dipole field generation and shielding
parameter optimization for the PSXM.

Everything here uses the one method of the technical note: the weighted
least-squares inverse solve I_shield = S @ I_coil, S = -K_s^+ K_6, from
nulling B on sample rings radially offset from the shield.  Output =
shield current points; input = B=0 sample locations; the system is
deliberately over-determined, so the solve minimizes a residual rather
than satisfying an exact null.

Three parts:

1. DIPOLE FIELD GENERATION
   Solves the 6 PSXM coil currents needed to produce a uniform Bx = B0
   dipole field at the centre (as opposed to the quadrupole studied
   elsewhere).

2. SHIELDING PARAMETER SCAN
   Scans shield_n, gap_mm and n_between, reporting the redundancy of the
   least-squares system, the achieved residual, and the leakage
   suppression at the 0.419 m benchmark -- i.e. whether the design
   outcome depends on the sampling layout.

3. LEAKAGE FIGURE
   A two-panel field/leakage figure for the dipole at the chosen
   parameter set.

Run:  python dipole_shield_optimization.py
      (saves figures/dipole_parameter_scan.png,
       figures/dipole_shield_leakage.png)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from shield_common import (make_template, solve_dipole_coils, ls_shield_currents,
                           build_pair, ring_meanB, leakage_report,
                           shield_zero_solver, MAX_CURRENT,
                           R_MAX_MM, MARKS_MM)

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

import matplotlib
matplotlib.use("Agg")

# --- configuration ----------------------------------------------------------
B0 = 1e-3                      # T, central dipole target (Bx = B0, By = 0)
SHIELD_NS = (60, 100, 200, 400)        # shield discretization points to scan
GAPS_MM = (1.0, 2.0, 5.0, 10.0, 20.0)  # sample gaps to scan
N_BETWEEN = 3                  # samples between adjacent shield points (default)
BENCH_R_MM = 419.0             # nearest-beam benchmark distance (mm)


# ============================================================================
# Part 1: Dipole field generation and current solve
# ============================================================================
def part1_dipole_solve():
    """Solve for dipole coil currents and print the result."""
    tpl = make_template(shield_n=200)
    I_coil = solve_dipole_coils(tpl, B0=B0)

    print("=" * 65)
    print("PART 1: DIPOLE FIELD GENERATION")
    print("=" * 65)
    print(f"Target: uniform Bx = {B0*1e3:.2f} mT, By = 0 at centre")
    print(f"solved I1..I6 (A) = [{', '.join(f'{x:8.1f}' for x in I_coil)}]")
    print(f"  max |I| = {np.max(np.abs(I_coil)):.1f} A  (norm. to {MAX_CURRENT:.0f} A)")

    # Quick validation: compute the actual field at the centre
    coils = PSXMCoils(currents=I_coil)
    Bx0, By0 = coils.B_field(0.0, 0.0)
    Bx1, By1 = coils.B_field(1.0, 0.0)
    print(f"central field: Bx(0,0) = {Bx0*1e3:.4f} mT,  By(0,0) = {By0*1e3:.4f} mT")
    print(f"off-axis   (1,0): Bx = {Bx1*1e6:.2f} uT,  By = {By1*1e6:.2f} uT\n")

    # Also compute LS shield currents at default params
    tpl_s = make_template(shield_n=200)
    I_shield = ls_shield_currents(tpl_s, I_coil)
    print(f"LS shield (shield_n=200, gap=5mm): peak |I| = {np.max(np.abs(I_shield)):.1f} A\n")
    return I_coil


# ============================================================================
# Part 2: Parameter scan
# ============================================================================
def part2_parameter_scan(I_coil):
    """Systematic scan over (shield_n, gap_mm, n_between) for the dipole.

    The question this answers is not "is the solver right" but "does the
    answer depend on how I sampled".  Redundancy = n_equations /
    n_unknowns; chi2_rel = rms residual field at the samples relative to
    the typical field there, which is small but non-zero precisely
    because the system is over-determined.
    """
    print("=" * 65)
    print("PART 2: SHIELDING PARAMETER SCAN (DIPOLE)")
    print("=" * 65)

    n_betweens = (1, 3, 6)
    rows = []
    header = (f"{'gap_mm':>7} {'n_between':>9} {'shield_n':>9} {'redund.':>8} "
              f"{'chi2_rel':>10} {'suppress':>10}")
    print(header)
    print("-" * len(header))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for nb in n_betweens:
        for gap in GAPS_MM:
            for sn in SHIELD_NS:
                tpl = make_template(shield_n=sn)
                solver = shield_zero_solver(tpl, gap_mm=gap, n_between=nb,
                                            outer_mm=gap / 2.0)
                KM = solver.coefficient_matrix() @ tpl.group_matrix()
                K6, Ksh = KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:]
                n_eq, n_unk = Ksh.shape
                X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
                I_s = (-X) @ I_coil

                # Achieved residual
                resid = Ksh @ I_s + K6 @ I_coil
                chi2 = float(np.sum(resid ** 2))
                B_scale = float(np.max(np.abs(K6 @ I_coil)))
                chi2_rel = float(np.sqrt(chi2 / n_eq) / B_scale) if B_scale > 0 else 0.0

                # Leakage suppression
                shielded = PSXMCoils(currents=I_coil, shield=True,
                                     shield_radius=tpl.shield_radius,
                                     shield_n=sn, shield_currents=I_s)
                unshielded = PSXMCoils(currents=I_coil)
                bu, bs = ring_meanB(unshielded, BENCH_R_MM), ring_meanB(shielded, BENCH_R_MM)
                supp = float(bu / bs) if bs > 0 else 1e6

                rows.append(dict(gap_mm=gap, n_between=nb, shield_n=sn,
                                 redundancy=n_eq / n_unk, chi2_rel=chi2_rel,
                                 suppression=supp))
                print(f"{gap:7.1f} {nb:9d} {sn:9d} {n_eq / n_unk:8.1f} "
                      f"{chi2_rel:10.2e} {supp:10.1f}")

    for sn in SHIELD_NS[-2:]:  # 200, 400
        rs = [r for r in rows if r["shield_n"] == sn and r["n_between"] == 3]
        if rs:
            ax1.semilogy([r["gap_mm"] for r in rs], [r["suppression"] for r in rs],
                         "o-", label=f"shield_n={sn}")
            ax2.semilogy([r["gap_mm"] for r in rs], [r["chi2_rel"] for r in rs],
                         "o-", label=f"shield_n={sn}")
    ax1.set_xlabel("sample gap (mm)")
    ax1.set_ylabel("suppression at {:.0f} mm".format(BENCH_R_MM))
    ax1.set_title("leakage suppression vs gap (n_between=3)")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    ax2.set_xlabel("sample gap (mm)")
    ax2.set_ylabel("rms residual B / typical B")
    ax2.set_title("achieved residual vs gap (n_between=3)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("figures/dipole_parameter_scan.png", dpi=150)
    print("\nsaved figures/dipole_parameter_scan.png\n")
    return


# ============================================================================
# Part 3: Leakage figure for the dipole
# ============================================================================
def part3_leakage_figure(I_coil):
    """Two-panel leakage figure for the dipole case with chosen params."""
    print("=" * 65)
    print("PART 3: DIPOLE LEAKAGE FIGURE")
    print("=" * 65)

    shield_n = 200
    gap_mm = 5.0
    tpl = make_template(shield_n=shield_n)
    I_shield = ls_shield_currents(tpl, I_coil, gap_mm=gap_mm)
    shielded, unshielded = build_pair(tpl, I_coil, I_shield)
    leakage_report(shielded, unshielded)

    fig, (axf, axl) = plt.subplots(1, 2, figsize=(13.5, 6))
    shielded.draw(axf, n_grid=400, extent=R_MAX_MM / shielded.shield_radius, legend=False)
    axf.set_title(f"dipole Bx = {B0*1e3:.1f} mT  (+-{R_MAX_MM:.0f} mm)")
    ins = axf.inset_axes([0.66, 0.66, 0.33, 0.33])
    shielded.draw(ins, n_grid=150, extent=40.0 / shielded.shield_radius, legend=False)
    ins.set_title("zoom 40 mm", fontsize=7)
    ins.set_xlabel(""); ins.set_ylabel("")
    ins.tick_params(labelsize=6)

    rho = np.geomspace(1.0, R_MAX_MM, 140)
    Bun = np.array([ring_meanB(unshielded, r) for r in rho])
    Bsh = np.array([ring_meanB(shielded, r) for r in rho])
    axl.loglog(rho / 1000, Bun * 1e3, label="no shield")
    axl.loglog(rho / 1000, Bsh * 1e3, label="with shield")
    for xr in (shielded.radius, shielded.shield_radius, *MARKS_MM):
        axl.axvline(xr / 1000, color="gray", ls="--", lw=0.7)
    axl.set_xlabel("radial distance rho (m)")
    axl.set_ylabel("ring-averaged |B| (mT)")
    axl.set_title("dipole leakage: |B| vs radius")
    axl.legend()
    axl.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/dipole_shield_leakage.png", dpi=170, bbox_inches="tight")
    print("saved figures/dipole_shield_leakage.png\n")


# ============================================================================
# main
# ============================================================================
def main():
    I_coil = part1_dipole_solve()
    part2_parameter_scan(I_coil)
    part3_leakage_figure(I_coil)

    print("All 3 parts complete.")
    print("  figures/dipole_parameter_scan.png   - parameter scan results")
    print("  figures/dipole_shield_leakage.png   - two-panel leakage figure")


if __name__ == "__main__":
    main()
