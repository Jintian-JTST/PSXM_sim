"""optimize_shield_radius.py -- find the best shield radius for the PSXM.

Design question: does enlarging the shield can improve shielding?  There is
a genuine trade-off:

  * a shield ring closer to the coil ring cancels the leakage better AND
    absorbs more of the central field (so the coils must push more current
    to hold the same central field);
  * a shield ring further away barely perturbs the centre (less current)
    but lets more leakage escape at intermediate radius.

This script scans shield_radius at a FIXED central field target and
reports, for each radius: the required coil current, the required shield
current, and the leakage at 100 mm and 419 mm, then flags which radii are
feasible under a given coil-current budget.

Two field types are supported:

  * ``quadrupole`` -- uniform central gradient Bx = G*y, By = G*x.
    The central target is the gradient G (T/mm).
  * ``dipole``     -- uniform central field Bx = B0, By = 0.
    The central target is the field B0 (T).

Units: positions in mm, current in A, field in T (ring_meanB returns T).

Run:  python optimize_shield_radius.py quadrupole
      python optimize_shield_radius.py dipole
      (prints the table and saves figures/shield_radius_optimization_<field>.png)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from shield_common import (solve_quad_coils, solve_dipole_coils,
                           ls_shield_currents, build_pair, ring_meanB)

# --- design targets -------------------------------------------------------
G_TARGET = 1.0e-3          # T/mm, central quadrupole gradient target
B0_TARGET = 1.0e-3         # T, central dipole field target
MAX_CURRENT = 1000.0       # A, coil current budget
SHIELD_N = 200
R_SCAN = np.arange(25.0, 80.0, 1.0)     # mm


def design(R, field, target):
    """Return (I_coil_max, I_shield_max, leak@100mm uT, leak@419mm uT, suppr)
    for a shield of radius R (mm) holding the target central field.

    The solve is done once at the 1000 A-normalized operating point and the
    whole solution is scaled to the target (the field/current system is
    linear, so the relative current distribution is preserved).
    """
    tpl = PSXMCoils(currents=np.zeros(6), shield=True, shield_radius=R, shield_n=SHIELD_N)
    Ic = solve_quad_coils(tpl) if field == 'quadrupole' else solve_dipole_coils(tpl)
    Ish = ls_shield_currents(tpl, Ic)
    sh, un = build_pair(tpl, Ic, Ish)

    if field == 'quadrupole':
        g = ring_meanB(sh, 1.0) / 1.0           # achieved gradient T/mm at 1000 A
        f = target / g                          # current scale to hit the target gradient
    else:
        b = ring_meanB(sh, 1.0)                 # achieved |B| (T) at 1 mm, ~B0 at 1000 A
        f = target / b
    I_coil = f * np.max(np.abs(Ic))
    I_sh = f * np.max(np.abs(Ish))
    l100 = f * ring_meanB(sh, 100.0) * 1e6      # uT
    l419 = f * ring_meanB(sh, 419.0) * 1e6      # uT
    bu419 = f * ring_meanB(un, 419.0) * 1e6     # uT, unshielded reference at same scale
    suppr = bu419 / l419 if l419 > 0 else np.inf
    return I_coil, I_sh, l100, l419, suppr, f


def main():
    field = sys.argv[1] if len(sys.argv) > 1 else 'quadrupole'
    if field not in ('quadrupole', 'dipole'):
        sys.exit(f"unknown field type '{field}'; use 'quadrupole' or 'dipole'")
    target = G_TARGET if field == 'quadrupole' else B0_TARGET
    target_label = f"{target*1e3:.3f} mT/mm" if field == 'quadrupole' else f"{target*1e3:.3f} mT"

    rows = []
    for R in R_SCAN:
        Ic, Is, l100, l419, suppr, f = design(R, field, target)
        feasible = Ic <= MAX_CURRENT
        rows.append((R, Ic, Is, l100, l419, suppr, feasible))

    # --- print -------------------------------------------------------------
    print(f"field = {field}, target {target_label}, coil budget {MAX_CURRENT:.0f} A\n")
    print(f"{'R_sh':>5} | {'I_coil':>8} {'I_shield':>9} | {'leak@100 uT':>12} "
          f"{'leak@419 uT':>12} {'suppr@419':>10} | {'feas':>4}")
    for R, Ic, Is, l100, l419, suppr, feas in rows:
        print(f"{R:5.1f} | {Ic:8.0f} {Is:9.1f} | {l100:12.4f} {l419:12.5f} "
              f"{suppr:10.0f} | {'yes' if feas else 'no '}")

    feas_rows = [r for r in rows if r[-1]]
    if feas_rows:
        # pick the radius with the lowest leakage among feasible radii
        best = min(feas_rows, key=lambda r: r[3])
        print(f"\nbest feasible: R_sh = {best[0]:.1f} mm  "
              f"(I_coil={best[1]:.0f} A, leak@419 = {best[3]:.5f} uT, "
              f"suppr = {best[4]:.0f}x)")
    else:
        print("\nno feasible radius under the current budget; relax MAX_CURRENT or target")

    # --- figure -------------------------------------------------------------
    Rs = np.array([r[0] for r in rows])
    Ic = np.array([r[1] for r in rows])
    l100 = np.array([r[3] for r in rows])
    feasc = np.array([r[-1] for r in rows])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(Rs, Ic / 1e3, 'o-', ms=3)
    ax[0].axhline(MAX_CURRENT / 1e3, color='r', ls='--', label=f'budget {MAX_CURRENT:.0f} A')
    ax[0].set_xlabel('shield radius (mm)'); ax[0].set_ylabel('required I_coil (kA)')
    ax[0].set_title('current needed for fixed target'); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(Rs, l100, 'o-', ms=3, color='tab:orange')
    ax[1].set_xlabel('shield radius (mm)'); ax[1].set_ylabel('leak@100mm (uT)')
    ax[1].set_title('leakage at 100 mm'); ax[1].set_yscale('log'); ax[1].grid(alpha=0.3)

    ax[2].plot(Rs, feasc, 'o-', ms=3, color='tab:green')
    ax[2].set_xlabel('shield radius (mm)'); ax[2].set_ylabel('feasible under budget')
    ax[2].set_ylim(-0.1, 1.1); ax[2].set_yticks([0, 1]); ax[2].grid(alpha=0.3)

    if feas_rows:
        br = best[0]
        for a in ax:
            a.axvline(br, color='tab:red', ls=':', lw=1)
        ax[0].annotate(f'best R={br:.0f} mm', xy=(br, best[1] / 1e3),
                       xytext=(br + 1, best[1] / 1e3 * 0.9), fontsize=8)
    fig.suptitle(f'{field}: shield-radius scan at fixed target {target_label}')
    fig.tight_layout()
    out = f'figures/shield_radius_optimization_{field}.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
