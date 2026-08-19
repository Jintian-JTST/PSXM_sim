"""dipole_bare_field.py -- the PSXM dipole field WITHOUT the shield.

Companion to dipole_shield_optimization.py / two_panel(): same solve, same
measurement, but only the bare six-coil ring.  The purpose is a clean
"no shield" picture: a single field-line panel over -50 .. +50 mm with the
solved dipole currents, plus the achieved central field at 1000 A and the
uniformity vs radius printed to the console.

Calculation mode (identical to the report's Sec. 3.1 "Dipole mode"):
  * target: uniform Bx = B0, By = 0 on a 1 mm sampling ring (12 points);
  * coefficient matrix K from the infinite-straight-wire model (T/A),
    reduced by the coil group matrix to the 6 physical coil DOFs;
  * currents = weighted least-squares solve of K I = B_target,
    renormalised to max|I| = 1000 A (MAX_CURRENT);
  * achieved field measured with multipoles() on the same 1 mm ring.

Run:  python dipole_bare_field.py
      (saves figures/dipole_bare_field.png and
             ../PSXM_design_report/figures/dipole_bare_field.png)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from shield_common import (make_template, solve_dipole_coils, ring_meanB,
                           MARKS_MM)
from node5_common import multipoles, uniformity, save_fig

B0 = 1e-3                    # T, central dipole target (Bx = B0, By = 0)
MAX_CURRENT = 1000.0         # A, hardware current budget per coil
COIL_R = 22.5                # mm, coil-ring radius (also the marker)
BENCH_R_MM = 419.0           # mm, nearest-beam benchmark distance


def main():
    tpl = make_template(shield_n=200)          # shield geometry unused here
    I_coil = solve_dipole_coils(tpl, B0=B0)    # bare-ring dipole currents

    # --- achieved-field numbers -------------------------------------------
    bare = PSXMCoils(currents=I_coil)          # no shield
    m = multipoles(bare, r0=1.0, n=64)
    uni = uniformity(bare, radii=(5.0, 10.0))
    print("=" * 60)
    print("BARE-RING DIPOLE  (no shield)")
    print("=" * 60)
    print(f"solved I1..I6 (A) = [{', '.join(f'{x:8.1f}' for x in I_coil)}]")
    print(f"max |I| = {np.max(np.abs(I_coil)):.1f} A  (normalised to {MAX_CURRENT:.0f} A)")
    print(f"central dipole: B0 = {m['B0mag']*1e3:.3f} mT along "
          f"{m['B0deg']:.1f} deg")
    print(f"fit residual on 1 mm ring: rms = {m['rms']*1e9:.2f} nT "
          f"(purity {100*m['purity']:.3f} %)")
    print(f"uniformity: {100*uni[5.0]:.2f} % at r = 5 mm, "
          f"{100*uni[10.0]:.2f} % at r = 10 mm")
    for d in MARKS_MM:
        b = ring_meanB(bare, d)
        print(f"ring-averaged |B| at {d/1000:.3f} m: {b*1e6:10.4f} uT")

    # --- figure ------------------------------------------------------------
    # single panel: field lines of the bare dipole over -50 .. +50 mm.
    # draw()'s extent is relative to the coil-ring radius (22.5 mm), so
    # extent = 50 / 22.5 gives exactly +/-50 mm.
    fig, axf = plt.subplots(figsize=(7.5, 6.5))
    bare.draw(axf, n_grid=400, extent=50.0 / bare.radius, legend=True)
    axf.set_title(f"bare dipole (no shield)  Bx = {B0*1e3:.1f} mT target  "
                  f"(±50 mm)")

    fig.tight_layout()
    save_fig(fig, "dipole_bare_field.png", dpi=170)
    print("\nsaved figures/dipole_bare_field.png  and report copy")


if __name__ == "__main__":
    main()
