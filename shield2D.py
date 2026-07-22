"""shield2D.py -- LEAST-SQUARES shield, example-style two-panel figure.

Coil currents solved for the centre quadrupole; shield currents from the
least-squares response I_s = S @ I_coil with the B=0 samples offset
radially by SAMPLE_GAP_MM from the shield (see shield_common). The left
panel shows the field lines with the red shield-current arrows (zoom
inset); the right panel shows |B| vs radius out to 0.42 m.

Run:  python shield2D.py
"""

import numpy as np

from shield_common import (make_template, solve_quad_coils, ls_shield_currents,
                           build_pair, leakage_report, two_panel)


def main():
    tpl = make_template()
    I_coil = solve_quad_coils(tpl)
    I_shield = ls_shield_currents(tpl, I_coil)

    print(f"coil current: max {np.max(np.abs(I_coil)):.1f} A;  "
          f"LS shield current: peak {np.max(np.abs(I_shield)):.1f} A, "
          f"range [{I_shield.min():.1f}, {I_shield.max():.1f}] A")

    shielded, unshielded = build_pair(tpl, I_coil, I_shield)
    leakage_report(shielded, unshielded)
    two_panel(shielded, unshielded,
              "least-squares shield", "shield2D.png")


if __name__ == "__main__":
    main()
