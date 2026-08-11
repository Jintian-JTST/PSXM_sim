"""induced_shield.py -- EXPLORATORY: physical induced eddy-current shield.

This is not part of the design pipeline and is not used in the technical
note.  The note's shield model is the weighted least-squares inverse
solve of current_solver.py / shield_common.ls_shield_currents(); this
file is a separate, dynamic model kept for future work.

It prints the per-mode shielding-factor table for a thin conducting shell
driven by a pulsed main coil (multipole moments x thin-shell L/R
response, implemented in shield_common.induced_shield_currents).

WARNING: the material and pulse parameters in shield_common (copper,
2 mm wall, 1 us pulse) are placeholders and are NOT confirmed against the
PSXM specification.  The published ASSM design for the same position uses
a 0.29 ms pulse at 25 A through a 1.5 mm SS304 duct, which is a very
different regime.  Nothing quantitative from this file should be quoted
until the real parameters are obtained.

Run:  python induced_shield.py
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from shield_common import (MU0, induced_shield_currents,
                           SIGMA_CU, D_SHIELD, T_PULSE, M_MODES)

A_SHIELD = 27.5e-3    # m, shield radius (PSXM_sim model value)
I_COIL = 1000.0       # A, main-coil peak current
SHIELD_N = 100        # express K as an equivalent per-point current


def main():
    coil_currents = np.array([0, I_COIL, I_COIL, 0, -I_COIL, -I_COIL], float)
    tpl = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=SHIELD_N,
                    shield_radius=A_SHIELD * 1e3)

    I_shield, info = induced_shield_currents(tpl, coil_currents, t_pulse=T_PULSE,
                                             sigma=SIGMA_CU, d=D_SHIELD)
    omega = np.pi / T_PULSE

    print("PLACEHOLDER PARAMETERS -- see the warning in the module docstring\n")
    print(f"pulse : I_coil={I_COIL:.0f} A, width {T_PULSE*1e6:.2f} us -> "
          f"f~{omega/(2*np.pi)/1e6:.3f} MHz")
    print(f"copper: sigma={SIGMA_CU:.2e} S/m, skin depth={info['delta']*1e6:.1f} um, "
          f"d={D_SHIELD*1e3:.2f} mm -> d_eff={info['d_eff']*1e6:.1f} um")
    print(f"shield: radius a={A_SHIELD*1e3:.1f} mm\n")
    print(f"dominant quadrupole (m=2) shielding factor: {info['S2']:.3e}")
    print(f"induced (real shell) per-point peak current: "
          f"{np.max(np.abs(I_shield)):.1f} A")

    # Per-mode shielding factor 1/|1+i w tau|, with tau_m ~ 1/m.  Note the
    # direction: tau falls with m, so w*tau falls with m, so the factor
    # RISES with m -- higher multipoles are shielded WORSE by a real shell,
    # not better.  (A perfect conductor would shield all modes equally.)
    mm = np.arange(1, M_MODES + 1)
    tau = MU0 * SIGMA_CU * info["d_eff"] * A_SHIELD / (2 * mm)
    Sfac = 1.0 / np.abs(1 + 1j * omega * tau)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.semilogy(mm, Sfac, "o-")
    ax.set_xlabel("azimuthal mode m")
    ax.set_ylabel("residual field factor $1/|1+i\\omega\\tau_m|$")
    ax.set_title("shielding factor per mode (lower = better;\n"
                 "higher modes are shielded worse, since $\\tau_m \\propto 1/m$)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("figures/induced_shielding_factor.png", dpi=150)
    print("\nsaved figures/induced_shielding_factor.png")


if __name__ == "__main__":
    main()
