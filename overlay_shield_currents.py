"""overlay_shield_currents.py -- shield current vs angle: least-squares vs
physically-induced eddy-current model, overlaid on one plot.

ls_shield_currents() and induced_shield_currents() (both in shield_common)
each return one current value per shield discretization point, in Amps, so
the two are directly comparable with no rescaling.

Run:  python overlay_shield_currents.py   (saves figures/shield_current_ls_vs_induced.png)
"""

import numpy as np
import matplotlib.pyplot as plt

from shield_common import (make_template, solve_quad_coils, ls_shield_currents,
                           induced_shield_currents)


def main():
    tpl = make_template()
    I_coil = solve_quad_coils(tpl)

    I_ls = ls_shield_currents(tpl, I_coil)
    I_ind, info = induced_shield_currents(tpl, I_coil)

    theta = tpl.shield_angles   # degrees, one value per shield point

    print(f"LS shield current      : peak {np.max(np.abs(I_ls)):.1f} A")
    print(f"induced shield current  : peak {np.max(np.abs(I_ind)):.1f} A  "
          f"(m=2 shielding factor {info['S2']:.3e})")
    print(f"peak ratio (induced/LS) : {np.max(np.abs(I_ind)) / np.max(np.abs(I_ls)):.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta, I_ls, ".-", ms=3, lw=0.8, label="least-squares (field-nulling)")
    ax.plot(theta, I_ind, ".-", ms=3, lw=0.8, label="physically induced (eddy current)")
    ax.set_xlabel("shield angle θ (deg)")
    ax.set_ylabel("shield current per point (A)")
    ax.set_title("shield current vs angle: LS vs physically induced")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/shield_current_ls_vs_induced.png", dpi=170)
    print("saved figures/shield_current_ls_vs_induced.png")


if __name__ == "__main__":
    main()
