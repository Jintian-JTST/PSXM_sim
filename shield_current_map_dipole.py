"""shield_current_map_dipole.py -- dipole version of the shield-current
field-line map.

The quadrupole counterpart (figures/shield_current_map.png) is drawn inside
report_figures.fig_exterior_profiles() only in the quadrupole branch; this
script reproduces the same panel for the dipole design, with the same solve,
the same shield radius and the same field-line rendering.

Run:  python shield_current_map_dipole.py
      (saves figures/shield_current_map_dipole.png and the report copy)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from shield_common import solve_dipole_coils, ls_shield_currents, build_pair
from node5_common import save_fig

SHIELD_RADIUS = 27.5
SHIELD_N = 200
MAX_CURRENT = 1000.0

tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                shield_radius=SHIELD_RADIUS, shield_n=SHIELD_N)
Ic = solve_dipole_coils(tpl)          # dipole: uniform Bx = B0 at the centre
Ish = ls_shield_currents(tpl, Ic)     # least-squares response shield currents
sh, un = build_pair(tpl, Ic, Ish)

print(f"I_coil = [{', '.join(f'{x:8.1f}' for x in Ic)}] A")
print(f"peak |I_shield| = {np.max(np.abs(Ish)):.1f} A  "
      f"(max I_coil = {np.max(np.abs(Ic)):.1f} A)")

figm, axm = plt.subplots(figsize=(6.2, 6.0))
sh.draw(axm, n_grid=340, extent=45.0 / SHIELD_RADIUS, legend=False)
axm.set_title("")
axm.set_xlabel("$x$ (mm)"); axm.set_ylabel("$y$ (mm)")
figm.tight_layout()
save_fig(figm, "shield_current_map_dipole.png")
