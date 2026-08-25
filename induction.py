"""induction.py -- physical induced-eddy response of the shield can.

The report (section 7.1) lists building this circuit-level induction
model as the next step; it is NOT part of the current report's numbers,
whose shield is the ideal flux-excluding sheet of ``shield.py``.  This
module is kept for that future work and has no caller in the current
pipeline.
"""

import numpy as np

from psxm_coils import PSXMCoils

MU0 = 4e-7 * np.pi

SIGMA_CU = 5.8e7        # S/m, copper conductivity
D_SHIELD = 5.0e-3       # m, copper wall thickness (report section 7.1 baseline)
T_PULSE = 1.0e-6        # s, pulse width
M_MODES = 10            # multipole orders summed


def induced_shield_currents(tpl, I_coil, t_pulse=T_PULSE, sigma=SIGMA_CU, d=D_SHIELD):
    """Physical induced eddy currents: multipole moments x thin-shell L/R
    response, sampled onto the shield points. Returns (I_shield, info dict)."""
    omega = np.pi / t_pulse
    delta = np.sqrt(2.0 / (MU0 * sigma * omega))
    d_eff = min(d, delta)
    a_m = tpl.shield_radius * 1e-3

    legs = PSXMCoils(currents=I_coil)
    z = (np.asarray(legs.x) + 1j * np.asarray(legs.y)) * 1e-3
    Ileg = np.asarray(legs.I)
    th = np.radians(tpl.shield_angles)
    K = np.zeros_like(th)
    S2 = None
    for m in range(1, M_MODES + 1):
        C = np.sum(Ileg * z ** m)
        tau = MU0 * sigma * d_eff * a_m / (2 * m)
        f = 1j * omega * tau / (1 + 1j * omega * tau)
        K += -np.real(f * C * np.exp(-1j * m * th)) / (np.pi * a_m ** (m + 1))
        if m == 2:
            S2 = 1.0 / abs(1 + 1j * omega * tau)
    seg = 2 * np.pi * a_m / tpl.shield_n
    return K * seg, {"delta": delta, "d_eff": d_eff, "S2": S2}
