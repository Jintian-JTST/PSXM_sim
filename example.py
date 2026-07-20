"""PSXM 2-D leakage using the PHYSICAL induced shield current.

The coils are a fixed quadrupole. The shield current is NOT solved by
least squares (that over-fits with unphysical currents -> fake 1e-9
leakage); it is the current a real copper shell actually INDUCES:

  1. coil exterior multipole moments   C_m = sum_j I_j z_j^m
  2. a real thin shell reduces mode m by the L/R factor
       f_m = i w tau_m / (1 + i w tau_m),   tau_m = mu0 sigma d_eff a /(2 m)
  3. induced surface current
       K(theta) = - sum_m Re[f_m C_m e^{-i m theta}] / (pi a^{m+1})   [A/m]
     sampled onto shield_n line currents (I_k = K(theta_k) * 2*pi*a/shield_n).

Leakage is then the honest physical result: the quadrupole exterior is
reduced by ~1/(1+i w tau_2) (a few x 10^-3), NOT 1e-9.
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils

MU0 = 4e-7 * np.pi

# --- literature / design parameters (edit for the real PSXM) --------------
SIGMA_CU = 5.8e7       # S/m, copper conductivity
A_SHIELD_MM = 27.5     # mm, shield radius
D_SHIELD = 2.0e-3      # m, copper thickness (nominal)
I_COIL = 1000.0        # A, main-coil peak current
T_PULSE = 1.0e-6       # s, pulse width (mentor: ~1 kA, <= 1 us)
SHIELD_N = 120         # shield discretization points
M_MODES = 10           # azimuthal modes kept
R_BEAM = 5.0           # mm, reference beam radius


def ring_meanB(coils, rho, n=96):
    """Mean |B| (T) on a circle of radius rho (mm), skipping conductor hits."""
    vals = []
    for a in np.linspace(0.017, 2 * np.pi + 0.017, n, endpoint=False):
        try:
            vals.append(coils.B_magnitude(rho * np.cos(a), rho * np.sin(a)))
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else np.nan


def induced_shield_currents(coil_currents, a_m, shield_angles_deg):
    """Physical induced shield line currents (A) at the shield points, plus
    the skin depth and dominant-mode shielding factor."""
    omega = np.pi / T_PULSE
    delta = np.sqrt(2.0 / (MU0 * SIGMA_CU * omega))
    d_eff = min(D_SHIELD, delta)

    legs = PSXMCoils(currents=coil_currents)                       # 12 coil legs
    z = (np.asarray(legs.x) + 1j * np.asarray(legs.y)) * 1e-3      # m
    Ileg = np.asarray(legs.I)
    C = [np.sum(Ileg * z ** m) for m in range(M_MODES + 1)]        # multipole moments

    th = np.radians(shield_angles_deg)
    K = np.zeros_like(th)
    S2 = None
    for m in range(1, M_MODES + 1):
        tau = MU0 * SIGMA_CU * d_eff * a_m / (2 * m)
        f = 1j * omega * tau / (1 + 1j * omega * tau)
        K += -np.real(f * C[m] * np.exp(-1j * m * th)) / (np.pi * a_m ** (m + 1))
        if m == 2:
            S2 = 1.0 / abs(1 + 1j * omega * tau)
    seg = 2 * np.pi * a_m / len(th)
    return K * seg, delta, d_eff, S2


def main():
    coil_currents = np.array([0, I_COIL, I_COIL, 0, -I_COIL, -I_COIL], float)  # quadrupole
    a_m = A_SHIELD_MM * 1e-3
    tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                    shield_radius=A_SHIELD_MM, shield_n=SHIELD_N)
    I_shield, delta, d_eff, S2 = induced_shield_currents(coil_currents, a_m, tpl.shield_angles)

    print(f"pulse width {T_PULSE*1e6:.2f} us -> skin depth {delta*1e6:.1f} um, d_eff {d_eff*1e6:.1f} um")
    print(f"coil current {I_COIL:.0f} A/leg;  induced shield: peak {np.max(np.abs(I_shield)):.1f} A, "
          f"total(one way) {0.5*np.sum(np.abs(I_shield)):.1f} A")
    print(f"dominant quadrupole (m=2) shielding factor: {S2:.3e}\n")

    kw = dict(radius=22.5, coil_length=20.0)
    shielded = PSXMCoils(currents=coil_currents, shield=True, shield_radius=A_SHIELD_MM,
                         shield_n=SHIELD_N, shield_currents=I_shield, **kw)
    unshielded = PSXMCoils(currents=coil_currents, **kw)

    # leakage far enough out that the shield-point ripple has decayed
    Bbeam = ring_meanB(unshielded, R_BEAM)
    print(f"beam |B| (r={R_BEAM:.0f} mm): {Bbeam*1e3:.3f} mT")
    for r_far in (2 * A_SHIELD_MM, 419.0):
        Bo_un = ring_meanB(unshielded, r_far)
        Bo_sh = ring_meanB(shielded, r_far)
        print(f"exterior at r={r_far:6.1f} mm:  no-shield {Bo_un*1e6:10.3f} µT   "
              f"shielded {Bo_sh*1e6:10.3f} µT   suppression {Bo_un/Bo_sh:.1f}x  "
              f"({Bo_sh/Bo_un*100:.2f} % leaks)")

    # --- plots ---
    fig, (axf, axl) = plt.subplots(1, 2, figsize=(13, 6))
    shielded.draw(axf, extent=2, legend=False)
    axf.set_title("field lines + induced shield")

    rho = np.geomspace(1.0, 420.0, 140)      # out to 0.42 m (nearest-beam distance)
    Bsh = np.array([ring_meanB(shielded, r) for r in rho])
    Bun = np.array([ring_meanB(unshielded, r) for r in rho])
    axl.loglog(rho, Bun * 1e3, label="no shield")
    axl.loglog(rho, Bsh * 1e3, label="with induced shield")
    for xr, lbl in [(22.5, "coils"), (A_SHIELD_MM, "shield"), (R_BEAM, "beam r"),
                    (419.0, "0.419 m")]:
        axl.axvline(xr, color="gray", ls="--", lw=0.7)
        axl.text(xr, axl.get_ylim()[1], lbl, fontsize=7, rotation=90, va="top")
    axl.set_xlabel("radius ρ (mm)")
    axl.set_ylabel("ring-averaged |B| (mT)")
    axl.set_title("leakage: |B| vs radius (physical shield)")
    axl.legend()
    axl.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("example.png", dpi=200, bbox_inches="tight")
    print("saved example.png")


if __name__ == "__main__":
    main()
