"""Induced shield current from ELECTROMAGNETIC INDUCTION (2-D thin-shell
eddy-current model).

Answers: given the main-coil current and real copper / pulse parameters,
what shield current does a real conductor actually INDUCE, and how well
does it screen? -- as opposed to solving for an arbitrary field-nulling
current (which is not guaranteed to be physical).

Method (2-D, axial currents, azimuthal Fourier modes m):
  1. coil exterior field  -> multipole moments  C_m = sum_j I_j z_j^m
  2. a perfect conductor at radius a carries the surface current that
     cancels every exterior multipole (B.n = 0 on the shell):
       K_ideal(t) = - sum_m Re[C_m e^{-i m t}] / (pi a^{m+1})     [A/m]
  3. a REAL thin shell (sigma, thickness d) responds mode-by-mode with a
     first-order L/R filter:
       f_m(w) = i w tau_m / (1 + i w tau_m),  tau_m = mu0 sigma d_eff a /(2 m)
     induced current  K = sum_m f_m * (ideal mode m);
     residual exterior field of mode m reduced by  1/(1 + i w tau_m).
  Fast-pulse / good-conductor limit (w tau_m >> 1): f_m -> 1, the shell
  induces ~the full ideal screening current -> strong shielding.

Parameters are literature / design values (cited inline); edit the globals
for the real PSXM numbers.

Run:  python induced_shield.py
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils

MU0 = 4e-7 * np.pi

# --- literature / design parameters (EDIT for the real PSXM) --------------
SIGMA_CU = 5.8e7      # S/m, copper conductivity (~IACS standard)
A_SHIELD = 27.5e-3    # m, shield radius (PSXM_sim model value)
D_SHIELD = 2.0e-3     # m, copper thickness (NOMINAL -- confirm from design)
I_COIL = 1000.0       # A, main-coil peak current
T_PULSE = 1.0e-6      # s, pulse width (mentor slide: ~1 kA, <= 1 us)
SHIELD_N = 100        # express K as an equivalent per-point current
M_MODES = 8           # azimuthal modes to keep


def main():
    omega = np.pi / T_PULSE                       # ~ angular freq of a half-sine of width T
    f0 = omega / (2 * np.pi)
    delta = np.sqrt(2.0 / (MU0 * SIGMA_CU * omega))   # skin depth
    d_eff = min(D_SHIELD, delta)                  # skin effect: current in ~delta layer if delta<d

    print(f"pulse : I_coil={I_COIL:.0f} A, width {T_PULSE*1e6:.2f} us -> f~{f0/1e6:.3f} MHz")
    print(f"copper: sigma={SIGMA_CU:.2e} S/m, skin depth={delta*1e6:.1f} um, "
          f"d={D_SHIELD*1e3:.2f} mm -> d_eff={d_eff*1e6:.1f} um")
    print(f"shield: radius a={A_SHIELD*1e3:.1f} mm\n")

    # --- step 1: coil multipole moments C_m = sum_j I_j z_j^m ------------
    coils = PSXMCoils(currents=[0, I_COIL, I_COIL, 0, -I_COIL, -I_COIL])  # quadrupole
    z = (np.asarray(coils.x) + 1j * np.asarray(coils.y)) * 1e-3           # leg positions, m
    Ileg = np.asarray(coils.I)
    C = np.array([np.sum(Ileg * z ** m) for m in range(M_MODES + 1)])

    # --- steps 2-3: per-mode time constant and L/R response -------------
    tau = np.zeros(M_MODES + 1)
    fmod = np.zeros(M_MODES + 1, dtype=complex)
    print(" m   |C_m|         w*tau_m      shielding 1/|1+iwt|   |f_m| (induced frac)")
    for m in range(1, M_MODES + 1):
        tau[m] = MU0 * SIGMA_CU * d_eff * A_SHIELD / (2 * m)
        wt = omega * tau[m]
        fmod[m] = 1j * wt / (1 + 1j * wt)
        print(f" {m}   {abs(C[m]):.3e}   {wt:10.1f}    {1.0/abs(1+1j*wt):.3e}          {abs(fmod[m]):.4f}")

    # --- reconstruct surface current K(theta) [A/m] --------------------
    th = np.linspace(0, 2 * np.pi, 361)
    K_ideal = np.zeros_like(th)
    K_real = np.zeros_like(th)
    for m in range(1, M_MODES + 1):
        base = np.real(C[m] * np.exp(-1j * m * th)) / (np.pi * A_SHIELD ** (m + 1))
        K_ideal += -base
        K_real += -np.real(fmod[m] * C[m] * np.exp(-1j * m * th)) / (np.pi * A_SHIELD ** (m + 1))

    dth = th[1] - th[0]
    I_oneway_ideal = 0.5 * np.sum(np.abs(K_ideal)) * A_SHIELD * dth
    I_oneway_real = 0.5 * np.sum(np.abs(K_real)) * A_SHIELD * dth
    seg = 2 * np.pi * A_SHIELD / SHIELD_N
    print(f"\nideal screening current : peak |K|={np.max(np.abs(K_ideal)):8.1f} A/m,  "
          f"total(one way)={I_oneway_ideal:7.1f} A,  per-point peak={np.max(np.abs(K_ideal))*seg:6.1f} A")
    print(f"induced (real shell)    : peak |K|={np.max(np.abs(K_real)):8.1f} A/m,  "
          f"total(one way)={I_oneway_real:7.1f} A,  per-point peak={np.max(np.abs(K_real))*seg:6.1f} A")
    print(f"induced / ideal (peak)  : {np.max(np.abs(K_real))/np.max(np.abs(K_ideal)):.3f}")
    print(f"coil current for reference: {I_COIL:.0f} A per leg")
    print(f"dominant quadrupole (m=2) shielding factor: {1.0/abs(1+1j*omega*tau[2]):.3e}")

    # --- plot ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(np.degrees(th), K_ideal, )
    #ax[0].plot(np.degrees(th), K_real, "--", label="induced (real shell)")
    ax[0].set_xlabel("θ (deg)")
    ax[0].set_ylabel("surface current K (A/m)")
    ax[0].set_title("induced shield surface current")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    mm = np.arange(1, M_MODES + 1)
    Sfac = np.array([1.0 / abs(1 + 1j * omega * tau[m]) for m in mm])
    ax[1].semilogy(mm, Sfac, "o-")
    ax[1].set_xlabel("azimuthal mode m")
    ax[1].set_ylabel("residual field factor 1/|1+iωτ|")
    ax[1].set_title("shielding factor per mode (lower = better)")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("induced_shield.png", dpi=150)
    print("\nsaved induced_shield.png")


if __name__ == "__main__":
    main()
