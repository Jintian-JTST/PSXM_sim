"""shield2D.py -- 2-D (infinite straight wire) PSXM leakage vs RADIUS,
using the interactive_shield response-matrix method.

The coil currents are solved for the centre quadrupole using the COIL DOF only
(and normalized to MAX_CURRENT on the coils). The shield currents are the
decoupled response  I_shield = S @ I_coil  with  S = -K_s^+ K_6  (least
squares nulling the field on/around the shield). This is NOT the joint
106-variable solve (which normalizes coil+shield together and gives a
broken, backwards result).

Leakage = ring-averaged |B| vs radial distance rho (2-D, no z).
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver

MAX_CURRENT = 1000.0        # A, coil currents normalized so max|I_coil| = this
SHIELD_N = 100
G = 1e-3                    # T/mm, target central quadrupole gradient
SAMPLE_GAP_MM = 2.0         # mm: radial offset between the shield line currents
                            #     and the B=0 sample points (see note below)
DIST_MARKS_MM = (100.0, 419.0)   # mm: 0.1 m benchmark, ~0.42 m nearest beam
R_MAX_MM = 420.0
SHIELD_RADIUS = 250        # mm, shield radius (PSXM_sim model value)

def ring_meanB(coils, rho, n=96):
    vals = []
    for a in np.linspace(0.017, 2 * np.pi + 0.017, n, endpoint=False):
        try:
            vals.append(coils.B_magnitude(rho * np.cos(a), rho * np.sin(a)))
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else np.nan


def solve_currents(shield_n=SHIELD_N, n_between=3, outer=5.0):
    """interactive_shield method: coil currents for the centre quadrupole
    (coil DOF only, normalized on the coils), then shield response
    I_shield = S @ I_coil."""
    tpl = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=shield_n,shield_radius=SHIELD_RADIUS)
    M = tpl.group_matrix()

    # --- coil currents for the centre quadrupole (coil DOF only) ---
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    solver_c = CurrentSolver.from_current_source(tpl)
    for a in ang:
        x, y = np.cos(a), np.sin(a)                            # 1 mm ring
        solver_c.add_sample_point(x, y, Bx=G * y, By=G * x)    # quadrupole target
    Kc = solver_c.coefficient_matrix() @ M
    Kcoil = Kc[:, :PSXMCoils.N_COILS]
    Btarget = solver_c.target_field()
    I_coil, *_ = np.linalg.lstsq(Kcoil, Btarget, rcond=None)
    I_coil = CurrentSolver.normalize_currents(I_coil, MAX_CURRENT)

    # --- shield response S: null the field just OFF the shield ---
    # The B=0 sample points are offset radially by SAMPLE_GAP_MM from the
    # discrete shield line currents. Sampling right on the shield puts them
    # ~0.9 mm from a current (in the azimuthal gaps), inside the singular
    # 1/r near-field, which forces the least squares to keep the shield
    # currents tiny and kills the shielding.
    solver_s = CurrentSolver.from_current_source(tpl)
    gap = 360.0 / shield_n
    for radius in (tpl.shield_radius + SAMPLE_GAP_MM, tpl.shield_radius + outer):
        for base in tpl.shield_angles:
            for j in range(1, n_between + 1):
                a = np.radians(base + j * gap / (n_between + 1))
                solver_s.add_sample_point(radius * np.cos(a), radius * np.sin(a), 0.0, 0.0)
    KM = solver_s.coefficient_matrix() @ M
    K6 = KM[:, :PSXMCoils.N_COILS]
    Ksh = KM[:, PSXMCoils.N_COILS:]
    X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
    S = -X
    I_shield = S @ I_coil
    return tpl, I_coil, I_shield


def main():
    tpl, I_coil, I_shield = solve_currents()
    kw = dict(radius=tpl.radius, coil_length=tpl.coil_length)
    shielded = PSXMCoils(currents=I_coil, shield=True, shield_radius=tpl.shield_radius,
                         shield_n=SHIELD_N, shield_currents=I_shield, **kw)
    unshielded = PSXMCoils(currents=I_coil, **kw)

    print(f"coil current: max {np.max(np.abs(I_coil)):.1f} A;  "
          f"shield current: peak {np.max(np.abs(I_shield)):.1f} A, "
          f"range [{I_shield.min():.1f}, {I_shield.max():.1f}] A")
    for d in DIST_MARKS_MM:
        bu, bs = ring_meanB(unshielded, d), ring_meanB(shielded, d)
        print(f"radial leakage at {d/1000:.3f} m:  no-shield {bu*1e3:9.4f} mT   "
              f"shielded {bs*1e6:8.3f} µT   (x{bu/bs:.1f})")

    rho = np.geomspace(1.0, R_MAX_MM, 200)
    Bun = np.array([ring_meanB(unshielded, r) for r in rho])
    Bsh = np.array([ring_meanB(shielded, r) for r in rho])

    plt.figure(figsize=(7.5, 5.5))
    plt.loglog(rho / 1000, Bun * 1e3, label="no shield")
    plt.loglog(rho / 1000, Bsh * 1e3, label="coils + shield")
    for d in (tpl.radius, tpl.shield_radius, *DIST_MARKS_MM):
        plt.axvline(d / 1000, color="gray", ls="--", lw=0.7)
    plt.xlabel("radial distance ρ (m)")
    plt.ylabel("ring-averaged |B| (mT)")
    plt.title("PSXM leakage vs RADIUS (2-D infinite wire, response-matrix shield)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("shield2D.png", dpi=150)
    print("saved shield2D.png")


if __name__ == "__main__":
    main()
