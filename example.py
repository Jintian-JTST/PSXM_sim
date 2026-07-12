"""Inverse-design demo: solve the 6 main-coil currents and 100 shield
currents that give a quadrupole at the centre and low leakage outside.

Shield boundary condition (toggle USE_BN_ON_SURFACE):
  - True  : on the shield ring, require only the NORMAL field to vanish
            (B_n = Bx*cosθ + By*sinθ = 0), the physical perfect-conductor
            boundary condition (tangential B is supported by the surface
            current and is generally nonzero).
  - False : on the shield ring, require the full field to vanish
            (Bx, By) = (0, 0)  -- the original, over-constrained version.
The OUTER ring (shield_radius + 5 mm) always requires the full field to
vanish: that ring is the actual leakage objective, not a conductor
surface.

Because CurrentSolver.add_sample_point can only target Bx and By
separately (not a combination like B_n), the weighted least-squares
system is assembled by hand here, reusing the same infinite-straight-wire
coefficients as coils.py / current_solver.py.
"""

import numpy as np
import matplotlib.pyplot as plt

from coils import MU0
from PSXM_coils import PSXMCoils

# --- configuration ---------------------------------------------------------
USE_BN_ON_SURFACE = True   # True: B_n = 0 on the shield ring; False: full (Bx,By)=0
G = 1e-3                    # target quadrupole gradient, T/mm
N_CENTER = 12             # center sample points on a 1 mm circle
SHIELD_N = 100           # discrete shield current points
N_BETWEEN = 3            # zero-field samples between adjacent shield points
OUTER_OFFSET = 5.0       # outer leakage-check ring, mm beyond the shield
SHIELD_WEIGHT_SCALE = 1.0  # down-weights the shield groups vs the center
MAX_CURRENT = 1000.0     # display currents are rescaled so max|I| = this (A)


def main():
    # Geometry only: all 6 coil + 100 shield currents are unknowns.
    source = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=SHIELD_N)
    M = source.group_matrix()          # per-point currents = M @ free vars

    def field_rows(x, y):
        """Coefficient rows (Kx, Ky) [T/A], each length n_free, giving the
        field at (x, y) [mm] as (Bx, By) = (Kx @ I_free, Ky @ I_free).
        Same infinite-straight-wire model as coils.B_field."""
        dx = (x - source.x) * 1e-3     # mm -> m
        dy = (y - source.y) * 1e-3
        rho2 = np.maximum(dx ** 2 + dy ** 2, 1e-18)
        Kx = (-MU0 * dy / (2 * np.pi * rho2)) @ M
        Ky = (MU0 * dx / (2 * np.pi * rho2)) @ M
        return Kx, Ky

    rows, targets, weights = [], [], []
    center_pts, surface_pts, outside_pts = [], [], []

    # --- target 1: quadrupole (Bx=G*y, By=G*x) on a 1 mm circle -----------
    cw = 1.0 / N_CENTER
    for ang in np.linspace(0, 2 * np.pi, N_CENTER, endpoint=False):
        x, y = np.cos(ang), np.sin(ang)
        Kx, Ky = field_rows(x, y)
        rows.append(Kx); targets.append(G * y); weights.append(cw)
        rows.append(Ky); targets.append(G * x); weights.append(cw)
        center_pts.append((x, y))

    n_shield_samples = SHIELD_N * N_BETWEEN
    sw = SHIELD_WEIGHT_SCALE / n_shield_samples
    gap = 360.0 / SHIELD_N

    # --- target 2: shield ring -------------------------------------------
    # B_n = 0 (perfect-conductor surface) or full (Bx,By)=0, per the toggle.
    for base in source.shield_angles:
        for j in range(1, N_BETWEEN + 1):
            a = np.radians(base + j * gap / (N_BETWEEN + 1))
            x, y = source.shield_radius * np.cos(a), source.shield_radius * np.sin(a)
            Kx, Ky = field_rows(x, y)
            if USE_BN_ON_SURFACE:
                nx, ny = np.cos(a), np.sin(a)           # outward radial normal
                rows.append(nx * Kx + ny * Ky); targets.append(0.0); weights.append(sw)
            else:
                rows.append(Kx); targets.append(0.0); weights.append(sw)
                rows.append(Ky); targets.append(0.0); weights.append(sw)
            surface_pts.append((x, y))

    # --- target 3: outer ring, full field = 0 (leakage objective) ---------
    r_out = source.shield_radius + OUTER_OFFSET
    for base in source.shield_angles:
        for j in range(1, N_BETWEEN + 1):
            a = np.radians(base + j * gap / (N_BETWEEN + 1))
            x, y = r_out * np.cos(a), r_out * np.sin(a)
            Kx, Ky = field_rows(x, y)
            rows.append(Kx); targets.append(0.0); weights.append(sw)
            rows.append(Ky); targets.append(0.0); weights.append(sw)
            outside_pts.append((x, y))

    # --- weighted least squares -------------------------------------------
    A = np.asarray(rows)
    b = np.asarray(targets)
    sqrt_w = np.sqrt(np.asarray(weights))
    I_free, *_ = np.linalg.lstsq(A * sqrt_w[:, None], b * sqrt_w, rcond=None)

    # --- residuals --------------------------------------------------------
    def field_at(x, y):
        Kx, Ky = field_rows(x, y)
        return Kx @ I_free, Ky @ I_free

    center_res = 0.0
    for x, y in center_pts:
        Bx, By = field_at(x, y)
        center_res = max(center_res, abs(Bx - G * y), abs(By - G * x))

    surface_bn, surface_full = 0.0, 0.0
    for x, y in surface_pts:
        Bx, By = field_at(x, y)
        a = np.arctan2(y, x)
        surface_bn = max(surface_bn, abs(np.cos(a) * Bx + np.sin(a) * By))
        surface_full = max(surface_full, np.hypot(Bx, By))

    outside_res = 0.0
    for x, y in outside_pts:
        Bx, By = field_at(x, y)
        outside_res = max(outside_res, np.hypot(Bx, By))

    # --- normalize + split ------------------------------------------------
    peak = float(np.max(np.abs(I_free)))
    I_scaled = I_free * (MAX_CURRENT / peak) if peak > 0 else I_free
    main_currents = I_scaled[:PSXMCoils.N_COILS]
    shield_currents = I_scaled[PSXMCoils.N_COILS:]

    mode = "B_n = 0 on shield surface" if USE_BN_ON_SURFACE else "(Bx,By) = 0 on shield surface"
    print(f"shield-surface boundary condition: {mode}")
    print("main coil currents I1..I6 (A):", main_currents)
    print(f"shield current range (A): [{shield_currents.min():.3f}, {shield_currents.max():.3f}]")
    print("max abs residual at center (T):        ", center_res)
    print("max abs B_n on shield ring (T):        ", surface_bn)
    print("max abs |B| on shield ring (T):        ", surface_full, "(tangential allowed nonzero)")
    print("max abs |B| on outer ring +5mm (T):    ", outside_res)

    # --- plot with the three groups of test points ------------------------
    solved = PSXMCoils(
        currents=main_currents, radius=source.radius, coil_length=source.coil_length,
        start_angle=source.start_angle, shield=True, shield_radius=source.shield_radius,
        shield_n=SHIELD_N, shield_currents=shield_currents,
    )
    fig, ax = plt.subplots(figsize=(7.5, 7))
    solved.draw(ax, extent=2, legend=False)

    cx, cy = np.array(center_pts).T
    sx, sy = np.array(surface_pts).T
    ox, oy = np.array(outside_pts).T
    ax.scatter(cx, cy, s=30, c="tab:blue", marker="o", zorder=5, label="center: quadrupole")
    surf_label = "shield ring: $B_n=0$" if USE_BN_ON_SURFACE else "shield ring: $B=0$"
    ax.scatter(sx, sy, s=12, c="tab:green", marker="o", zorder=5, label=surf_label)
    ax.scatter(ox, oy, s=12, c="tab:orange", marker="x", zorder=5, label="outside +5mm: $B=0$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    ax.set_title(mode, fontsize=10)

    fig.savefig("example.png", dpi=200, bbox_inches="tight")
    print("saved example.png")


if __name__ == "__main__":
    main()
