"""Finite-length (3-D) PSXM leakage field: coils-only vs coils+shield,
both on one plot.

The 2-D infinite-wire model has no z-dependence, so it cannot give the
axial fall-off of the leakage. Here the 6 coils are closed rectangular
loops (axial length COIL_LENGTH_Z); the shield's solved surface currents
are finite axial current sticks of the same length at the shield radius.
We solve a *shielded dipole* (uniform Bx at the centre + B=0 on/around the
shield) and compare |B| vs distance with and without the shield.

Caveat: the shield sticks are bare finite segments (no explicit axial
return path). The net shield current is ~0 so the far field is
dipole-dominated -- a reasonable first estimate; a fully charge-conserving
3-D shield model is the natural refinement.

Run:  python field3d.py
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver

MU0 = 4e-7 * np.pi
COIL_LENGTH_Z = 0.10   # m, axial length of the PSXM coils (ASSM eff. length ~0.1 m)
MAX_CURRENT = 1000.0   # A, currents rescaled so max|I| = this
SPEC_UT = 3.0          # µT, benchmark = 1e-6 of the 3 T main field
DIST_MARKS = (0.235, 0.42)   # m, distances of interest (nearest beam ~0.42 m)


# ----------------------------------------------------------- Biot-Savart
def seg_field(P1, P2, I, pts):
    """B (T) at pts (N,3) [m] from a straight filament P1->P2 [m], current I [A].
    Closed-form finite-segment Biot-Savart (verified vs the
    perpendicular-bisector result)."""
    P1 = np.asarray(P1, float); P2 = np.asarray(P2, float)
    a = pts - P1
    b = pts - P2
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cross = np.cross(a, b)
    denom = na * nb * (na * nb + np.einsum("ij,ij->i", a, b))
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    fac = (MU0 * I / (4 * np.pi)) * (na + nb) / denom
    return cross * fac[:, None]


def loop_field(vertices, I, pts):
    B = np.zeros((len(pts), 3))
    n = len(vertices)
    for k in range(n):
        B += seg_field(vertices[k], vertices[(k + 1) % n], I, pts)
    return B


# ----------------------------------------------------------- geometry
def coil_loops(coils, L=COIL_LENGTH_Z):
    """6 rectangular loops [m] from a (no-shield) PSXMCoils: legs (2k,2k+1)
    become a loop carrying the per-leg current coils.I[2k]."""
    x = coils.x * 1e-3; y = coils.y * 1e-3
    half = L / 2.0
    loops = []
    for k in range(PSXMCoils.N_COILS):
        xa, ya = x[2 * k], y[2 * k]
        xb, yb = x[2 * k + 1], y[2 * k + 1]
        verts = [np.array([xa, ya, -half]), np.array([xa, ya, half]),
                 np.array([xb, yb, half]), np.array([xb, yb, -half])]
        loops.append((verts, coils.I[2 * k]))
    return loops


def coils_field(loops, pts):
    B = np.zeros((len(pts), 3))
    for verts, I in loops:
        B += loop_field(verts, I, pts)
    return B


def shield_field(src, shield_I, pts, L=COIL_LENGTH_Z):
    """Field from the shield currents as finite axial segments [m]."""
    xs = src.x[-src.shield_n:] * 1e-3
    ys = src.y[-src.shield_n:] * 1e-3
    half = L / 2.0
    B = np.zeros((len(pts), 3))
    for x, y, I in zip(xs, ys, shield_I):
        B += seg_field([x, y, -half], [x, y, half], I, pts)
    return B


# ----------------------------------------------------------- solve
def add_zero_field_ring(solver, src, radius, n_between, weight):
    gap = 360.0 / src.shield_n
    for base in src.shield_angles:
        for j in range(1, n_between + 1):
            a = np.radians(base + j * gap / (n_between + 1))
            solver.add_sample_point(radius * np.cos(a), radius * np.sin(a),
                                    Bx=0.0, By=0.0, weight=weight)


def solve_shielded_dipole(B0=1e-3, shield_n=100, n_between=3, outer=5.0):
    """6 coil currents + shield currents for a uniform dipole (Bx=B0) at the
    centre with B=0 on/around the shield."""
    src = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=shield_n)
    solver = CurrentSolver.from_current_source(src)
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        solver.add_sample_point(np.cos(ang), np.sin(ang), Bx=B0, By=0.0, weight=1 / 12)
    sw = 1.0 / (shield_n * n_between)
    add_zero_field_ring(solver, src, src.shield_radius, n_between, sw)
    add_zero_field_ring(solver, src, src.shield_radius + outer, n_between, sw)
    I = CurrentSolver.normalize_currents(solver.solve(), MAX_CURRENT)
    return src, I[:PSXMCoils.N_COILS], I[PSXMCoils.N_COILS:]


# ----------------------------------------------------------- main
def main():
    src, main_I, shield_I = solve_shielded_dipole()
    loops = coil_loops(PSXMCoils(currents=main_I))

    def B_coils(pts):
        return coils_field(loops, pts)

    def B_total(pts):
        return coils_field(loops, pts) + shield_field(src, shield_I, pts)

    B_center = np.linalg.norm(B_total(np.array([[1e-3, 0.0, 0.0]]))[0])
    print(f"coil axial length L = {COIL_LENGTH_Z} m,  shield_n = {src.shield_n}")
    print(f"central dipole |B| (1 mm off-axis): {B_center*1e3:.3f} mT")
    print(f"spec benchmark: {SPEC_UT} µT  (1e-6 of 3 T)\n")

    dist = np.geomspace(2e-3, 0.42, 200)
    axial = np.column_stack([np.zeros_like(dist), np.zeros_like(dist), dist])
    Bc = np.linalg.norm(B_coils(axial), axis=1)
    Bt = np.linalg.norm(B_total(axial), axis=1)

    for d0 in DIST_MARKS:
        i = int(np.argmin(np.abs(dist - d0)))
        supp = Bc[i] / Bt[i] if Bt[i] > 0 else np.inf
        print(f"axial leakage at {d0:.3f} m:  coils only {Bc[i]*1e6:8.3f} µT   "
              f"coils+shield {Bt[i]*1e6:8.3f} µT   (shield x{supp:.1f})")

    plt.figure(figsize=(7, 5))
    plt.loglog(dist, Bc * 1e6, label="coils only")
    plt.loglog(dist, Bt * 1e6, label="coils + shield")
    plt.axhline(SPEC_UT, color="r", ls=":", lw=1, label=f"spec {SPEC_UT} µT")
    for d0 in DIST_MARKS:
        plt.axvline(d0, color="gray", ls="--", lw=0.8)
    plt.xlabel("axial distance from PSXM centre (m)")
    plt.ylabel("|B| (µT)")
    plt.title(f"PSXM axial leakage vs distance (L={COIL_LENGTH_Z} m)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("leakage_vs_distance.png", dpi=150)
    print("\nsaved leakage_vs_distance.png")


if __name__ == "__main__":
    main()
