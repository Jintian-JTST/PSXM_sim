import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver


def add_zero_field_ring(solver, source, radius, n_between, weight):
    """Add sample points requiring zero field, evenly spaced between every
    pair of adjacent shield current points, on the circle of given radius."""
    shield_n = source.shield_n
    gap = 360.0 / shield_n
    for base_angle in source.shield_angles:
        for j in range(1, n_between + 1):
            angle = np.radians(base_angle + j * gap / (n_between + 1))
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            solver.add_sample_point(x, y, Bx=0.0, By=0.0, weight=weight)


def main():
    # Geometry only: both the 6 main-coil currents and the shield-can
    # currents are unknowns to be solved for, not prescribed here.
    shield_n = 100
    source = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=shield_n)

    solver = CurrentSolver.from_current_source(source)

    # Target 1: an ideal quadrupole field (Bx = G*y, By = G*x) near the
    # center (~1 mm).
    G = 1e-3  # T/mm
    n_center = 12
    center_weight = 1.0 / n_center
    for angle in np.linspace(0, 2 * np.pi, n_center, endpoint=False):
        x, y = np.cos(angle), np.sin(angle)
        solver.add_sample_point(x, y, Bx=G * y, By=G * x, weight=center_weight)

    # Targets 2 & 3: zero field at n_between points evenly spaced between
    # every pair of adjacent shield current points, both on the shield
    # circle itself and 1 mm beyond it (checking that leakage stays low
    # just outside the shield too).
    #
    # There are far more of these than center points (shield_n * n_between
    # vs n_center) -- normalizing by count alone isn't enough, though: the
    # shield sits only ~5 mm outside the main coils, so any current strong
    # enough to matter at the center induces a *much* larger field there
    # (an order of magnitude more than the 1 mm quadrupole target). An
    # unweighted (or count-only-weighted) fit is dominated by trivially
    # keeping the shield's residual small via near-zero currents, at the
    # expense of the quadrupole target actually being reached. shield_weight
    # is an extra knob down-weighting the shield groups' importance
    # relative to the center group, to reflect that we care more about
    # hitting the quadrupole than perfectly nulling the shield; lower it to
    # trade shield leakage for a better quadrupole fit (and vice versa).
    n_between = 3
    n_shield_samples = shield_n * n_between
    shield_weight_scale = 1
    shield_weight = shield_weight_scale / n_shield_samples

    add_zero_field_ring(solver, source, source.shield_radius, n_between, shield_weight)
    add_zero_field_ring(solver, source, source.test_radius, n_between, shield_weight)

    I_free = solver.solve()
    B_fit = solver.predicted_field(I_free)
    B_target = solver.target_field()

    I_free_scaled = CurrentSolver.normalize_currents(I_free, max_current=1000.0)
    main_currents = I_free_scaled[:PSXMCoils.N_COILS]
    shield_currents = I_free_scaled[PSXMCoils.N_COILS:]

    n_pts = len(solver.sample_x)
    shield_start = n_center
    outside_start = n_center + n_shield_samples

    def group_residual(start, stop):
        idx = np.r_[start:stop, n_pts + start:n_pts + stop]
        return np.max(np.abs(B_fit - B_target)[idx])

    print("main coil currents I1..I6 (A):", main_currents)
    print(f"shield current range (A): [{shield_currents.min():.3f}, {shield_currents.max():.3f}]")
    print(f"total shield current (A): {shield_currents.sum():.3f}")
    print("max abs residual at center sample points (T): ", group_residual(0, n_center))
    print("max abs residual at shield sample points (T): ", group_residual(shield_start, outside_start))
    print("max abs residual at outside sample points (T):", group_residual(outside_start, n_pts))

    solved = PSXMCoils(
        currents=main_currents, radius=source.radius, coil_length=source.coil_length,
        start_angle=source.start_angle, shield=True, shield_radius=source.shield_radius,
        shield_n=shield_n, shield_currents=shield_currents,
    )

    # Draw the field, then overlay the three groups of sample ("test")
    # points so it's visible *where* each target is imposed:
    #   - center: the quadrupole target (Bx=G*y, By=G*x)
    #   - shield ring / outside ring: the B = 0 (leakage) targets
    fig, ax = plt.subplots(figsize=(7.5, 7))
    solved.draw(ax, extent=2, legend=False)

    sx, sy = np.asarray(solver.sample_x), np.asarray(solver.sample_y)
    ax.scatter(sx[:n_center], sy[:n_center], s=2, c="tab:blue",
               marker="o", zorder=5, label="center target (quadrupole)")
    ax.scatter(sx[shield_start:outside_start], sy[shield_start:outside_start], s=2,
               c="tab:green", marker="o", zorder=5, label="shield ring:  B = 0")
    ax.scatter(sx[outside_start:n_pts], sy[outside_start:n_pts], s=2,
               c="tab:orange", marker="x", zorder=5, label="outside (+5 mm):  B = 0")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)

    fig.savefig("example.png", dpi=200, bbox_inches="tight")
    print("saved example.png")


if __name__ == "__main__":
    main()
