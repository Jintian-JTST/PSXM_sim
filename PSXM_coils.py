import numpy as np
import matplotlib.pyplot as plt

from coils import Coils, as_array


class PSXMCoils(Coils):
    """
    Six coils of an PSXM sensor, evenly spaced around a circle.

    Each physical coil is a racetrack-style winding whose two straight legs
    pierce the x-y plane at the same radius but at slightly different
    angles (the "+" and "-" legs shown in the reference drawing). The
    angular separation between a coil's two legs is derived from
    ``coil_length`` (straight-line/chord distance between the legs) and
    ``radius`` via the chord formula:

        half_width = arcsin(coil_length / (2 * radius))

    Coil centers are spaced 60 degrees apart, starting at ``start_angle``
    and proceeding counterclockwise, matching the I1..I6 order in the
    reference drawing (I1 east, I2 north, I3 northwest, I4 west, I5 south,
    I6 southeast):

        center_angle[k] = start_angle + k * 60 deg   (k = 0..5, currents[k] = I{k+1})

    currents[k] is the current of coil k, split across its two legs with
    opposite sign. The sign assignment alternates with coil parity so that,
    per the reference drawing, the two legs facing each other across the
    small gap between neighboring coils always carry the same sign:
    for even k, the leg at (center_angle - half_width) is +currents[k] and
    the leg at (center_angle + half_width) is -currents[k]; for odd k it's
    the other way round.

    Note: these geometric assumptions are a simplified approximation of the
    reference drawing; adjust start_angle / leg sign convention if they
    don't match the physical hardware.

    If shield is True, shield_n additional current points are placed
    evenly around a circle of radius shield_radius, modeling the currents
    induced on a conducting shield can around the coils. Their currents
    (shield_currents, A) default to zero and are independent free
    variables (see group_matrix), meant to be solved for (e.g. with
    CurrentSolver) rather than prescribed.
    """

    N_COILS = 6

    def __init__(self, currents, radius=22.5, coil_length=20.0, start_angle=0,
                 shield=False, shield_radius=27.5, shield_n=100, shield_currents=None):
        currents = np.atleast_1d(np.asarray(currents, dtype=float))
        if len(currents) != self.N_COILS:
            raise ValueError(f"PSXMCoils requires exactly {self.N_COILS} currents, got {len(currents)}")
        if coil_length > 2 * radius:
            raise ValueError("coil_length cannot exceed 2 * radius")
        if shield_currents is not None and not shield:
            raise ValueError("shield_currents was given but shield is False")

        self.radius = radius
        self.coil_length = coil_length
        self.start_angle = start_angle
        self.currents = currents
        self.center_angles = start_angle + np.arange(self.N_COILS) * (360.0 / self.N_COILS)
        # +1/-1 per coil, alternating: leg at (center - half_width) carries
        # +leg_signs[k]*currents[k], leg at (center + half_width) carries
        # the opposite sign. Shared with group_matrix().
        self.leg_signs = np.where(np.arange(self.N_COILS) % 2 == 0, 1.0, -1.0)

        self.shield = shield
        self.shield_radius = shield_radius
        self.shield_n = shield_n if shield else 0

        half_width = np.degrees(np.arcsin(coil_length / (2.0 * radius)))

        x, y, I = [], [], []
        for k in range(self.N_COILS):
            theta_minus = np.radians(self.center_angles[k] - half_width)
            theta_plus = np.radians(self.center_angles[k] + half_width)
            sign = self.leg_signs[k]

            x.append(radius * np.cos(theta_minus))
            y.append(radius * np.sin(theta_minus))
            I.append(sign * currents[k])

            x.append(radius * np.cos(theta_plus))
            y.append(radius * np.sin(theta_plus))
            I.append(-sign * currents[k])

        # Invariant relied on elsewhere (e.g. _plot_shield, group_matrix):
        # the shield_n shield points, if any, are always the *last* entries
        # in self.x/self.y/self.I, right after the 2*N_COILS leg points.
        if shield:
            self.shield_angles = np.arange(shield_n) * (360.0 / shield_n)
            shield_theta = np.radians(self.shield_angles)
            shield_I = np.zeros(shield_n) if shield_currents is None else as_array(shield_currents)
            if len(shield_I) != shield_n:
                raise ValueError(f"shield_currents must have length shield_n={shield_n}, got {len(shield_I)}")

            x.extend((shield_radius * np.cos(shield_theta)).tolist())
            y.extend((shield_radius * np.sin(shield_theta)).tolist())
            I.extend(shield_I.tolist())

        super().__init__(x=x, y=y, I=I)

    def __repr__(self):
        return f"PSXMCoils(radius={self.radius}, coil_length={self.coil_length}, currents={self.currents.tolist()})"

    def group_matrix(self):
        """
        Grouping matrix M (n_points x n_free) mapping free variables to
        per-point currents, i.e. self.I == M @ free_currents. The first
        N_COILS free variables are the physical coil currents, split
        across each coil's two leg points with +/- sign as in __init__
        (self.I[:12] == M[:12] @ free_currents). If shield is enabled, the
        shield_n shield points are appended as independent free variables
        (an identity block), so free_currents has length N_COILS +
        shield_n. Meant for CurrentSolver.
        """
        n_free = self.N_COILS + self.shield_n
        M = np.zeros((len(self), n_free))
        for k in range(self.N_COILS):
            M[2 * k, k] = self.leg_signs[k]
            M[2 * k + 1, k] = -self.leg_signs[k]
        if self.shield:
            M[2 * self.N_COILS:, self.N_COILS:] = np.eye(self.shield_n)
        return M

    def plot(self, n_grid=400, n_levels=40, ax=None, path=None, extent=4.0 / 3.0, dpi=200):
        """
        Plot magnetic field lines around the coil ring (see Coils.plot),
        plus the coil ring boundary and I1..I6 labels/legend.

        The plotted range spans [-radius*extent, radius*extent] in both x
        and y (extent defaults to 4/3).

        If path is non-empty, the figure is saved there (at the given dpi)
        instead of shown.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        self._plot_field(n_grid, n_levels, ax, extent)

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(self.radius * np.cos(theta), self.radius * np.sin(theta), "k-", linewidth=1.2)

        label_r = self.radius * 1.15
        for k in range(self.N_COILS):
            angle = np.radians(self.center_angles[k])
            ax.text(
                label_r * np.cos(angle), label_r * np.sin(angle),
                f"$I_{k + 1}$", ha="center", va="center", fontsize=11, clip_on=True,
            )
            ax.plot([], [], " ", label=f"$I_{k + 1}$ = {self.currents[k]:.4g} A")

        if self.shield:
            self._plot_shield(ax)

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

        return self._show_or_save(ax, path, dpi=dpi)

    def _plot_shield(self, ax, arrow_max_length=None):
        """
        Draw the shield can boundary and an arrow per shield current point:
        arrow length is proportional to |current|, pointing inward for
        current flowing into the page (I < 0) and outward for current
        flowing out of the page (I > 0).
        """
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(self.shield_radius * np.cos(theta), self.shield_radius * np.sin(theta), "k--", linewidth=0.8)

        # relies on the shield points being the last shield_n entries (see
        # the invariant noted in __init__)
        shield_x = self.x[-self.shield_n:]
        shield_y = self.y[-self.shield_n:]
        shield_I = self.I[-self.shield_n:]

        max_abs = np.max(np.abs(shield_I))
        if max_abs == 0:
            return

        if arrow_max_length is None:
            arrow_max_length = 0.15 * self.shield_radius

        ux = shield_x / self.shield_radius
        uy = shield_y / self.shield_radius
        length = arrow_max_length * np.abs(shield_I) / max_abs
        direction = np.sign(shield_I)

        ax.quiver(
            shield_x, shield_y, direction * ux * length, direction * uy * length,
            angles="xy", scale_units="xy", scale=1, width=0.003, color="tab:red", zorder=3,
        )


if __name__ == "__main__":
    psxm = PSXMCoils(currents=[729.3, 1000, 270.7, -729.3, -1000, -270.7])
    print(psxm)
    print(psxm.positions)
    psxm.plot(path="test.png")
