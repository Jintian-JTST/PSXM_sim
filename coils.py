import numpy as np
import matplotlib.pyplot as plt

MU0 = 4e-7 * np.pi  # vacuum permeability, T*m/A


def as_array(v):
    """Coerce v to a 1-D float array, or an empty array if v is None."""
    return np.atleast_1d(np.asarray(v, dtype=float)) if v is not None else np.array([], dtype=float)


class Coils:
    """A collection of coils on a plane, each described by position (x, y) and current I."""

    def __init__(self, x=None, y=None, I=None):
        self.x = as_array(x)
        self.y = as_array(y)
        self.I = as_array(I)

        if not (len(self.x) == len(self.y) == len(self.I)):
            raise ValueError("x, y, and I must have the same length")

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        return f"Coils(n={len(self)})"

    def __iter__(self):
        for xi, yi, Ii in zip(self.x, self.y, self.I):
            yield xi, yi, Ii

    @property
    def positions(self):
        """Return an (n, 2) array of coil positions"""
        return np.column_stack((self.x, self.y))

    def add_coil(self, x, y, I):
        """Add a new coil"""
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.I = np.append(self.I, I)

    def remove_coil(self, index):
        """Remove the coil at the given index"""
        self.x = np.delete(self.x, index)
        self.y = np.delete(self.y, index)
        self.I = np.delete(self.I, index)

    def total_current(self):
        """Sum of all coil currents"""
        return float(np.sum(self.I))

    def B_field(self, x, y, min_distance=1e-6):
        """
        Compute the magnetic flux density B at point (x, y), in Tesla.

        Each coil is modeled as an infinite straight wire perpendicular to
        the x-y plane, piercing it at (self.x, self.y) and carrying current
        self.I (in A). Field point (x, y) and coil positions are in mm.

        min_distance: minimum allowed distance (mm) from the field point to
        any wire, to avoid a singularity if the point sits on a conductor.

        Returns (Bx, By), in Tesla.
        """
        dx = (x - self.x) * 1e-3  # mm -> m
        dy = (y - self.y) * 1e-3
        rho2 = dx**2 + dy**2

        if np.any(rho2 < (min_distance * 1e-3) ** 2):
            raise ValueError("Field point coincides with a coil position (singularity)")

        Bx = np.sum(-MU0 * self.I * dy / (2 * np.pi * rho2))
        By = np.sum(MU0 * self.I * dx / (2 * np.pi * rho2))
        return np.array([Bx, By])

    def B_magnitude(self, x, y, min_distance=1e-6):
        """Magnitude of the magnetic flux density |B| at (x, y), in Tesla."""
        return float(np.linalg.norm(self.B_field(x, y, min_distance=min_distance)))

    def A_z(self, x, y):
        """
        Magnetic vector potential Az at point(s) (x, y) [mm], due to all
        coils modeled as infinite straight wires perpendicular to the plane.

        Contour lines of Az are exactly the magnetic field lines, so this
        is meant to be plotted with a contour method rather than read as an
        absolute physical value (it depends on an arbitrary reference
        distance baked into the log term; only relative differences /
        contour shapes are meaningful).

        x, y may be scalars or arrays (e.g. a meshgrid); returns the same
        shape.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dx = (x[..., np.newaxis] - self.x) * 1e-3  # mm -> m
        dy = (y[..., np.newaxis] - self.y) * 1e-3
        rho2 = dx**2 + dy**2
        rho2 = np.where(rho2 < 1e-24, 1e-24, rho2)

        Az = -MU0 / (4 * np.pi) * np.sum(self.I * np.log(rho2), axis=-1)
        return Az if Az.shape else float(Az)

    def _plot_field(self, n_grid, n_levels, ax, extent):
        """
        Draw the Az contour lines (magnetic field lines), with the region
        Az < Az(0, 0) shaded, and the coil positions marked, onto ax.

        The plotted range spans [-R*extent, R*extent] in both x and y,
        where R is the largest distance from the origin among the coil
        positions (or 10 mm if there are no coils). Does not show/save.
        """
        R = np.max(np.hypot(self.x, self.y)) if len(self) else 10.0
        L = R * extent
        xs = np.linspace(-L, L, n_grid)
        ys = np.linspace(-L, L, n_grid)
        X, Y = np.meshgrid(xs, ys)
        Az = self.A_z(X, Y)

        Az0 = float(self.A_z(0.0, 0.0))
        if Az0 > Az.min():
            ax.contourf(X, Y, Az, levels=[Az.min(), Az0], colors=["0.6"], alpha=0.4, zorder=0)

        ax.contour(X, Y, Az, levels=n_levels, colors="k", linewidths=0.6)
        ax.plot(self.x, self.y, "ks", markersize=4)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Magnetic field lines (Az contours)")
        return ax

    @staticmethod
    def _show_or_save(ax, path, dpi=200):
        if path:
            ax.figure.savefig(path, bbox_inches="tight", dpi=dpi)
        else:
            plt.show()
        return ax

    def plot(self, n_grid=400, n_levels=40, ax=None, path=None, extent=1.3, dpi=200):
        """
        Plot magnetic field lines around the coils, as contour lines of the
        magnetic vector potential Az (contours of Az are exactly the field
        lines for this 2D line-current model), with the region where
        Az < Az(0, 0) shaded.

        The plotted range spans [-R*extent, R*extent] in both x and y,
        where R is the largest distance from the origin among the coil
        positions (or 10 mm if there are no coils).

        If path is non-empty, the figure is saved there (at the given dpi)
        instead of shown.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        self._plot_field(n_grid, n_levels, ax, extent)
        return self._show_or_save(ax, path, dpi=dpi)


if __name__ == "__main__":
    coils = Coils(x=[0, 1, 2], y=[0, 1, 0], I=[1.0, -1.0, 0.5])
    print(coils)
    print(coils.positions)
    coils.add_coil(3, 3, 2.0)
    print(coils)
    print("total current:", coils.total_current())
