import numpy as np

from coils import Coils, MU0, as_array


class CurrentSolver:
    """
    Solves for the currents at a set of fixed current points that best
    reproduce, in a least-squares sense, a prescribed magnetic field at a
    set of sample points.

    Sample points: locations (mm) where the desired field (Bx, By in
    Tesla) is specified.
    Current points: locations (mm) where an unknown current (A) is to be
    solved for.

    The field-current relationship is the same infinite-straight-wire
    model used by Coils.B_field: each current point contributes a field at
    a sample point proportional to its current, and the total field is the
    linear superposition B = K @ I, where K is the coefficient matrix
    returned by coefficient_matrix().

    Some current sources (e.g. PSXMCoils) have fewer physical degrees of
    freedom than current points, because one physical coil pierces the
    plane at several points whose currents are a fixed linear combination
    of a single coil current (e.g. +I_k / -I_k on its two legs). Set
    group_matrix (n_current_points x n_free) so that
    per-point currents = group_matrix @ free currents; solve() then solves
    for the smaller free-variable vector instead of one DOF per point. Use
    from_current_source() to build this automatically from a source object
    that defines a group_matrix() method (e.g. PSXMCoils).
    """

    def __init__(self, sample_x=None, sample_y=None, sample_Bx=None, sample_By=None,
                 current_x=None, current_y=None, group_matrix=None):
        self.sample_x = as_array(sample_x)
        self.sample_y = as_array(sample_y)
        self.sample_Bx = as_array(sample_Bx)
        self.sample_By = as_array(sample_By)
        self.sample_weight = np.ones(len(self.sample_x))

        if not (len(self.sample_x) == len(self.sample_y) == len(self.sample_Bx) == len(self.sample_By)):
            raise ValueError("sample_x, sample_y, sample_Bx, and sample_By must have the same length")

        self.current_x = as_array(current_x)
        self.current_y = as_array(current_y)

        if len(self.current_x) != len(self.current_y):
            raise ValueError("current_x and current_y must have the same length")

        self.group_matrix = np.asarray(group_matrix, dtype=float) if group_matrix is not None else None
        if self.group_matrix is not None and self.group_matrix.shape[0] != len(self.current_x):
            raise ValueError("group_matrix must have one row per current point")

    @classmethod
    def from_current_source(cls, source, sample_x=None, sample_y=None, sample_Bx=None, sample_By=None):
        """
        Build a CurrentSolver whose current points are the positions of
        `source` (any Coils instance). If source defines a group_matrix()
        method (e.g. PSXMCoils), it is used to reduce the number of free
        variables solved for to source's physical degrees of freedom.
        """
        group_matrix = source.group_matrix() if hasattr(source, "group_matrix") else None
        return cls(sample_x=sample_x, sample_y=sample_y, sample_Bx=sample_Bx, sample_By=sample_By,
                   current_x=source.x, current_y=source.y, group_matrix=group_matrix)

    def add_sample_point(self, x, y, Bx, By, weight=1.0):
        """
        Add a point (mm) with a prescribed target field (Bx, By in Tesla).

        weight: relative importance of this point in the least-squares fit
        (see solve()). Raise it to prioritize a point/group of points over
        others, e.g. when one group vastly outnumbers another and would
        otherwise dominate an unweighted fit.
        """
        self.sample_x = np.append(self.sample_x, x)
        self.sample_y = np.append(self.sample_y, y)
        self.sample_Bx = np.append(self.sample_Bx, Bx)
        self.sample_By = np.append(self.sample_By, By)
        self.sample_weight = np.append(self.sample_weight, weight)

    def add_current_point(self, x, y):
        """Add a point (mm) where an unknown current is to be solved for."""
        if self.group_matrix is not None:
            raise ValueError(
                "cannot add a current point once group_matrix is set (its row "
                "count would no longer match the number of current points); "
                "add all current points first, then set group_matrix"
            )
        self.current_x = np.append(self.current_x, x)
        self.current_y = np.append(self.current_y, y)

    def coefficient_matrix(self):
        """
        Coefficient matrix K (T/A) such that B = K @ I, with I the vector
        of currents at the current points, and B the target field vector
        stacked as [Bx at every sample point, By at every sample point].

        Uses the same infinite-straight-wire model as Coils.B_field.
        """
        dx = (self.sample_x[:, None] - self.current_x[None, :]) * 1e-3  # mm -> m
        dy = (self.sample_y[:, None] - self.current_y[None, :]) * 1e-3
        rho2 = dx**2 + dy**2

        if np.any(rho2 < 1e-18):
            raise ValueError("A sample point coincides with a current point (singularity)")

        Kx = -MU0 * dy / (2 * np.pi * rho2)
        Ky = MU0 * dx / (2 * np.pi * rho2)
        return np.vstack([Kx, Ky])

    def target_field(self):
        """Target field vector B, stacked as [Bx..., By...] (Tesla)."""
        return np.concatenate([self.sample_Bx, self.sample_By])

    def solve(self):
        """
        Solve for the free-variable currents (A) that best reproduce the
        prescribed sample-point fields in a weighted least-squares sense
        (minimizing sum(weight * residual**2), using each sample point's
        weight for both its Bx and By residual).

        If group_matrix is None, the free variables are the per-current-
        point currents directly. Otherwise, they are the reduced/coupled
        variables in per-point currents = group_matrix @ free currents
        (e.g. one DOF per physical coil instead of per point).

        Returns the raw, unnormalized solution -- see normalize_currents()
        to rescale it for e.g. a hardware current limit. point_currents(),
        predicted_field(), and to_coils() all take an explicit I_free
        argument (defaulting to a fresh call to solve()) rather than
        re-solving with different options, so a single I_free -- solved,
        then optionally normalized -- stays consistent across all of them.
        """
        K = self.coefficient_matrix()
        if self.group_matrix is not None:
            K = K @ self.group_matrix
        B = self.target_field()

        sqrt_w = np.sqrt(np.concatenate([self.sample_weight, self.sample_weight]))
        I_free, *_ = np.linalg.lstsq(K * sqrt_w[:, None], B * sqrt_w, rcond=None)
        return I_free

    @staticmethod
    def normalize_currents(I_free, max_current):
        """
        Scale a current vector so its largest absolute value equals
        max_current (A). Preserves the relative current distribution (and
        hence the field shape), but the scaled currents generally no
        longer reproduce the target field's exact magnitude.
        """
        I_free = np.asarray(I_free, dtype=float)
        return I_free * (max_current / np.max(np.abs(I_free)))

    def point_currents(self, I_free=None):
        """
        Current (A) at every current point, expanding I_free through
        group_matrix if set. I_free defaults to a fresh solve().
        """
        if I_free is None:
            I_free = self.solve()
        return self.group_matrix @ I_free if self.group_matrix is not None else np.asarray(I_free, dtype=float)

    def predicted_field(self, I_free=None):
        """Fitted field vector B = K @ I for I_free (Tesla); I_free defaults to a fresh solve()."""
        return self.coefficient_matrix() @ self.point_currents(I_free)

    def to_coils(self, I_free=None):
        """Build a Coils object at the current points, for I_free (defaults to a fresh solve())."""
        return Coils(x=self.current_x, y=self.current_y, I=self.point_currents(I_free))


if __name__ == "__main__":
    from PSXM_coils import PSXMCoils

    truth = PSXMCoils(currents=[729.3, 1000, 270.7, -729.3, -1000, -270.7])

    # target: an ideal quadrupole field (Bx = G*y, By = G*x) sampled on a
    # small (~1 mm) circle around the origin. Solve for the 6 physical
    # PSXM coil currents (not one DOF per leg point) via group_matrix.
    G = 1e-2  # T/mm
    solver = CurrentSolver.from_current_source(truth)
    phi = np.deg2rad(10)          # rotation angle
    c, s = np.cos(2*phi), np.sin(2*phi)
    r =20  # mm
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x, y = r * np.cos(angle), r * np.sin(angle)
        solver.add_sample_point(x, y,
            Bx=G*(y*c - x*s),
            By=G*(x*c + y*s))

    I_free = solver.solve()
    B_fit = solver.predicted_field(I_free)
    B_target = solver.target_field()

    print("solved I1..I6:  ", I_free)
    print("target B (T):  ", B_target)
    print("fitted B (T):  ", B_fit)
    print("max abs field residual (T):", np.max(np.abs(B_fit - B_target)))

    I_free_scaled = CurrentSolver.normalize_currents(I_free, max_current=1000.0)
    solved_psxm = PSXMCoils(
        currents=I_free_scaled, radius=truth.radius, coil_length=truth.coil_length, start_angle=truth.start_angle,
    )
    solved_psxm.plot(path="figures/solved_psxm_field.png", extent=2.44)
