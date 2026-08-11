"""node5_common.py -- shared helpers for the node-5 studies.

The three node-5 questions (rotational quadrupole, maximum reachable
field, 2D -> 3D edge effects) all need the same two primitives, so they
live here rather than being re-derived in each script:

  * ``multipoles()`` -- given any ``Coils``-like object, measure what
    field it actually makes near the centre by fitting the dipole and
    quadrupole (normal + skew) content on a small sampling ring.  Every
    "achieved field" number in the report comes from this one function,
    so the three studies are on the same footing.

  * ``analytic_ceilings()`` / ``unit_response()`` -- the closed-form and
    numerical hardware ceilings of the six-coil ring.  ``multipoles`` of
    a least-squares solve tells you what a *field-quality-constrained*
    design achieves; these tell you what the hardware could do at all.
    Their ratio is the price of field quality.

Plus two bits of plumbing so no number in the design report is typed by
hand: ``save_fig`` writes a figure into both ``figures/`` and the
report's ``figures/``, and ``write_macros`` emits a LaTeX fragment of
``\\renewcommand``s that ``main.tex`` inputs.

Conventions.  Positions mm, currents A, fields T, gradients T/mm.  Note
1 T/m == 1 mT/mm, which is the unit the report quotes gradients in.

Multipole convention (matches ``current_solver.py``'s rotated-quadrupole
demo): writing the near-centre field as

    Bx = B0x + Gs*x + Gn*y
    By = B0y + Gn*x - Gs*y

``Gn`` is the normal quadrupole gradient, ``Gs`` the skew one.  A
normal quadrupole rotated mechanically by phi has
``(Gn, Gs) = |G| (cos 2phi, -sin 2phi)``, i.e. the quadrupole moment is a
spin-2 object: a 180 deg mechanical rotation maps it back to itself.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")           # batch scripts: always save, never show
import matplotlib.pyplot as plt

from coils import Coils, MU0
from PSXM_coils import PSXMCoils

# --- geometry / hardware defaults (kept in sync with PSXM_coils.py) --------
RADIUS = 22.5          # mm, coil-ring radius
COIL_LENGTH = 20.0     # mm, chord between a coil's two legs
MAX_CURRENT = 1000.0   # A, hardware current budget per coil

REPORT_DIR = os.path.join("..", "PSXM_design_report")
FIG_DIRS = ("figures", os.path.join(REPORT_DIR, "figures"))


# ==========================================================================
# measurement
# ==========================================================================
def multipoles(coils, r0=1.0, n=64):
    """Dipole and quadrupole content of ``coils`` near the origin.

    Samples B on a ring of radius ``r0`` (mm) at ``n`` angles and fits

        Bx = B0x + Gs*x + Gn*y,     By = B0y + Gn*x - Gs*y

    by least squares.  The residual of that fit is the non-dipole,
    non-quadrupole content (sextupole and up) on the ring, so it doubles
    as a field-quality figure of merit.

    Returns a dict with B0x, B0y (T), B0mag (T), B0deg (deg), Gn, Gs,
    Gmag (T/mm), phideg (the mechanical rotation angle of the
    quadrupole, deg), rms (T, fit residual) and purity (rms divided by
    the field scale on the ring).
    """
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y = r0 * np.cos(a), r0 * np.sin(a)
    B = np.array([coils.B_field(xi, yi) for xi, yi in zip(x, y)])
    Bx, By = B[:, 0], B[:, 1]

    M = np.zeros((2 * n, 4))
    M[:n, 0] = 1.0          # B0x
    M[:n, 3] = x            # Gs * x
    M[:n, 2] = y            # Gn * y
    M[n:, 1] = 1.0          # B0y
    M[n:, 2] = x            # Gn * x
    M[n:, 3] = -y           # -Gs * y

    rhs = np.concatenate([Bx, By])
    p, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    res = rhs - M @ p
    B0x, B0y, Gn, Gs = p

    scale = float(np.sqrt(np.mean(rhs ** 2)))
    rms = float(np.sqrt(np.mean(res ** 2)))
    return dict(
        B0x=B0x, B0y=B0y,
        B0mag=float(np.hypot(B0x, B0y)),
        B0deg=float(np.degrees(np.arctan2(B0y, B0x))),
        Gn=Gn, Gs=Gs,
        Gmag=float(np.hypot(Gn, Gs)),
        phideg=float(np.degrees(-0.5 * np.arctan2(Gs, Gn))),
        rms=rms,
        purity=(rms / scale if scale > 0 else np.nan),
    )


def uniformity(coils, radii=(5.0, 10.0), n=48):
    """Peak-to-mean non-uniformity of |B| on rings of the given radii.

    For a dipole design this is the number that matters for steering:
    how flat is the field the muon actually sees across the aperture.
    Returns {radius_mm: max|B - B_centre| / |B_centre|}.
    """
    B0 = np.linalg.norm(coils.B_field(0.0, 0.0))
    out = {}
    for r in radii:
        a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        dev = [np.linalg.norm(coils.B_field(r * np.cos(t), r * np.sin(t))
                              - coils.B_field(0.0, 0.0)) for t in a]
        out[r] = float(np.max(dev) / B0) if B0 > 0 else np.nan
    return out


# ==========================================================================
# hardware ceilings
# ==========================================================================
def analytic_ceilings(radius=RADIUS, coil_length=COIL_LENGTH,
                      max_current=MAX_CURRENT):
    """Closed-form maximum centre dipole / quadrupole of the six-coil ring.

    Derivation (see the design report, "Maximum reachable field"):
    writing the 2D field of a wire at complex position w as
    ``By + i Bx = mu0 I / (2 pi (z - w))`` and expanding about z = 0, the
    dipole and quadrupole moments of the whole ring reduce to

        B0  ~ 2 sin(w_half)   * mu0/(2 pi R  ) * |S1|,   S1 = sum_k s_k I_k e^{-i c_k}
        G   ~ 2 sin(2 w_half) * mu0/(2 pi R^2) * |S2|,   S2 = sum_k s_k I_k e^{-2i c_k}

    with c_k = 60k deg the coil centre angles, s_k = (-1)^k the leg-sign
    convention of ``PSXMCoils``, and w_half = arcsin(L/2R) the half-angle
    between a coil's two legs.  Both S1 and S2 collapse onto three unit
    vectors 120 deg apart, so under |I_k| <= I_max their modulus is at
    most 4 I_max in the best direction and 2*sqrt(3) I_max in the worst,
    a ripple of exactly 2/sqrt(3) = 15.5%.

    Returns a dict in T (dipole) and T/m == mT/mm (gradient).
    """
    R = radius * 1e-3
    wh = np.arcsin(coil_length / (2.0 * radius))
    b1 = MU0 / (2.0 * np.pi * R)          # T/A, one leg's field at the centre
    g1 = MU0 / (2.0 * np.pi * R ** 2)     # T/(A m), one leg's gradient
    dip_best = 2.0 * np.sin(wh) * b1 * 4.0 * max_current
    quad_best = 2.0 * np.sin(2.0 * wh) * g1 * 4.0 * max_current
    k = np.sqrt(3.0) / 2.0
    return dict(
        half_angle_deg=float(np.degrees(wh)),
        dipole_best=float(dip_best), dipole_worst=float(dip_best * k),
        quad_best=float(quad_best), quad_worst=float(quad_best * k),
        ripple=float(2.0 / np.sqrt(3.0)),
    )


def unit_response(radius=RADIUS, coil_length=COIL_LENGTH, r0=1.0, n=64):
    """Per-coil centre response: row k = (B0x, B0y, Gn, Gs) for 1 A in coil k.

    Purely numerical (it just calls ``multipoles`` on a one-coil-excited
    ring), so it is an independent check on ``analytic_ceilings``.
    """
    rows = []
    for k in range(PSXMCoils.N_COILS):
        I = np.zeros(PSXMCoils.N_COILS)
        I[k] = 1.0
        c = PSXMCoils(currents=I, radius=radius, coil_length=coil_length)
        m = multipoles(c, r0=r0, n=n)
        rows.append([m["B0x"], m["B0y"], m["Gn"], m["Gs"]])
    return np.array(rows)


def lp_ceiling(cols, psi, max_current=MAX_CURRENT):
    """Largest moment reachable along direction ``psi`` under |I_k| <= I_max.

    ``cols`` is the (6, 2) block of ``unit_response`` for the moment of
    interest (columns 0:2 for the dipole, 2:4 for the quadrupole).  The
    moment is linear in the currents, so the maximum of its projection on
    a unit direction is attained at a vertex of the current box:
    I_k = I_max * sign(n . c_k), giving I_max * sum_k |n . c_k|.

    For the quadrupole the "direction" psi is 2*phi, because the
    quadrupole moment is spin-2.
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    n = np.stack([np.cos(psi), np.sin(psi)], axis=-1)        # (m, 2)
    proj = n @ cols.T                                         # (m, 6)
    return max_current * np.sum(np.abs(proj), axis=-1)


# ==========================================================================
# finite-length (2.5D) correction -- used by node5_edge_effects.py
# ==========================================================================
def length_factor(rho_mm, length_mm):
    """Mid-plane field of a finite straight segment / that of an infinite wire.

    A segment of length L centred on the mid-plane produces, at
    perpendicular distance rho in that plane,

        B = (mu0 I / 2 pi rho) * (L/2) / sqrt(rho^2 + (L/2)^2)

    so the correction to the infinite-wire model is exactly
    f = [1 + (2 rho / L)^2]^{-1/2}.  It is always < 1: the 2D model
    always *overestimates*, and increasingly so with distance.
    """
    return 1.0 / np.sqrt(1.0 + (2.0 * np.asarray(rho_mm, dtype=float)
                                / float(length_mm)) ** 2)


def validity_radius(length_mm, tol=0.10):
    """Radius out to which the 2D model is accurate to within ``tol``.

    From f >= 1 - tol: rho <= (L/2) sqrt((1-tol)^-2 - 1).  For tol = 10%
    this is rho <~ L/4.
    """
    return 0.5 * length_mm * np.sqrt((1.0 - tol) ** -2 - 1.0)


class FiniteCoils(Coils):
    """A ``Coils`` whose wires are finite segments of length ``length`` (mm).

    Only valid in the mid-plane, where the correction is the exact
    ``length_factor`` above applied per wire.  It models the truncation
    of the straight legs; it does *not* model the field of the end turns
    that close the circuit, which is the remaining genuinely-3D piece.
    """

    def __init__(self, base, length):
        super().__init__(x=base.x, y=base.y, I=base.I)
        self.length = float(length)

    def B_field(self, x, y, min_distance=1e-6):
        dx = (x - self.x) * 1e-3
        dy = (y - self.y) * 1e-3
        rho2 = dx ** 2 + dy ** 2
        if np.any(rho2 < (min_distance * 1e-3) ** 2):
            raise ValueError("Field point coincides with a coil position")
        L = self.length * 1e-3
        f = 1.0 / np.sqrt(1.0 + (4.0 * rho2) / (L ** 2))
        Bx = np.sum(-MU0 * self.I * f * dy / (2 * np.pi * rho2))
        By = np.sum(MU0 * self.I * f * dx / (2 * np.pi * rho2))
        return np.array([Bx, By])


# ==========================================================================
# output plumbing
# ==========================================================================
def save_fig(fig, name, dpi=170):
    """Save into both figures/ and the report's figures/ directory."""
    for d in FIG_DIRS:
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print("saved", p)
    plt.close(fig)


def write_macros(fname, macros):
    """Emit a LaTeX fragment of \\renewcommand's for the design report.

    ``main.tex`` declares every macro with a placeholder default and then
    \\input's this file if it exists, so the report compiles before the
    scripts have been run and picks up the real numbers afterwards.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    p = os.path.join(REPORT_DIR, fname)
    with open(p, "w", encoding="utf-8") as f:
        f.write("% auto-generated by the node-5 scripts -- do not edit\n")
        for k, v in macros.items():
            f.write("\\renewcommand{\\%s}{%s}\n" % (k, v))
    print("wrote", p)
