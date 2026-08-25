"""shield.py -- the ideal-sheet shield response of the PSXM.

The shield-current solve shared by the analysis scripts: the quadrupole
and dipole coil solves, the B=0 sample-point layout, the least-squares
shield response matrix S, and the ring-averaged |B| measurement.  The
physical induced-eddy model lives separately in ``induction.py`` (report
section 7.1, future work) and is not part of the current report.

Conventions: quadrupole target G [T/mm], coil currents normalized to
MAX_CURRENT, radial axis out to R_MAX_MM.
"""

import numpy as np

from config import G, MAX_CURRENT, SHIELD_N, SAMPLE_GAP_MM, OUTER_MM, N_BETWEEN, DIPOLE_TARGET
from psxm_coils import PSXMCoils
from current_solver import CurrentSolver

MU0 = 4e-7 * np.pi

R_MAX_MM = 420.0       # radial axis limit (0.42 m ~ nearest beam)


# --- geometry / solving ---------------------------------------------------
def make_template(shield_n=SHIELD_N):
    return PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS), shield=True, shield_n=shield_n)


def solve_quad_coils(tpl):
    """Coil currents (coil DOF only) for the centre quadrupole, normalized."""
    solver = CurrentSolver.from_current_source(tpl)
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x, y = np.cos(a), np.sin(a)                       # 1 mm ring
        solver.add_sample_point(x, y, Bx=G * y, By=G * x)
    K = (solver.coefficient_matrix() @ tpl.group_matrix())[:, :PSXMCoils.N_COILS]
    I, *_ = np.linalg.lstsq(K, solver.target_field(), rcond=None)
    return CurrentSolver.normalize_currents(I, MAX_CURRENT)

def solve_dipole_coils(tpl, B0=DIPOLE_TARGET):
    """Coil currents (coil DOF only) for a uniform dipole Bx = B0, By = 0
    at the centre, normalized to MAX_CURRENT."""
    solver = CurrentSolver.from_current_source(tpl)
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x, y = np.cos(a), np.sin(a)                       # 1 mm ring
        solver.add_sample_point(x, y, Bx=B0, By=0.0)
    K = (solver.coefficient_matrix() @ tpl.group_matrix())[:, :PSXMCoils.N_COILS]
    I, *_ = np.linalg.lstsq(K, solver.target_field(), rcond=None)
    return CurrentSolver.normalize_currents(I, MAX_CURRENT)




def shield_zero_solver(tpl, gap_mm=SAMPLE_GAP_MM, outer_mm=OUTER_MM, n_between=N_BETWEEN):
    """Solver with B=0 samples on (shield+gap_mm) and (shield+outer_mm)
    rings, placed in the azimuthal gaps between shield points.

    gap_mm / outer_mm / n_between default to the module constants but are
    exposed as arguments so callers (chi2_scan.py, verify_ls.py, ...) can
    scan them without re-deriving this sampling layout themselves."""
    solver = CurrentSolver.from_current_source(tpl)
    dphi = 360.0 / tpl.shield_n
    for radius in (tpl.shield_radius + gap_mm, tpl.shield_radius + outer_mm):
        for base in tpl.shield_angles:
            for j in range(1, n_between + 1):
                a = np.radians(base + j * dphi / (n_between + 1))
                solver.add_sample_point(radius * np.cos(a), radius * np.sin(a), 0.0, 0.0)
    return solver


def net_current_fraction(I):
    """|sum(I)| / sum|I|: how far a current vector is from zero net current.

    The exact exterior cancellation of the ideal two-dimensional shell
    (Sec. 2.4 of the report) requires the enclosed axial current to have
    zero net value, for the coil legs by construction and for the solved
    shield sheet as a numerical property.  The plain least-squares solve
    carries no equality constraint for this; the check below reports how
    well the solution satisfies it anyway.
    """
    denom = float(np.sum(np.abs(I)))
    if denom == 0.0:
        return 0.0
    return float(np.abs(np.sum(I))) / denom


def ls_shield_currents(tpl, I_coil, gap_mm=SAMPLE_GAP_MM, outer_mm=OUTER_MM, n_between=N_BETWEEN):
    """Least-squares response shield currents I_s = S @ I_coil,
    S = -K_s^+ K_6 from nulling B on the offset rings."""
    KM = (shield_zero_solver(tpl, gap_mm, outer_mm, n_between).coefficient_matrix()
          @ tpl.group_matrix())
    K6, Ksh = KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:]
    X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
    I_sh = (-X) @ I_coil
    eps_I = net_current_fraction(I_sh)
    # 1e-5 relative net current is still far below every reported exterior
    # residual; the threshold only guards against a genuinely unbalanced
    # sheet (e.g. a broken sampling layout).
    if eps_I > 1e-5:
        print(f"[ls_shield_currents] WARNING: shield net-current fraction "
              f"eps_I = {eps_I:.2e} exceeds 1e-5; the exact-cancellation "
              f"argument of the report does not apply to this solution.")
    return I_sh


def build_pair(tpl, I_coil, I_shield):
    """(shielded, unshielded) PSXMCoils with the same coil currents."""
    kw = dict(radius=tpl.radius, coil_length=tpl.coil_length)
    shielded = PSXMCoils(currents=I_coil, shield=True, shield_radius=tpl.shield_radius,
                         shield_n=tpl.shield_n, shield_currents=I_shield, **kw)
    return shielded, PSXMCoils(currents=I_coil, **kw)


# --- measurement / reporting ---------------------------------------------
def ring_meanB(coils, rho, n=96):
    """Mean |B| (T) on a circle of radius rho (mm), skipping conductor hits."""
    vals = []
    for a in np.linspace(0.017, 2 * np.pi + 0.017, n, endpoint=False):
        try:
            vals.append(coils.B_magnitude(rho * np.cos(a), rho * np.sin(a)))
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else np.nan
