"""shield_common.py -- shared code for the PSXM shield studies.

One place for everything example.py / shield2D.py (and friends) used to
duplicate: the quadrupole coil solve, the B=0 sample-point layout, the
least-squares shield response, the physical induced-current model, ring-
averaged |B|, the leakage report and the example-style two-panel figure.

All scripts share the same conventions: quadrupole target G [T/mm], coil
currents normalized to MAX_CURRENT, radial axis out to R_MAX_MM.
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver

MU0 = 4e-7 * np.pi

# --- shared configuration -------------------------------------------------
G = 1e-3               # T/mm, central quadrupole gradient target
MAX_CURRENT = 1000.0   # A, coil currents normalized so max|I_coil| = this
SHIELD_N = 200         # shield discretization points
SAMPLE_GAP_MM = 5.0    # radial gap: shield currents -> B=0 sample ring
OUTER_MM = 2.0         # second B=0 ring, this far outside the shield
N_BETWEEN = 3          # B=0 samples between adjacent shield points
R_MAX_MM = 420.0       # radial axis limit (0.42 m ~ nearest beam)
MARKS_MM = (100.0, 419.0)   # benchmark distances to mark/report

# physical (induced-eddy) model parameters
SIGMA_CU = 5.8e7       # S/m copper
D_SHIELD = 2.0e-3      # m copper thickness (nominal)
T_PULSE = 1.0e-6       # s pulse width
M_MODES = 10


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

def solve_dipole_coils(tpl, B0=1e-3):
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


def ls_shield_currents(tpl, I_coil, gap_mm=SAMPLE_GAP_MM, outer_mm=OUTER_MM, n_between=N_BETWEEN):
    """Least-squares response shield currents I_s = S @ I_coil,
    S = -K_s^+ K_6 from nulling B on the offset rings."""
    KM = (shield_zero_solver(tpl, gap_mm, outer_mm, n_between).coefficient_matrix()
          @ tpl.group_matrix())
    K6, Ksh = KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:]
    X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
    return (-X) @ I_coil


def induced_shield_currents(tpl, I_coil, t_pulse=T_PULSE, sigma=SIGMA_CU, d=D_SHIELD):
    """Physical induced eddy currents: multipole moments x thin-shell L/R
    response, sampled onto the shield points. Returns (I_shield, info dict)."""
    omega = np.pi / t_pulse
    delta = np.sqrt(2.0 / (MU0 * sigma * omega))
    d_eff = min(d, delta)
    a_m = tpl.shield_radius * 1e-3

    legs = PSXMCoils(currents=I_coil)
    z = (np.asarray(legs.x) + 1j * np.asarray(legs.y)) * 1e-3
    Ileg = np.asarray(legs.I)
    th = np.radians(tpl.shield_angles)
    K = np.zeros_like(th)
    S2 = None
    for m in range(1, M_MODES + 1):
        C = np.sum(Ileg * z ** m)
        tau = MU0 * sigma * d_eff * a_m / (2 * m)
        f = 1j * omega * tau / (1 + 1j * omega * tau)
        K += -np.real(f * C * np.exp(-1j * m * th)) / (np.pi * a_m ** (m + 1))
        if m == 2:
            S2 = 1.0 / abs(1 + 1j * omega * tau)
    seg = 2 * np.pi * a_m / tpl.shield_n
    return K * seg, {"delta": delta, "d_eff": d_eff, "S2": S2}


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


def leakage_report(shielded, unshielded, marks=MARKS_MM):
    print(f"beam |B| (r=5 mm): {ring_meanB(unshielded, 5.0)*1e3:.3f} mT")
    for d in marks:
        bu, bs = ring_meanB(unshielded, d), ring_meanB(shielded, d)
        print(f"leakage at {d/1000:.6f} m:  no-shield {bu*1e6:10.6f} uT   "
              f"shielded {bs*1e6:10.6f} uT   suppression {bu/bs:.1f}x")


def two_panel(shielded, unshielded, title, fname, marks=MARKS_MM):
    """example-style figure: left = field lines +/-R_MAX with a zoom inset
    showing the shield-current arrows; right = |B| vs radius, both curves."""
    fig, (axf, axl) = plt.subplots(1, 2, figsize=(13.5, 6))

    # left: full extent so the far benchmarks are on-screen
    shielded.draw(axf, n_grid=400, extent=R_MAX_MM / shielded.shield_radius, legend=False)
    axf.set_title(f"{title}  (±{R_MAX_MM:.0f} mm)")
    # inset: zoom on the magnet so the red shield arrows are visible
    ins = axf.inset_axes([0.66, 0.66, 0.33, 0.33])
    shielded.draw(ins, n_grid=150, extent=40.0 / shielded.shield_radius, legend=False)
    ins.set_title("zoom 40 mm", fontsize=7)
    ins.set_xlabel(""); ins.set_ylabel("")
    ins.tick_params(labelsize=6)

    # right: |B| vs radius
    rho = np.geomspace(1.0, R_MAX_MM, 140)
    Bun = np.array([ring_meanB(unshielded, r) for r in rho])
    Bsh = np.array([ring_meanB(shielded, r) for r in rho])
    axl.loglog(rho / 1000, Bun * 1e3, label="no shield")
    axl.loglog(rho / 1000, Bsh * 1e3, label="with shield")
    for xr in (shielded.radius, shielded.shield_radius, *marks):
        axl.axvline(xr / 1000, color="gray", ls="--", lw=0.7)
    axl.set_xlabel("radial distance ρ (m)")
    axl.set_ylabel("ring-averaged |B| (mT)")
    axl.set_title("leakage: |B| vs radius")
    axl.legend()
    axl.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=170, bbox_inches="tight")
    print(f"saved {fname}")
