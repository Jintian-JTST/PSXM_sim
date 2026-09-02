"""verify_ls.py -- validate the least-squares shield calculation.

Three of the tests check the solver against *itself* or against the field
routine it is built from; the fourth checks it against the analytic
ideal-shell solution of the report's Sec. 2.4.

T1 round-trip   : known currents -> field at samples -> LS solve -> must
                  recover the currents exactly (code correctness).
T2 forward check: K @ I  vs  PSXMCoils.B_field at the same points (the
                  matrix and the field routine must agree to float
                  precision).
T3 conditioning : cond(K_shield) and the singular-value spread of the
                  shield response problem (is it well-posed?).
T4 analytic     : the multipole reduction of an ideal cylindrical shell,
                  F_m(R_s) = F_bare [1 - (R/R_s)^(2m)], must be reproduced
                  by the LS solve (external analytic cross-check).

Run:  python verify_ls.py     (prints a PASS/FAIL summary)
"""

import numpy as np

from psxm_coils import PSXMCoils
from current_solver import CurrentSolver
from shield import (shield_zero_solver, solve_quad_coils,
                           solve_dipole_coils, ls_shield_currents,
                           build_pair)
from field_analysis import multipoles

I_COIL = 1000.0
SAMPLE_GAP_MM = 5.0      # radial gap between shield currents and B=0 samples
OUTER_MM = 2.0
N_BETWEEN = 3


# ---------------------------------------------------------------- helpers
def quad_currents():
    return np.array([0, I_COIL, I_COIL, 0, -I_COIL, -I_COIL], float)


def build_shield_solver(tpl, gap_mm=SAMPLE_GAP_MM):
    """B=0 samples on (shield+gap) and (shield+OUTER) rings, in the azimuthal
    gaps between shield points (shared layout, see shield). Returns
    the K blocks plus the solver itself (so callers can reuse its exact
    sample points instead of rebuilding them)."""
    solver = shield_zero_solver(tpl, gap_mm=gap_mm, outer_mm=OUTER_MM, n_between=N_BETWEEN)
    KM = solver.coefficient_matrix() @ tpl.group_matrix()
    return KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:], solver


# ---------------------------------------------------------------- tests
def t1_roundtrip():
    """Known coil currents -> exact fields as targets -> LS must recover them."""
    tpl = PSXMCoils(currents=np.zeros(6))
    I_true = np.array([123.0, -456.0, 789.0, 321.0, -654.0, 987.0])
    truth = PSXMCoils(currents=I_true)
    solver = CurrentSolver.from_current_source(tpl)
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        x, y = 8.0 * np.cos(a), 8.0 * np.sin(a)
        Bx, By = truth.B_field(x, y)
        solver.add_sample_point(x, y, Bx, By)
    I_rec = solver.solve()
    err = np.max(np.abs(I_rec - I_true)) / np.max(np.abs(I_true))
    ok = err < 1e-9
    print(f"T1 round-trip recovery : max rel err = {err:.2e}   {'PASS' if ok else 'FAIL'}")
    return ok


def t2_forward():
    """K @ I must equal PSXMCoils.B_field at the same sample points."""
    tpl = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=60)
    K6, Ksh, solver = build_shield_solver(tpl)
    I_c = quad_currents()
    rng = np.random.default_rng(0)
    I_s = rng.normal(0, 100, tpl.shield_n)
    # solver.sample_x/sample_y ARE the exact points K6/Ksh were built from --
    # no need to separately rebuild them to cross-check against B_field.
    pts = list(zip(solver.sample_x, solver.sample_y))
    coils = PSXMCoils(currents=I_c, shield=True, shield_n=tpl.shield_n,
                      shield_currents=I_s)
    B_direct = np.array([coils.B_field(x, y) for x, y in pts])   # (n,2)
    B_matrix = K6 @ I_c + Ksh @ I_s                              # stacked [Bx..., By...]
    err = np.max(np.abs(np.concatenate([B_direct[:, 0], B_direct[:, 1]]) - B_matrix))
    scale = np.max(np.abs(B_matrix))
    ok = err / scale < 1e-10
    print(f"T2 forward consistency : max rel err = {err/scale:.2e}   {'PASS' if ok else 'FAIL'}")
    return ok


def t3_conditioning():
    tpl = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=100)
    _, Ksh, _ = build_shield_solver(tpl)
    s = np.linalg.svd(Ksh, compute_uv=False)
    cond = s[0] / s[-1]
    print(f"T3 conditioning        : cond(K_shield) = {cond:.2e}  "
          f"(sv range {s[-1]:.2e} .. {s[0]:.2e})")
    print( "                         (>1e10 would mean ill-posed; large but finite is OK with lstsq)")
    return True


def t4_analytic_ideal_shell():
    """LS shield solve must reproduce the analytic ideal-shell multipole
    reduction F_m(R_s) = F_bare * [1 - (R/R_s)^(2m)] (report Sec. 2.4).

    This is the cross-check against an external analytic model that the
    other tests deliberately avoid: the bare ring numbers are the m = 1
    (dipole) and m = 2 (quadrupole) multipole amplitudes measured by the
    same fitting convention the report uses, and the shield radius enters
    only through the analytic factor.
    """
    R, Rs = 22.5, 27.5
    bare = PSXMCoils(currents=np.zeros(6))

    # bare-ring multipole amplitudes, measured with the report's convention
    dip_bare = multipoles(PSXMCoils(
        currents=solve_dipole_coils(bare)))["B0mag"] * 1e3
    quad_bare = multipoles(PSXMCoils(
        currents=solve_quad_coils(bare)))["Gmag"] * 1e3

    # analytic predictions
    dip_ana = dip_bare * (1 - (R / Rs) ** 2)
    quad_ana = quad_bare * (1 - (R / Rs) ** 4)

    # numerical: same solves the report uses, normalized to 1000 A
    Iq = solve_quad_coils(bare)             # normalized inside
    Id = solve_dipole_coils(bare)
    tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                    shield_radius=Rs, shield_n=200)
    Iq_sh = solve_quad_coils(tpl)
    Id_sh = solve_dipole_coils(tpl)
    oq, _ = build_pair(tpl, Iq_sh, ls_shield_currents(tpl, Iq_sh))
    od, _ = build_pair(tpl, Id_sh, ls_shield_currents(tpl, Id_sh))
    dip_num = multipoles(od)["B0mag"] * 1e3
    quad_num = multipoles(oq)["Gmag"] * 1e3

    ok = (abs(dip_num - dip_ana) / dip_ana < 1e-3
          and abs(quad_num - quad_ana) / quad_ana < 1e-3)
    print(f"T4 analytic ideal shell : dipole  {dip_num:.4f} vs analytic "
          f"{dip_ana:.4f} mT   {'PASS' if ok else 'FAIL'}")
    print(f"                         quadrupole {quad_num:.4f} vs analytic "
          f"{quad_ana:.4f} mT/mm")
    return ok


def main():
    checks = (
        t1_roundtrip(),
        t2_forward(),
        t3_conditioning(),
        t4_analytic_ideal_shell(),
    )
    if not all(checks):
        raise SystemExit("least-squares verification FAILED")
    print("All least-squares verification checks PASS")


if __name__ == "__main__":
    main()
