"""report_conditioning.py -- conditioning and net-current checks of the shield solve.

Two reviewer-facing robustness checks for the design report:

1.  Zero-net-current property of the whole response operator.
    The exact exterior cancellation of the ideal two-dimensional shell
    (Sec. 2.4 of the report) requires zero net enclosed axial current.
    The baseline solution satisfies this numerically (eps_I = 1.4e-16),
    but that could be a coincidence of the quadrupole current pattern.
    This script checks the property column by column for the response
    matrix S = -K_s^+ K_6 (one unit current in each of the six main-coil
    degrees of freedom), so the check covers every current configuration
    expressible through the six coil channels, not just the baseline one.

2.  Conditioning of the least-squares inverse.
    The exterior residual of the discrete ideal sheet is known to span
    many orders of magnitude with the sampling layout (Sec. 5.3), so a
    reviewer will ask whether the inversion is ill-conditioned.  The
    answer that matters is not the condition number itself but whether
    the observables the report draws conclusions from -- the delivered
    central gradient G, the required shield sheet current K_s, and the
    threshold radius R_s,min -- are stable against the regularisation of
    the inverse.  This script repeats the baseline quadrupole solve with
    progressively more aggressive rcond cutoffs of the SVD and records
    those observables together with the exterior residual (which is
    expected to move, since it is a discretisation residual).

Run:  python report_conditioning.py
      (writes ../PSXM_design_report/results_cond.tex, prints a summary)
"""

import os

import numpy as np

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver
from shield_common import (SHIELD_N, G as G_TARGET, solve_quad_coils,
                           solve_dipole_coils, build_pair, ring_meanB,
                           shield_zero_solver, net_current_fraction)
from node5_common import multipoles

MAX_CURRENT = 1000.0
G_REQ = 1.0            # mT/mm, quadrupole working benchmark
RS_BASE = 27.5         # mm, default shield radius
DIAG_FAR = 419.0       # mm, far diagnostic radius
RCDS = (1e-12, 1e-8, 1e-4, 1e-2)   # rcond cutoffs to scan
N_BETWEEN = 3


def response_operator(tpl):
    """Return (K6, Ksh) blocks of the baseline zero-field layout."""
    KM = (shield_zero_solver(tpl, gap_mm=5.0, outer_mm=2.0,
                             n_between=N_BETWEEN).coefficient_matrix()
          @ tpl.group_matrix())
    return KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:]


def shield_response(tpl, rcond=None):
    """S = -K_s^+ K_6 for the baseline layout, with the given rcond cutoff."""
    K6, Ksh = response_operator(tpl)
    X, *_ = np.linalg.lstsq(Ksh, K6, rcond=rcond)
    return -X, K6, Ksh


def achieved_residual(tpl, Ic):
    """dimensionless residual eps_W (report Eq. 20) of the baseline layout."""
    K6, Ksh = response_operator(tpl)
    S, _, _ = shield_response(tpl, rcond=None)
    Ish = S @ Ic
    B = K6 @ Ic + Ksh @ Ish
    norm = float(np.max(np.abs(K6 @ Ic)))
    return float(np.sqrt(np.mean(np.abs(B) ** 2))) / norm


def design_measure(tpl, Ic, Ish):
    """Central gradient (mT/mm), sheet current (A/mm), far residual (uT)."""
    sh, _ = build_pair(tpl, Ic, Ish)
    G = multipoles(sh)["Gmag"] * 1e3
    Ksheet = float(np.max(np.abs(Ish))) * tpl.shield_n / (2 * np.pi * tpl.shield_radius)
    leak = ring_meanB(sh, DIAG_FAR) * 1e6
    return G, Ksheet, leak


def threshold_rcond(rcond, lo=22.6, hi=45.0, tol=1e-4):
    """Smallest R_s whose solve with this rcond still delivers G_REQ."""
    def g(R):
        tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                        shield_radius=R, shield_n=SHIELD_N)
        Ic = solve_quad_coils(tpl)
        S, _, _ = shield_response(tpl, rcond)
        sh, _ = build_pair(tpl, Ic, S @ Ic)
        return multipoles(sh)["Gmag"] * 1e3
    assert g(lo) < G_REQ < g(hi), (g(lo), g(hi))
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if g(mid) < G_REQ:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                    shield_radius=RS_BASE, shield_n=SHIELD_N)
    Ic = solve_quad_coils(tpl)
    epsW_quad = achieved_residual(tpl, Ic)
    epsW_dip = achieved_residual(tpl, solve_dipole_coils(tpl))

    # --- 1. net-current property of the response operator -----------------
    S, K6, Ksh = shield_response(tpl, rcond=None)
    col_eps = [net_current_fraction(S[:, k]) for k in range(PSXMCoils.N_COILS)]

    # --- 2. singular-value record of the inverted block --------------------
    sv = np.linalg.svd(Ksh, compute_uv=False)
    svmax, svmin = sv[0], sv[-1]
    kappa_sh = svmax / svmin
    # coil-only design solve (12 samples on the 1 mm ring, 6 DOF)
    solver = CurrentSolver.from_current_source(tpl)
    for a in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        x, y = np.cos(a), np.sin(a)
        solver.add_sample_point(x, y, Bx=G_TARGET * y, By=G_TARGET * x)
    Kc = (solver.coefficient_matrix() @ tpl.group_matrix())[:, :PSXMCoils.N_COILS]
    svc = np.linalg.svd(Kc, compute_uv=False)
    kappa_coil = svc[0] / svc[-1]
    rank = {r: int(np.sum(sv >= r * svmax)) for r in RCDS}

    # --- 3. rcond sensitivity of the design observables --------------------
    print(f"{'rcond':>8} {'G mT/mm':>10} {'Ksheet A/mm':>13} "
          f"{'R_s,min mm':>11} {'resid@419 uT':>14}")
    rows = []
    for r in RCDS:
        Sr, _, _ = shield_response(tpl, r)
        Ish = Sr @ Ic
        G, Ksheet, leak = design_measure(tpl, Ic, Ish)
        thr = threshold_rcond(r)
        rows.append((r, G, Ksheet, thr, leak))
        print(f"{r:8.0e} {G:10.4f} {Ksheet:13.3f} {thr:11.3f} {leak:14.2e}")

    # baseline (rcond=None) for comparison
    Ish = S @ Ic
    G0, Ksheet0, leak0 = design_measure(tpl, Ic, Ish)
    thr0 = threshold_rcond(None)
    print(f"{'None':>8} {G0:10.4f} {Ksheet0:13.3f} {thr0:11.3f} "
          f"{leak0:14.2e}   <- baseline")

    print("\ncondition number of K_s (shield block):  %.3e" % kappa_sh)
    print("singular values of K_s:  %.3e ... %.3e T/A" % (svmax, svmin))
    print("condition number of the 6-coil design solve:  %.3e" % kappa_coil)
    print("singular values kept for rcond = 1e-12 / 1e-8 / 1e-4 / 1e-2:",
          [rank[r] for r in RCDS], "of", len(sv))
    print("\nper-column net-current fraction of S (k = 1..6):",
          ["%.2e" % e for e in col_eps])
    print("maximum:", "%.2e" % max(col_eps))

    sci = lambda v: "\\num{%.1e}" % v
    macros = {
        "CONDkappash": "\\num{%.2e}" % kappa_sh,
        "CONDsvmax": "\\num{%.2e}" % svmax,
        "CONDsvmin": "\\num{%.2e}" % svmin,
        "CONDrankA": "%d" % rank[1e-12], "CONDrankB": "%d" % rank[1e-8],
        "CONDrankC": "%d" % rank[1e-4], "CONDrankD": "%d" % rank[1e-2],
        "CONDnetlist": ", ".join("%.1e" % e for e in col_eps),
        "CONDnetmax": "\\num{%.1e}" % max(col_eps),
        # baseline (rcond=None) row, same layout as the A-D rows
        "CONDGNone": "%.4f" % G0,
        "CONDKNone": "%.3f" % Ksheet0,
        "CONDthrNone": "%.3f" % thr0,
        "CONDresNone": sci(leak0),
        # dimensionless achieved residual eps_W (Eq. 20), baseline layout
        "EPSWquad": "\\num{%.2e}" % epsW_quad,
        "EPSWdip": "\\num{%.2e}" % epsW_dip,
    }
    for r, (rc, G, Ksheet, thr, leak) in zip("ABCD", rows):
        macros["CONDrcond" + r] = sci(rc)
        macros["CONDG" + r] = "%.4f" % G
        macros["CONDK" + r] = "%.3f" % Ksheet
        macros["CONDthr" + r] = "%.3f" % thr
        macros["CONDres" + r] = sci(leak)

    out = os.path.join("..", "PSXM_design_report", "results_cond.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("% auto-generated by report_conditioning.py -- do not edit\n")
        for k, v in macros.items():
            f.write("\\renewcommand{\\%s}{%s}\n" % (k, v))
    print("wrote", out)


if __name__ == "__main__":
    main()
