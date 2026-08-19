"""report_robust.py -- robustness of the shield-radius conclusion.

Two tests that the earlier scans did not cover.

1.  Does the *sign* of the radius dependence of the leakage survive a change
    of the sampling layout?  The absolute leakage of the inverse design is
    known to depend strongly on the radius at which B = 0 is imposed
    (report_figures.fig_sampling), so the shield-radius argument may only be
    used if the trend itself does not.  The scan below repeats the whole
    R_s scan for several radial offsets of the two zero-field rings and
    checks d(leak)/d(R_s) > 0 over the feasible region, together with the
    radius at which the 1 mT/mm target becomes deliverable.

2.  Does the threshold radius survive replacing the (non-physical)
    free-space constraint by the ideal-conductor boundary condition
    B_n = 0 on the shield surface itself?  The baseline pipeline imposes the
    full vector B = 0 on two rings standing off in free space and imposes
    nothing at all on the shield surface; this is the closest the present
    formulation can get to a passive shield.

Run:  python report_robust.py     (writes robust_scan.npz, prints a summary)
"""

import numpy as np

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver
from shield_common import (SHIELD_N, solve_quad_coils, build_pair, ring_meanB,
                           shield_zero_solver)
from node5_common import multipoles

MAX_CURRENT = 1000.0
G_REQ = 1.0            # mT/mm
DIAG = (200.0, 419.0)   # near radius = published 0.2 m distance; far = nearest beam
N_BETWEEN = 3


# =========================================================================
# helpers
# =========================================================================
def _blocks(solver, tpl):
    KM = solver.coefficient_matrix() @ tpl.group_matrix()
    return KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:]


def shield_currents_freespace(tpl, I_coil, dr_outer, dr_inner, n_between=N_BETWEEN):
    """Baseline formulation: full vector B = 0 on two rings in free space."""
    solver = shield_zero_solver(tpl, gap_mm=dr_outer, outer_mm=dr_inner,
                                n_between=n_between)
    K6, Ksh = _blocks(solver, tpl)
    X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
    return (-X) @ I_coil


def shield_currents_bn(tpl, I_coil, n_between=N_BETWEEN, outer_mm=None):
    """Ideal-conductor formulation: B_n = 0 on the shield surface.

    The normal-component row at a surface point (angle theta) is
    cos(theta) * (Bx row) + sin(theta) * (By row) of the coefficient matrix.
    Sample points sit in the azimuthal gaps between shield elements, since a
    sample point on a current point is singular.  If ``outer_mm`` is given,
    the full vector B = 0 is additionally imposed on one ring that far
    outside the shield, which is the mixed formulation.
    """
    a = tpl.shield_radius
    dphi = 360.0 / tpl.shield_n
    solver = CurrentSolver.from_current_source(tpl)
    thetas = []
    for base in tpl.shield_angles:
        for j in range(1, n_between + 1):
            th = np.radians(base + j * dphi / (n_between + 1))
            solver.add_sample_point(a * np.cos(th), a * np.sin(th), 0.0, 0.0)
            thetas.append(th)
    n = len(thetas)
    K6, Ksh = _blocks(solver, tpl)
    c, s = np.cos(thetas), np.sin(thetas)
    # normal-component rows only
    Kn6 = c[:, None] * K6[:n] + s[:, None] * K6[n:]
    Knsh = c[:, None] * Ksh[:n] + s[:, None] * Ksh[n:]

    if outer_mm is not None:
        solver2 = CurrentSolver.from_current_source(tpl)
        r = a + outer_mm
        for base in tpl.shield_angles:
            for j in range(1, n_between + 1):
                th = np.radians(base + j * dphi / (n_between + 1))
                solver2.add_sample_point(r * np.cos(th), r * np.sin(th), 0.0, 0.0)
        K6b, Kshb = _blocks(solver2, tpl)
        Kn6 = np.vstack([Kn6, K6b])
        Knsh = np.vstack([Knsh, Kshb])

    X, *_ = np.linalg.lstsq(Knsh, Kn6, rcond=None)
    return (-X) @ I_coil


def measure(tpl, Ic, Ish):
    sh, un = build_pair(tpl, Ic, Ish)
    m = multipoles(sh)
    out = dict(G=m["Gmag"] * 1e3, B0=m["B0mag"] * 1e3,
               Ish=float(np.max(np.abs(Ish))))
    for key, d in zip(("near", "far"), DIAG):
        out["leak" + key] = ring_meanB(sh, d) * 1e6
        out["bare" + key] = ring_meanB(un, d) * 1e6
    return out


def scan(R_list, mode="freespace", dr_outer=5.0, dr_inner=2.0, outer_mm=None):
    rows = []
    for R in R_list:
        tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                        shield_radius=R, shield_n=SHIELD_N)
        Ic = solve_quad_coils(tpl)
        if mode == "freespace":
            Ish = shield_currents_freespace(tpl, Ic, dr_outer, dr_inner)
        else:
            Ish = shield_currents_bn(tpl, Ic, outer_mm=outer_mm)
        d = measure(tpl, Ic, Ish)
        d["R"] = R
        rows.append(d)
    return rows


def threshold(rows, key="G", target=G_REQ):
    """Linear interpolation of the radius at which ``key`` reaches target."""
    R = np.array([r["R"] for r in rows])
    v = np.array([r[key] for r in rows])
    i = np.argmax(v >= target)
    if i == 0 or not np.any(v >= target):
        return np.nan
    R0, R1, v0, v1 = R[i - 1], R[i], v[i - 1], v[i]
    return float(R0 + (target - v0) * (R1 - R0) / (v1 - v0))


def monotone_above(rows, R_min, key="leakfar"):
    R = np.array([r["R"] for r in rows])
    v = np.array([r[key] for r in rows])
    sel = R >= R_min
    return bool(np.all(np.diff(v[sel]) > 0))


# =========================================================================
def main():
    R_list = np.round(np.arange(22.8, 60.01, 0.6), 3)
    DRS = (2.0, 5.0, 10.0, 20.0)        # outer-ring offset; inner at dr/2

    print("=== 1. leakage trend versus the sampling layout "
          "(quadrupole, 1000 A) ===")
    fs = {}
    for dr in DRS:
        rows = scan(R_list, "freespace", dr_outer=dr, dr_inner=dr / 2.0)
        fs[dr] = rows
        thr = threshold(rows)
        mono100 = monotone_above(rows, 26.0, "leaknear")
        mono419 = monotone_above(rows, 26.0, "leakfar")
        l1 = [r for r in rows if r["R"] >= 26.0][0]
        l2 = rows[-1]
        print(f"  dr_outer={dr:5.1f} mm | G=1 mT/mm at R_s={thr:7.3f} mm | "
              f"leak@419 {l1['leakfar']:9.2e} -> {l2['leakfar']:9.2e} uT "
              f"(26 -> 60 mm) | monotone above 26 mm: "
              f"100mm {mono100}, 419mm {mono419}")

    # the pipeline default layout (5 mm / 2 mm), for reference
    base = scan(R_list, "freespace", dr_outer=5.0, dr_inner=2.0)
    print(f"  pipeline default 5.0/2.0 mm | G=1 mT/mm at "
          f"R_s={threshold(base):7.3f} mm | monotone above 26 mm: "
          f"100mm {monotone_above(base, 26.0, 'leaknear')}, "
          f"419mm {monotone_above(base, 26.0, 'leakfar')}")

    print("\n=== 2. ideal-conductor boundary condition B_n = 0 ===")
    bn = scan(R_list, "bn")
    print(f"  B_n=0 on the shield surface only        | "
          f"G=1 mT/mm at R_s={threshold(bn):7.3f} mm")
    bn_mix = scan(R_list, "bn", outer_mm=2.0)
    print(f"  B_n=0 on the surface + B=0 at +2 mm     | "
          f"G=1 mT/mm at R_s={threshold(bn_mix):7.3f} mm")
    for lbl, rows in (("B_n=0 only", bn), ("B_n=0 + outer", bn_mix)):
        r27 = [r for r in rows if abs(r["R"] - 27.6) < 0.31][0]
        print(f"    {lbl:14s} at R_s={r27['R']:.1f} mm: G={r27['G']:.3f} mT/mm, "
              f"I_s^max={r27['Ish']:6.1f} A, leak@419={r27['leakfar']:.2e} uT, "
              f"suppression={r27['barefar']/r27['leakfar']:.2e}, "
              f"monotone above 26 mm: {monotone_above(rows, 26.0, 'leakfar')}")

    keys = ("R", "G", "Ish", "leaknear", "leakfar", "barenear", "barefar")
    np.savez("robust_scan.npz",
             **{f"dr{int(dr)}": np.array([[r[k] for k in keys] for r in fs[dr]])
                for dr in DRS},
             base=np.array([[r[k] for k in keys] for r in base]),
             bn=np.array([[r[k] for k in keys] for r in bn]),
             bnmix=np.array([[r[k] for k in keys] for r in bn_mix]),
             drs=np.array(DRS))
    print("\nwrote robust_scan.npz")


if __name__ == "__main__":
    main()


# =========================================================================
# precise thresholds and the ideal-limit demonstration, for the report
# =========================================================================
def threshold_bisect(dr_outer, dr_inner, target=G_REQ, lo=22.6, hi=45.0, tol=1e-4):
    def g(R):
        tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                        shield_radius=R, shield_n=SHIELD_N)
        Ic = solve_quad_coils(tpl)
        Ish = shield_currents_freespace(tpl, Ic, dr_outer, dr_inner)
        sh, _ = build_pair(tpl, Ic, Ish)
        return multipoles(sh)["Gmag"] * 1e3
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if g(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def discretization_series(R_s=27.5, dr_outer=5.0, dr_inner=2.0,
                          Ns_list=(50, 100, 200, 400)):
    """Exterior residual at 419 mm versus the shield discretization.

    In the two-dimensional ideal-shield limit a closed flux-excluding shell
    cancels the exterior field of an interior source with zero net current
    exactly, so whatever leakage this model reports is the residual of a
    finite-N_s sheet.  This series is the check: the residual falls by ten
    orders of magnitude as the sheet is refined.
    """
    out = []
    for Ns in Ns_list:
        tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                        shield_radius=R_s, shield_n=Ns)
        Ic = solve_quad_coils(tpl)
        Ish = shield_currents_freespace(tpl, Ic, dr_outer, dr_inner)
        sh, un = build_pair(tpl, Ic, Ish)
        out.append(dict(Ns=Ns, G=multipoles(sh)["Gmag"] * 1e3,
                        leak=ring_meanB(sh, 419.0) * 1e6,
                        suppr=ring_meanB(un, 419.0) / ring_meanB(sh, 419.0),
                        Ksheet=float(np.max(np.abs(Ish))) * Ns / (2 * np.pi * R_s)))
    return out


def flux_exclusion_check(R_s=27.5, Ns=SHIELD_N, n=4096, mmax=4):
    """How much of the m = 2 mode of A_z on the shell each solve leaves.

    Flux exclusion is A_z = const on the shell, so the ratio below measures how
    nearly each formulation enforces the ideal-conductor condition.
    """
    tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                    shield_radius=R_s, shield_n=Ns)
    Ic = solve_quad_coils(tpl)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / Ns

    def m2(obj):
        A = np.array([obj.A_z(R_s * np.cos(t), R_s * np.sin(t)) for t in th])
        return float(np.abs(np.fft.rfft(A)[2]) / n * 2)

    base = m2(PSXMCoils(currents=Ic))
    out = {}
    for lbl, Ish in (("freespace",
                      shield_currents_freespace(tpl, Ic, 5.0, 2.0)),
                     ("bn", shield_currents_bn(tpl, Ic))):
        sh, _ = build_pair(tpl, Ic, Ish)
        out[lbl] = m2(sh) / base
    return out


def write_macros(fname="results_robust.tex"):
    import os
    from node5_common import REPORT_DIR
    os.makedirs(REPORT_DIR, exist_ok=True)
    thr = {dr: threshold_bisect(dr, dr / 2.0) for dr in (2.0, 5.0, 10.0, 20.0)}
    disc = discretization_series()
    fx = flux_exclusion_check()
    sci = lambda v: "\\num{%.0e}" % v
    macros = {
        "RBthr": "%.3f" % thr[5.0],
        "RBthrlist": ", ".join("%.3f" % thr[d] for d in (2.0, 5.0, 10.0, 20.0)),
        "RBthrspread": "%.3f" % (max(thr.values()) - min(thr.values())),
        "RBleaklo": sci(disc[0]["leak"]), "RBleakhi": sci(disc[-1]["leak"]),
        "RBnslo": "%d" % disc[0]["Ns"], "RBnshi": "%d" % disc[-1]["Ns"],
        "RBfluxfree": sci(fx["freespace"]), "RBfluxbn": "%.2f" % fx["bn"],
    }
    p = os.path.join(REPORT_DIR, fname)
    with open(p, "w", encoding="utf-8") as f:
        f.write("% auto-generated by report_robust.py -- do not edit\n")
        for k, v in macros.items():
            f.write("\\renewcommand{\\%s}{%s}\n" % (k, v))
    print("wrote", p)
    return macros
