"""report_scan.py -- shield-radius scan at the fixed 1000 A operating point.

Everything downstream (the trade-off figure, the shield-radius figures and
the report's shield-radius table) is built from this one scan, so the
figures and the table cannot disagree with each other.

For each candidate shield radius the least-squares design is solved once,
normalized so that max|I_coil| = 1000 A, and then measured: the central
dipole and quadrupole content, the peak shield current, and the
ring-averaged leakage at the two benchmark distances.  Nothing is
re-scaled afterwards, so B_measured = K I_reported holds by construction.

Run:  python report_scan.py        (writes scan_1000A.npz, prints the table)
"""

import numpy as np

from psxm_coils import PSXMCoils
from shield import (SHIELD_N, solve_quad_coils, solve_dipole_coils,
                           ls_shield_currents, build_pair, ring_meanB,
                           net_current_fraction)
from field_analysis import multipoles

MAX_CURRENT = 1000.0
G_REQ = 1.0          # mT/mm, quadrupole working benchmark
B_REQ = 1.0          # mT,    dipole working benchmark (central-field criterion)
# The published first field integral BL = 1.0e-3 T.m (Abe et al.) is a
# beam-deflection requirement, not a central field.  Converting it needs an
# effective magnetic length: Abe's 2D design assumed B = 0.01 T over
# L = 0.1 m, and the collaboration's updated material quotes an effective
# length of 0.1 m with 60 G at 15 A (10 mT at the rated 25 A).  Under that
# reading the implied central field is 10 mT.  This is an ILLUSTRATIVE
# sensitivity reading only: the 164 mm physical length of the 3D device is
# NOT an effective magnetic length and must not be used for this
# conversion (the report's Sec. 6.7 explains why the dipole radius stays
# open until the field integral is evaluated in a finite-length model).
B_BL_ILL = 10.0      # mT, dipole central field implied by BL with L_eff = 0.1 m
LEAK_REQ = 1.0       # uT,    published leakage requirement at 0.2 m
DIAG = (200.0, 419.0)   # near radius = published 0.2 m distance; far = nearest beam


def design_at(R, mode, shield_n=SHIELD_N):
    """Least-squares design at shield radius R, at the 1000 A operating point.

    Returns dict with the achieved central field/gradient (mT, mT/mm), the
    peak shield current (A), and shielded / unshielded ring-averaged
    leakage (uT) at the benchmark radii -- all from the same current
    vector.
    """
    tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                    shield_radius=R, shield_n=shield_n)
    Ic = solve_quad_coils(tpl) if mode == "quad" else solve_dipole_coils(tpl)
    Ish = ls_shield_currents(tpl, Ic)
    sh, un = build_pair(tpl, Ic, Ish)
    m = multipoles(sh)
    out = dict(R=R, mode=mode,
               Icoil=float(np.max(np.abs(Ic))),
               Ish=float(np.max(np.abs(Ish))),
               epsI=net_current_fraction(Ish),
               B0=m["B0mag"] * 1e3, G=m["Gmag"] * 1e3,
               purity=m["purity"])
    for key, d in zip(("near", "far"), DIAG):
        out["leak" + key] = ring_meanB(sh, d) * 1e6
        out["bare" + key] = ring_meanB(un, d) * 1e6
    return out


def capability(R, mode):
    d = design_at(R, mode)
    return d["G"] if mode == "quad" else d["B0"]


def smallest_feasible(mode, target, lo=22.60, hi=45.0, tol=1e-4):
    """Smallest shield radius whose 1000 A design still delivers ``target``.

    The central field grows monotonically with the shield radius, so the
    feasible set is a half-line and bisection is exact.
    """
    assert capability(lo, mode) < target < capability(hi, mode)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if capability(mid, mode) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    R_fine = np.round(np.arange(22.8, 60.01, 0.4), 3)
    R_table = (22.6, 23.0, 23.5, 25.0, 26.233, 27.5, 30.0, 40.0, 60.0, 80.0)

    scan = {}
    for mode in ("quad", "dipole"):
        rows = [design_at(R, mode) for R in R_fine]
        scan[mode] = rows
        print(f"--- {mode}: fine scan done ({len(rows)} radii)")

    tab = {mode: [design_at(R, mode) for R in R_table] for mode in ("quad", "dipole")}

    opt = dict(
        quad=smallest_feasible("quad", G_REQ),
        dipole=smallest_feasible("dipole", B_REQ),
        dipole_BL=smallest_feasible("dipole", B_BL_ILL),
    )
    print("\noptimum shield radii (smallest feasible at 1000 A):")
    print(f"  quadrupole, G >= {G_REQ} mT/mm      : {opt['quad']:.3f} mm")
    print(f"  dipole,     B >= {B_REQ} mT         : {opt['dipole']:.3f} mm")
    print(f"  dipole,     B >= {B_BL_ILL} mT (illustrative BL/L_eff) : {opt['dipole_BL']:.3f} mm")

    print("\nquadrupole table at fixed 1000 A:")
    print(f"{'R':>7} {'G mT/mm':>9} {'Ish A':>7} {'leaknear':>11} {'leakfar':>11} {'suppr419':>10}")
    for d in tab["quad"]:
        print(f"{d['R']:7.3f} {d['G']:9.4f} {d['Ish']:7.1f} {d['leaknear']:11.3e} "
              f"{d['leakfar']:11.3e} {d['barefar']/d['leakfar']:10.2e}")
    print("\ndipole table at fixed 1000 A:")
    print(f"{'R':>7} {'B mT':>9} {'Ish A':>7} {'leaknear':>11} {'leakfar':>11} {'suppr419':>10}")
    for d in tab["dipole"]:
        print(f"{d['R']:7.3f} {d['B0']:9.4f} {d['Ish']:7.1f} {d['leaknear']:11.3e} "
              f"{d['leakfar']:11.3e} {d['barefar']/d['leakfar']:10.2e}")

    np.savez("scan_1000A.npz",
             quad=np.array([[d[k] for k in ("R", "G", "B0", "Ish", "leaknear",
                                            "leakfar", "barenear", "barefar")]
                            for d in scan["quad"]]),
             dipole=np.array([[d[k] for k in ("R", "G", "B0", "Ish", "leaknear",
                                              "leakfar", "barenear", "barefar")]
                              for d in scan["dipole"]]),
             quad_tab=np.array([[d[k] for k in ("R", "G", "B0", "Ish", "leaknear",
                                                "leakfar", "barenear", "barefar")]
                                for d in tab["quad"]]),
             dipole_tab=np.array([[d[k] for k in ("R", "G", "B0", "Ish", "leaknear",
                                                  "leakfar", "barenear", "barefar")]
                                  for d in tab["dipole"]]),
             opt=np.array([opt["quad"], opt["dipole"], opt["dipole_BL"]]))
    print("\nwrote scan_1000A.npz")


if __name__ == "__main__":
    main()
