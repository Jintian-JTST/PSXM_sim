"""node5_max_field.py -- maximum reachable central field of the PSXM (node 5).

The design target (1 mT dipole, 1 mT/mm quadrupole) is a requirement, not
a capability.  This script asks the complementary question: with the coil
currents capped at 1000 A, what is the *most* the six-coil ring can do,
and how much of that ceiling does a field-quality-constrained design
actually get?

Two different ceilings are computed and they answer different questions.

*  **Hardware ceiling (linear programme).**  The centre moment is linear
   in the currents, so maximizing it under |I_k| <= I_max is a linear
   programme whose optimum sits on a vertex of the current box:
   I_k = I_max sign(n . c_k).  No field-quality constraint at all -- this
   is the field the magnet could make if the only thing that mattered
   were the value of B at the origin.  Closed form in
   ``node5_common.analytic_ceilings``, reproduced numerically here.

*  **Design ceiling (least squares).**  The currents that best reproduce
   a *pure* dipole or quadrupole on the 1 mm ring, renormalized to
   1000 A.  This is what a usable magnet delivers, because the muon sees
   the whole aperture and not just the origin.

Their ratio is the price of field quality.  Adding the shield can costs a
further, much larger, factor -- which is exactly the shield-radius
trade-off studied in the main report.

Run:  python node5_max_field.py
Outputs: figures/node5_max_field.png,
         ../PSXM_design_report/results_maxfield.tex
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver
from shield_common import (G, MAX_CURRENT, SHIELD_N, solve_quad_coils,
                           solve_dipole_coils, ls_shield_currents, build_pair)
from node5_common import (multipoles, uniformity, analytic_ceilings,
                          unit_response, lp_ceiling, save_fig, write_macros)

B0_TARGET = 1.0e-3                      # T
PSI = np.arange(0.0, 120.0 + 1e-9, 0.5)  # deg, dipole direction sweep
# shield radii to report: the bare ring, the default can, and the two
# optima derived in the main report
R_CASES = (None, 27.5, 26.233, 23.5)


def solve_rot_dipole(tpl, psi_deg, B0=B0_TARGET):
    """Coil currents (A, unnormalized) for a uniform dipole along psi_deg."""
    solver = CurrentSolver.from_current_source(tpl)
    bx, by = B0 * np.cos(np.radians(psi_deg)), B0 * np.sin(np.radians(psi_deg))
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        solver.add_sample_point(np.cos(a), np.sin(a), Bx=bx, By=by)
    K = (solver.coefficient_matrix() @ tpl.group_matrix())[:, :PSXMCoils.N_COILS]
    I, *_ = np.linalg.lstsq(K, solver.target_field(), rcond=None)
    return I


def achieved(mode, shield_radius):
    """Field a least-squares design achieves at the 1000 A operating point.

    ``shield_radius=None`` means the bare coil ring.  Returns
    (value, dict-of-multipoles, peak shield current).
    """
    if shield_radius is None:
        tpl = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS))
        Ic = solve_quad_coils(tpl) if mode == "quad" else solve_dipole_coils(tpl)
        obj, Ish = PSXMCoils(currents=Ic), np.zeros(1)
    else:
        tpl = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS), shield=True,
                        shield_radius=shield_radius, shield_n=SHIELD_N)
        Ic = solve_quad_coils(tpl) if mode == "quad" else solve_dipole_coils(tpl)
        Ish = ls_shield_currents(tpl, Ic)
        obj, _ = build_pair(tpl, Ic, Ish)
    m = multipoles(obj)
    return (m["Gmag"] if mode == "quad" else m["B0mag"]), m, float(np.max(np.abs(Ish)))


def main():
    ceil = analytic_ceilings()
    resp = unit_response()

    # --- 1. hardware ceiling, analytic vs numerical -----------------------
    psi = np.radians(PSI)
    lp_dip = lp_ceiling(resp[:, 0:2], psi) * 1e3            # mT
    lp_quad = lp_ceiling(resp[:, 2:4], psi) * 1e3           # mT/mm
    print("hardware ceiling at 1000 A (no field-quality constraint)")
    print(f"  dipole      analytic  best {ceil['dipole_best']*1e3:7.3f} mT   "
          f"worst {ceil['dipole_worst']*1e3:7.3f} mT")
    print(f"              numerical best {lp_dip.max():7.3f} mT   "
          f"worst {lp_dip.min():7.3f} mT")
    # note: analytic_ceilings returns the gradient in T/m, which IS mT/mm
    print(f"  quadrupole  analytic  best {ceil['quad_best']:7.4f} mT/mm "
          f"worst {ceil['quad_worst']:7.4f} mT/mm")
    print(f"              numerical best {lp_quad.max():7.4f} mT/mm "
          f"worst {lp_quad.min():7.4f} mT/mm")
    print(f"  ripple 2/sqrt(3) = {ceil['ripple']:.4f}, "
          f"numerical {lp_dip.max()/lp_dip.min():.4f}\n")

    # --- 2. design ceiling: least-squares, direction sweep ----------------
    bare = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS))
    ls_dip = []
    for p in PSI:
        I = solve_rot_dipole(bare, p)
        I = I * (MAX_CURRENT / np.max(np.abs(I)))
        ls_dip.append(multipoles(PSXMCoils(currents=I))["B0mag"] * 1e3)
    ls_dip = np.array(ls_dip)

    # --- 3. design ceiling vs shield radius --------------------------------
    print(f"{'shield':>9} | {'dipole (mT)':>12} {'uniform@10mm':>13} | "
          f"{'quad (mT/mm)':>13} {'I_shield (A)':>13}")
    table = []
    for R in R_CASES:
        bd, md, ishd = achieved("dipole", R)
        bq, mq, ishq = achieved("quad", R)
        obj = (PSXMCoils(currents=solve_dipole_coils(bare)) if R is None else None)
        u = uniformity(obj)[10.0] if obj is not None else np.nan
        label = "bare" if R is None else f"{R:g} mm"
        table.append((label, bd * 1e3, bq * 1e3, ishd, ishq, u))
        print(f"{label:>9} | {bd*1e3:12.4f} {u:13.4f} | "
              f"{bq*1e3:13.4f} {max(ishd, ishq):13.1f}")

    dip_bare, quad_bare = table[0][1], table[0][2]
    price_dip = dip_bare / (ceil["dipole_best"] * 1e3)
    price_quad = quad_bare / ceil["quad_best"]
    print(f"\nprice of field quality (bare ring): "
          f"dipole {100*price_dip:.1f}% of ceiling, "
          f"quadrupole {100*price_quad:.1f}% of ceiling")
    print(f"headroom over the 1 mT / 1 mT/mm specification: "
          f"dipole x{dip_bare/1.0:.1f}, quadrupole x{quad_bare/1.0:.1f} (bare ring)")

    # --- figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

    ax[0].plot(PSI, lp_dip, "k--", lw=1.2, label="hardware ceiling (LP)")
    ax[0].plot(PSI, ls_dip, lw=1.5, label="least-squares design")
    ax[0].axhline(1.0, color="tab:red", ls=":", lw=1.0, label="1 mT specification")
    ax[0].set_xlabel(r"dipole direction $\psi$ (deg)")
    ax[0].set_ylabel("centre $|B|$ at 1000 A (mT)")
    ax[0].set_title("dipole: ceiling vs. usable field")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(PSI, lp_quad, "k--", lw=1.2, label="hardware ceiling (LP)")
    ax[1].axhline(quad_bare, color="tab:blue", lw=1.4,
                  label="least-squares, bare ring")
    for lbl, _, q, _, _, _ in table[1:]:
        ax[1].axhline(q, lw=1.1, ls="-.", label=f"with shield {lbl}")
    ax[1].axhline(1.0, color="tab:red", ls=":", lw=1.0, label="1 mT/mm specification")
    ax[1].set_xlabel(r"quadrupole axis angle $2\varphi$ (deg)")
    ax[1].set_ylabel("centre $|G|$ at 1000 A (mT/mm)")
    ax[1].set_title("quadrupole: ceiling vs. usable gradient")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    labels = [t[0] for t in table]
    xs = np.arange(len(labels))
    ax[2].bar(xs - 0.2, [t[1] for t in table], 0.4, label="dipole (mT)")
    ax[2].bar(xs + 0.2, [t[2] for t in table], 0.4, label="quadrupole (mT/mm)")
    ax[2].axhline(1.0, color="tab:red", ls=":", lw=1.0, label="specification")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(labels)
    ax[2].set_yscale("log")
    ax[2].set_ylabel("achieved at 1000 A")
    ax[2].set_title("what the shield costs the centre")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="y", which="both")

    fig.suptitle("Maximum reachable central field: hardware ceiling, "
                 "field-quality cost, and shield cost")
    fig.tight_layout()
    save_fig(fig, "node5_max_field.png")

    # --- macros ------------------------------------------------------------
    write_macros("results_maxfield.tex", {
        "MFdipceil": f"{ceil['dipole_best']*1e3:.2f}",
        "MFdipceilw": f"{ceil['dipole_worst']*1e3:.2f}",
        "MFquadceil": f"{ceil['quad_best']:.3f}",
        "MFquadceilw": f"{ceil['quad_worst']:.3f}",
        "MFripple": f"{100*(ceil['ripple']-1):.1f}",
        "MFdipbare": f"{dip_bare:.2f}",
        "MFquadbare": f"{quad_bare:.3f}",
        "MFpricedip": f"{100*price_dip:.0f}",
        "MFpricequad": f"{100*price_quad:.0f}",
        "MFdipdefault": f"{table[1][1]:.3f}",
        "MFquaddefault": f"{table[1][2]:.3f}",
        "MFunif": f"{100*table[0][5]:.1f}",
    })


if __name__ == "__main__":
    main()
