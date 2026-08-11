"""node5_rot_quad.py -- rotational quadrupole for the PSXM (node 5, task 4 extension).

Question: the PSXM must be able to present a quadrupole field at an
arbitrary roll angle phi about the beam axis.  Can the six-coil ring do
that, and what does it cost?

Three things are established here.

1.  **Exact linearity.**  The field-current map B = K I is linear and the
    geometry does not move, so the currents for a quadrupole rotated by
    phi are an exact linear combination of the normal and skew solutions,

        I(phi) = cos(2 phi) I_normal + sin(2 phi) I_skew.

    The script checks this against a full re-solve at every angle; the
    residual should be at solver round-off.  Practically this means the
    magnet needs no new solve to roll the field -- two current patterns
    plus a knob.

2.  **The rotation is continuous, but the amplitude ripples.**  A
    six-coil ring is not azimuthally symmetric, so the current cost of a
    given gradient depends on phi.  The ripple has period 30 deg in phi
    (the quadrupole is spin-2, so a 60 deg coil-lattice symmetry becomes
    30 deg in the field angle) and amplitude 2/sqrt(3) between the best
    and worst angle.

3.  **The shield does not break it.**  The shield response I_s = S I_c is
    itself a fixed linear map, so the shielded solution rotates the same
    way; only the absolute achievable gradient is reduced.

Run:  python node5_rot_quad.py
Outputs: figures/node5_rot_quad_scan.png, figures/node5_rot_quad_fields.png,
         ../PSXM_design_report/results_rotquad.tex
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver
from shield_common import (G, MAX_CURRENT, SHIELD_N, shield_zero_solver,
                           build_pair)
from node5_common import (multipoles, analytic_ceilings, unit_response,
                          lp_ceiling, save_fig, write_macros)

PHI = np.arange(0.0, 90.0 + 1e-9, 1.0)          # deg, commanded roll angle
SHIELD_RADIUS = 27.5                            # mm, default can
PANEL_ANGLES = (0.0, 15.0, 30.0, 45.0)          # deg, field-line panels


def solve_rot_quad(tpl, phi_deg, gradient=G):
    """Coil currents (A, unnormalized) for a quadrupole rolled by phi_deg.

    Target on the 1 mm ring, in the same convention as
    ``shield_common.solve_quad_coils`` but with the quadrupole axes
    rotated mechanically by phi:

        Bx = G (y cos 2phi - x sin 2phi),   By = G (x cos 2phi + y sin 2phi).

    Only the six coil degrees of freedom are solved for (the shield
    columns of the grouping matrix are dropped), so the returned currents
    reproduce the target exactly in the unshielded problem.
    """
    solver = CurrentSolver.from_current_source(tpl)
    c, s = np.cos(2 * np.radians(phi_deg)), np.sin(2 * np.radians(phi_deg))
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x, y = np.cos(a), np.sin(a)
        solver.add_sample_point(x, y,
                                Bx=gradient * (y * c - x * s),
                                By=gradient * (x * c + y * s))
    K = (solver.coefficient_matrix() @ tpl.group_matrix())[:, :PSXMCoils.N_COILS]
    I, *_ = np.linalg.lstsq(K, solver.target_field(), rcond=None)
    return I


def shield_response(tpl):
    """The response matrix S of Eq. (S = -K_s^+ K_6), computed once.

    ``shield_common.ls_shield_currents`` rebuilds this for every call,
    which is wasteful here: S depends only on the geometry, so for a
    scan over roll angle it is computed once and reused.  Doing so also
    makes the point of the section concrete -- the shield's response to a
    rotated field is the *same* matrix applied to rotated currents.
    """
    KM = shield_zero_solver(tpl).coefficient_matrix() @ tpl.group_matrix()
    K6, Ksh = KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:]
    X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
    return -X


def main():
    bare = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS))
    tpl = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS), shield=True,
                    shield_radius=SHIELD_RADIUS, shield_n=SHIELD_N)
    S = shield_response(tpl)

    # --- 1. linearity ----------------------------------------------------
    I_norm = solve_rot_quad(bare, 0.0)      # normal quadrupole
    I_skew = solve_rot_quad(bare, 45.0)     # skew quadrupole (cos90=0, sin90=1)

    rows, lin_err = [], 0.0
    for phi in PHI:
        Ic = solve_rot_quad(bare, phi)
        c, s = np.cos(2 * np.radians(phi)), np.sin(2 * np.radians(phi))
        pred = c * I_norm + s * I_skew
        lin_err = max(lin_err, float(np.max(np.abs(Ic - pred))
                                     / np.max(np.abs(I_norm))))

        # scale to the 1000 A operating point and measure what we made
        scale = MAX_CURRENT / np.max(np.abs(Ic))
        Ic_n = Ic * scale
        made = multipoles(PSXMCoils(currents=Ic_n))

        # same solution with the shield can present
        Ish = S @ Ic_n
        sh, _ = build_pair(tpl, Ic_n, Ish)
        made_sh = multipoles(sh)

        rows.append((phi, Ic_n, made["Gmag"], made["phideg"], made["purity"],
                     made_sh["Gmag"], made_sh["phideg"],
                     float(np.max(np.abs(Ish)))))

    phi = np.array([r[0] for r in rows])
    Icur = np.array([r[1] for r in rows])                 # (n, 6) A
    Gbare = np.array([r[2] for r in rows]) * 1e3          # mT/mm
    phimeas = np.array([r[3] for r in rows])
    purity = np.array([r[4] for r in rows])
    Gsh = np.array([r[5] for r in rows]) * 1e3            # mT/mm
    phimeas_sh = np.array([r[6] for r in rows])
    Ishmax = np.array([r[7] for r in rows])

    # angle error, unwrapped onto (-90, 90] since the quadrupole is spin-2
    ang_err = (phimeas - phi + 90.0) % 180.0 - 90.0
    ang_err_sh = (phimeas_sh - phi + 90.0) % 180.0 - 90.0

    # --- 2. analytic ceiling for comparison ------------------------------
    ceil = analytic_ceilings()
    resp = unit_response()
    psi = 2 * np.radians(phi)
    lp = lp_ceiling(resp[:, 2:4], -psi) * 1e3             # mT/mm

    # --- report ----------------------------------------------------------
    print(f"linearity residual  max|I(phi) - (cos2phi I_N + sin2phi I_S)| "
          f"/ max|I_N| = {lin_err:.3e}\n")
    print(f"{'phi':>6} | {'|G| bare':>9} {'|G| shield':>10} | "
          f"{'ang err':>8} {'purity':>8} {'Ish max':>8}")
    for i in range(0, len(phi), max(1, len(phi) // 19)):
        print(f"{phi[i]:6.1f} | {Gbare[i]:9.4f} {Gsh[i]:10.4f} | "
              f"{ang_err[i]:8.2e} {purity[i]:8.2e} {Ishmax[i]:8.1f}")

    imax, imin = int(np.argmax(Gbare)), int(np.argmin(Gbare))
    print(f"\nbare ring:    |G| max {Gbare[imax]:.4f} mT/mm at phi={phi[imax]:.1f} deg, "
          f"min {Gbare[imin]:.4f} at phi={phi[imin]:.1f} deg, "
          f"ripple {Gbare[imax]/Gbare[imin]:.4f}")
    print(f"with shield:  |G| max {Gsh.max():.4f} mT/mm, min {Gsh.min():.4f} mT/mm, "
          f"shield absorbs {100*(1-Gsh.max()/Gbare[imax]):.1f}% of the centre gradient")
    # analytic_ceilings returns the gradient in T/m, which IS mT/mm
    print(f"analytic LP ceiling: best {ceil['quad_best']:.4f} mT/mm, "
          f"worst {ceil['quad_worst']:.4f} mT/mm, "
          f"ripple {ceil['ripple']:.4f}")

    # --- figures ----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

    for k in range(PSXMCoils.N_COILS):
        ax[0].plot(phi, Icur[:, k], lw=1.2, label=f"$I_{k+1}$")
    ax[0].set_xlabel(r"commanded roll angle $\varphi$ (deg)")
    ax[0].set_ylabel("coil current (A)")
    ax[0].set_title("currents rotate as $\\cos2\\varphi,\\ \\sin2\\varphi$")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)

    ax[1].plot(phi, lp, "k--", lw=1.0, label="hardware ceiling (LP)")
    ax[1].plot(phi, Gbare, lw=1.4, label="least-squares, no shield")
    ax[1].plot(phi, Gsh, lw=1.4, label=f"with shield $R_s$={SHIELD_RADIUS:g} mm")
    ax[1].set_xlabel(r"$\varphi$ (deg)")
    ax[1].set_ylabel("achievable $|G|$ at 1000 A (mT/mm)")
    ax[1].set_title("amplitude ripples with period 30$^\\circ$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].semilogy(phi, np.abs(ang_err) + 1e-18, lw=1.2, label="no shield")
    ax[2].semilogy(phi, np.abs(ang_err_sh) + 1e-18, lw=1.2, label="with shield")
    ax[2].semilogy(phi, purity, lw=1.0, ls=":", label="higher-order content")
    ax[2].set_xlabel(r"$\varphi$ (deg)")
    ax[2].set_ylabel("angle error (deg) / relative residual")
    ax[2].set_title("the rotation is exact")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, which="both")

    fig.suptitle("Rotational quadrupole: the six-coil ring rolls the field "
                 "continuously at a 15.5% amplitude ripple")
    fig.tight_layout()
    save_fig(fig, "node5_rot_quad_scan.png")

    # field-line panels
    fig2, axs = plt.subplots(2, 2, figsize=(11, 11))
    for a, p in zip(axs.ravel(), PANEL_ANGLES):
        Ic = solve_rot_quad(bare, p)
        Ic_n = Ic * (MAX_CURRENT / np.max(np.abs(Ic)))
        Ish = S @ Ic_n
        sh, _ = build_pair(tpl, Ic_n, Ish)
        sh.draw(a, n_grid=260, extent=45.0 / SHIELD_RADIUS, legend=False)
        m = multipoles(sh)
        a.set_title(f"$\\varphi$ = {p:g}$^\\circ$   "
                    f"($|G|$ = {m['Gmag']*1e3:.3f} mT/mm, "
                    f"measured {m['phideg']:.1f}$^\\circ$)", fontsize=10)
    fig2.suptitle("Rotated quadrupole with shield, all at the 1000 A operating point")
    fig2.tight_layout()
    save_fig(fig2, "node5_rot_quad_fields.png")

    # --- macros for the report -------------------------------------------
    write_macros("results_rotquad.tex", {
        "RQlinres": "\\num{%.1e}" % lin_err,
        "RQgmax": f"{Gbare[imax]:.3f}",
        "RQgmin": f"{Gbare[imin]:.3f}",
        "RQphimax": f"{phi[imax]:.1f}",
        "RQphimin": f"{phi[imin]:.1f}",
        "RQripple": f"{100*(Gbare[imax]/Gbare[imin]-1):.1f}",
        "RQgshmax": f"{Gsh.max():.3f}",
        "RQgshmin": f"{Gsh.min():.3f}",
        "RQabsorb": f"{100*(1-Gsh.max()/Gbare[imax]):.0f}",
        "RQangerr": "\\num{%.1e}" % np.max(np.abs(ang_err)),
        "RQpurity": "\\num{%.1e}" % np.max(purity),
        "RQishmax": f"{Ishmax.max():.0f}",
        "RQshieldradius": f"{SHIELD_RADIUS:g}",
    })


if __name__ == "__main__":
    main()
