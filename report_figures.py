"""report_figures.py -- the figures as they appear in the design report.

This script supersedes the plotting sections of field_capability.py,
rot_quad.py, chi2_scan.py, shield2D.py, dipole_shield_optimization.py
and optimize_shield_radius.py.  The physics is unchanged -- every number
still comes from those modules -- but the presentation follows four rules
that the earlier figures broke in places:

  1. one physical dimension per axis.  Quantities of different dimensions
     (mT and mT/mm, degrees and a dimensionless residual) are either put
     in separate panels or normalized to their requirement first.
  2. leakage is plotted in the unit the requirement is written in (uT),
     with the published 1 uT requirement drawn as a labelled line.
  3. every reference line is labelled; no bare dashed line.
  4. no statistical language for a design objective: the weighted
     least-squares residual is not a chi-square and is not called one.

Prerequisite:  python report_scan.py    (writes scan_1000A.npz)
Run:           python report_figures.py
Outputs:       figures/*.png  and  ../PSXM_design_report/{figures/*.png,
               results_scan.tex, table_scan.tex}
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psxm_coils import PSXMCoils
from shield import (SHIELD_N, solve_quad_coils, solve_dipole_coils,
                           ls_shield_currents, build_pair, ring_meanB,
                           R_MAX_MM, net_current_fraction)
from field_analysis import (multipoles, uniformity, unit_response, lp_ceiling,
                          analytic_ceilings, save_fig, write_macros)
from field_capability import solve_rot_dipole
from rot_quad import solve_rot_quad
from rot_quad import shield_response as ls_shield_response

MAX_CURRENT = 1000.0
G_REQ = 1.0        # mT/mm (working benchmark, not a published specification)
B_REQ = 1.0        # mT   (provisional central-field working benchmark)
# Illustrative reading of the published BL = 1.0e-3 T.m with the published
# effective magnetic length 0.1 m (Abe 2D design; collaboration material):
# B_centre ~ 10 mT.  The 164 mm physical length of the 3D device is NOT an
# effective magnetic length and must not be used for this conversion.
B_BL_ILL = 10.0    # mT   (illustrative, sensitivity only)
LEAK_REQ = 1.0     # uT   (published, at 0.2 m; drawn for scale, not verified here)
DIAG = (200.0, 419.0)  # diagnostic radii: the published 0.2 m, and the
                       # nearest approach of the stored beam
SHIELD_RADIUS = 27.5
COIL_R = 22.5

COL_CEIL = "0.25"
COL_LS = "tab:blue"
COL_SH = "tab:orange"
COL_REQ = "tab:red"
COL_BL = "tab:purple"


def _load():
    d = np.load("scan_1000A.npz")
    cols = ("R", "G", "B0", "Ish", "leaknear", "leakfar", "barenear", "barefar")
    unpack = lambda a: {k: a[:, i] for i, k in enumerate(cols)}
    return (unpack(d["quad"]), unpack(d["dipole"]),
            unpack(d["quad_tab"]), unpack(d["dipole_tab"]), d["opt"])


# =========================================================================
# 1. capability: hardware ceiling, least-squares design, and the shield cost
# =========================================================================
def fig_capability(opt):
    """Panels (a) dipole and (b) quadrupole capability at the current limit."""
    resp = unit_response()
    psi_deg = np.arange(0.0, 120.0 + 1e-9, 1.0)
    lp_dip = lp_ceiling(resp[:, 0:2], np.radians(psi_deg)) * 1e3      # mT
    lp_quad = lp_ceiling(resp[:, 2:4], np.radians(psi_deg)) * 1e3     # mT/mm

    bare = PSXMCoils(currents=np.zeros(6))
    ls_dip = []
    for p in psi_deg:
        I = solve_rot_dipole(bare, p)
        I = I * (MAX_CURRENT / np.max(np.abs(I)))
        ls_dip.append(multipoles(PSXMCoils(currents=I))["B0mag"] * 1e3)
    ls_dip = np.array(ls_dip)

    phi_deg = psi_deg / 2.0                     # mechanical roll angle
    ls_quad = []
    for p in phi_deg:
        I = solve_rot_quad(bare, p)
        I = I * (MAX_CURRENT / np.max(np.abs(I)))
        ls_quad.append(multipoles(PSXMCoils(currents=I))["Gmag"] * 1e3)
    ls_quad = np.array(ls_quad)

    # achieved field at 1000 A for the bare ring and three shield radii
    cases = [(None, "bare ring"), (27.5, "27.5 mm"),
             (26.233, "26.233 mm"), (23.5, "23.5 mm")]
    got = []
    for R, lbl in cases:
        if R is None:
            tpl = PSXMCoils(currents=np.zeros(6))
            Iq, Id = solve_quad_coils(tpl), solve_dipole_coils(tpl)
            oq, od = PSXMCoils(currents=Iq), PSXMCoils(currents=Id)
        else:
            tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                            shield_radius=R, shield_n=SHIELD_N)
            Iq, Id = solve_quad_coils(tpl), solve_dipole_coils(tpl)
            oq, _ = build_pair(tpl, Iq, ls_shield_currents(tpl, Iq))
            od, _ = build_pair(tpl, Id, ls_shield_currents(tpl, Id))
        got.append((lbl, multipoles(od)["B0mag"] * 1e3, multipoles(oq)["Gmag"] * 1e3))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax[0].plot(psi_deg, lp_dip, "--", color=COL_CEIL, lw=1.3,
               label="box-constrained maximum")
    ax[0].plot(psi_deg, ls_dip, color=COL_LS, lw=1.6,
               label="least-squares design, bare ring")
    ax[0].axhline(B_REQ, color=COL_REQ, ls=":", lw=1.2,
                  label=f"{B_REQ:g} mT working benchmark")
    ax[0].axhline(B_BL_ILL, color=COL_BL, ls="-.", lw=1.2,
                  label=f"{B_BL_ILL:g} mT illustrative ($BL$/$L_\\mathrm{{eff}}$)")
    ax[0].set_xlabel(r"Dipole direction $\psi$ (deg)")
    ax[0].set_ylabel(r"Central dipole field $B_0$ (mT)")
    ax[0].set_title("(a) Dipole field capability", fontsize=10)
    ax[0].legend(fontsize=8, loc="center right")
    ax[0].grid(alpha=0.3)
    ax[0].set_ylim(0, 1.08 * lp_dip.max())

    ax[1].plot(phi_deg, lp_quad, "--", color=COL_CEIL, lw=1.3,
               label="box-constrained maximum")
    ax[1].plot(phi_deg, ls_quad, color=COL_LS, lw=1.6,
               label="least-squares design, bare ring")
    ax[1].axhline(G_REQ, color=COL_REQ, ls=":", lw=1.2,
                  label=f"{G_REQ:g} mT/mm working benchmark")
    ax[1].set_xlabel(r"Roll angle $\varphi$ (deg)")
    ax[1].set_ylabel(r"Central gradient $G$ (mT/mm)")
    ax[1].set_title("(b) Quadrupole-gradient capability", fontsize=10)
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].grid(alpha=0.3)
    ax[1].set_ylim(0, 1.08 * lp_quad.max())

    fig.tight_layout()
    save_fig(fig, "max_field_capability.png")

    i0 = 0      # psi = 0: the orientation both bare-ring designs are solved at
    return dict(
        lp_dip_max=lp_dip.max(), lp_dip_min=lp_dip.min(),
        lp_quad_max=lp_quad.max(), lp_quad_min=lp_quad.min(),
        lp_dip_at0=lp_dip[i0], lp_quad_at0=lp_quad[i0],
        dip_bare=ls_dip[i0], quad_bare=ls_quad[i0],
        ripple=100.0 * (lp_dip.max() / lp_dip.min() - 1.0),
        unif10=100.0 * uniformity(PSXMCoils(
            currents=solve_dipole_coils(PSXMCoils(currents=np.zeros(6)))))[10.0],
        got=got, analytic=analytic_ceilings(),
    )


# =========================================================================
# 2. rotational quadrupole
# =========================================================================
def fig_rot_quad():
    """Panels (a) the two-pattern current basis and (b) field quality."""
    phi = np.arange(0.0, 90.0 + 1e-9, 1.0)
    bare = PSXMCoils(currents=np.zeros(6))
    tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                    shield_radius=SHIELD_RADIUS, shield_n=SHIELD_N)
    S = ls_shield_response(tpl)

    I_norm, I_skew = solve_rot_quad(bare, 0.0), solve_rot_quad(bare, 45.0)
    lin_err = 0.0
    Iraw = []
    Gbare, Gsh, purity, purity_sh, ang, ang_sh, Ishmax = ([] for _ in range(7))
    for p in phi:
        Ic = solve_rot_quad(bare, p)
        c, s_ = np.cos(2 * np.radians(p)), np.sin(2 * np.radians(p))
        lin_err = max(lin_err, float(np.max(np.abs(Ic - (c * I_norm + s_ * I_skew)))
                                     / np.max(np.abs(I_norm))))
        Iraw.append(Ic.copy())
        Ic_n = Ic * (MAX_CURRENT / np.max(np.abs(Ic)))
        m = multipoles(PSXMCoils(currents=Ic_n))
        Ish = S @ Ic_n
        sh, _ = build_pair(tpl, Ic_n, Ish)
        msh = multipoles(sh)
        Gbare.append(m["Gmag"] * 1e3); Gsh.append(msh["Gmag"] * 1e3)
        purity.append(m["purity"]); purity_sh.append(msh["purity"])
        ang.append((m["phideg"] - p + 90.0) % 180.0 - 90.0)
        ang_sh.append((msh["phideg"] - p + 90.0) % 180.0 - 90.0)
        Ishmax.append(float(np.max(np.abs(Ish))))
    Iraw = np.array(Iraw)
    Gbare, Gsh = np.array(Gbare), np.array(Gsh)
    purity, purity_sh = np.array(purity), np.array(purity_sh)
    ang, ang_sh = np.array(ang), np.array(ang_sh)
    Ishmax = np.array(Ishmax)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

    for k in range(6):
        ax[0].plot(phi, Iraw[:, k], lw=1.2, label=f"$I_{k+1}$")
    ax[0].set_xlabel(r"Roll angle $\varphi$ (deg)")
    ax[0].set_ylabel(r"Coil current $I_k$ (A)")
    ax[0].set_title("(a) Coil-current basis decomposition", fontsize=10)
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)

    ax[1].semilogy(phi, purity, color=COL_LS, lw=1.4, label="bare ring")
    ax[1].semilogy(phi, purity_sh, color=COL_SH, lw=1.4, ls="--",
                   label=rf"with shield, $R_s$ = {SHIELD_RADIUS:g} mm")
    ax[1].set_xlabel(r"Roll angle $\varphi$ (deg)")
    ax[1].set_ylabel("Relative higher-order field content")
    ax[1].set_title("(b) Higher-order field content versus roll angle",
                    fontsize=10)
    ax[1].set_ylim(1e-4, 1e-1)
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].grid(alpha=0.3, which="both")

    fig.suptitle("Rotational quadrupole characteristics")
    fig.tight_layout()
    save_fig(fig, "rot_quad_scan.png")

    # two representative roll angles, for the field-line panels
    fig2, axs = plt.subplots(1, 2, figsize=(10.5, 5.4))
    for a, p, tag in zip(axs, (0.0, 45.0), ("a", "b")):
        Ic = solve_rot_quad(bare, p)
        Ic_n = Ic * (MAX_CURRENT / np.max(np.abs(Ic)))
        sh, _ = build_pair(tpl, Ic_n, S @ Ic_n)
        sh.draw(a, n_grid=260, extent=45.0 / SHIELD_RADIUS, legend=False)
        m = multipoles(sh)
        a.set_title(f"({tag}) $\\varphi$ = {p:g}$^\\circ$, "
                    f"$|G|$ = {m['Gmag']*1e3:.3f} mT/mm", fontsize=10)
        a.set_xlabel("$x$ (mm)"); a.set_ylabel("$y$ (mm)")
    fig2.tight_layout()
    save_fig(fig2, "rot_quad_fields.png")

    imax, imin = int(np.argmax(Gbare)), int(np.argmin(Gbare))
    return dict(lin_err=lin_err, gmax=Gbare[imax], gmin=Gbare[imin],
                phimax=phi[imax], phimin=phi[imin],
                ripple=100 * (Gbare[imax] / Gbare[imin] - 1),
                gshmax=Gsh.max(), gshmin=Gsh.min(),
                absorb=100 * (1 - Gsh.max() / Gbare[imax]),
                angerr=max(np.max(np.abs(ang)), np.max(np.abs(ang_sh))),
                purity=max(purity.max(), purity_sh.max()),
                purity_bare=purity[0],       # bare-ring quadrupole at phi=0
                ishmax=Ishmax.max())


# =========================================================================
# 3. leakage figures, in the unit the requirement is written in
# =========================================================================
def fig_exterior_profiles():
    """Ring-averaged exterior field versus radius, both modes, plus a
    field-line map of the quadrupole design for the appendix."""
    tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                    shield_radius=SHIELD_RADIUS, shield_n=SHIELD_N)
    rho = np.geomspace(1.0, R_MAX_MM, 140)
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for k, (mode, tag, name) in enumerate((("quad", "a", "Quadrupole mode"),
                                           ("dipole", "b", "Dipole mode"))):
        Ic = solve_quad_coils(tpl) if mode == "quad" else solve_dipole_coils(tpl)
        Ish = ls_shield_currents(tpl, Ic)
        sh, un = build_pair(tpl, Ic, Ish)
        Bun = np.array([ring_meanB(un, r) for r in rho]) * 1e6
        Bsh = np.array([ring_meanB(sh, r) for r in rho]) * 1e6
        ax[k].loglog(rho, Bun, color=COL_CEIL, lw=1.5, label="no shield")
        ax[k].loglog(rho, Bsh, color=COL_LS, lw=1.6, label="with shield")
        ax[k].axhline(LEAK_REQ, color=COL_REQ, ls="-.", lw=1.4,
                      label=r"$1\ \mu$T level (for scale)")
        ax[k].set_ylim(1e-6, 3e5)
        for xr, lbl, ytxt in ((COIL_R, "coil ring", 3e2),
                              (SHIELD_RADIUS, "shield", 2e5),
                              (DIAG[0], "200 mm", 2e5),
                              (DIAG[1], "419 mm", 2e5)):
            ax[k].axvline(xr, color="gray", ls="--", lw=0.7)
            ax[k].text(xr, ytxt, " " + lbl, rotation=90, va="top", ha="left",
                       fontsize=6.5, color="0.35")
        ax[k].set_xlabel(r"Radial distance $\rho$ (mm)")
        ax[k].set_ylabel(r"Ring-averaged field ($\mu$T)")
        ax[k].set_title(f"({tag}) {name}", fontsize=10)
        ax[k].legend(fontsize=8, loc="lower left")
        ax[k].grid(True, which="both", alpha=0.3)
        out[mode] = dict(leaknear=ring_meanB(sh, DIAG[0]) * 1e6,
                         leakfar=ring_meanB(sh, DIAG[1]) * 1e6)
        if mode == "quad":
            figm, axm = plt.subplots(figsize=(6.2, 6.0))
            sh.draw(axm, n_grid=340, extent=45.0 / SHIELD_RADIUS, legend=False)
            axm.set_title("")
            axm.set_xlabel("$x$ (mm)"); axm.set_ylabel("$y$ (mm)")
            figm.tight_layout()
            save_fig(figm, "shield_current_map.png")
    fig.tight_layout()
    save_fig(fig, "exterior_profiles.png")
    return out


# =========================================================================
# 4. what the sampling layout does and does not determine
# =========================================================================
def fig_sampling():
    """Sensitivity of the shield solve to the B=0 sampling layout.

    Three quantities, three panels, because they behave differently: the
    delivered central gradient and the required shield sheet current
    converge and are independent of the discretization, while the exterior
    suppression factor is set by the radius at which B = 0 is demanded and
    is not a converged quantity at all.

    The shield current is reported as a linear current density
    K = I_s^max N_s / (2 pi R_s), which is what the continuum limit fixes;
    the per-element current I_s^max itself scales as 1/N_s and is a
    property of the discretization rather than of the design.
    """
    gaps = (1.0, 2.0, 5.0, 10.0, 20.0)
    nbs = (1, 3, 6)
    Ns_cases = (100, 200)

    res = {}
    for Ns in Ns_cases:
        for nb in nbs:
            row = []
            for gap in gaps:
                tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                                shield_radius=SHIELD_RADIUS, shield_n=Ns)
                Ic = solve_quad_coils(tpl)
                Ish = ls_shield_currents(tpl, Ic, gap_mm=gap,
                                         outer_mm=gap / 2.0, n_between=nb)
                sh, un = build_pair(tpl, Ic, Ish)
                G = multipoles(sh)["Gmag"] * 1e3
                leak = ring_meanB(sh, DIAG[1]) * 1e6
                bare = ring_meanB(un, DIAG[1]) * 1e6
                K = float(np.max(np.abs(Ish))) * Ns / (2 * np.pi * SHIELD_RADIUS)
                row.append((gap, G, K, leak, bare))
            res[(Ns, nb)] = np.array(row)

    # third panel: the exterior residual against the shield discretization.
    # A closed flux-excluding shell cancels the exterior field of an interior
    # source with zero net current exactly, so the leakage this model reports
    # is the residual of a finite-N_s sheet -- and it vanishes as the sheet is
    # refined, which is what this panel shows.
    Ns_series = (50, 100, 200, 400)
    disc = {}
    for dr in (5.0, 20.0):
        row = []
        for Ns in Ns_series:
            tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                            shield_radius=SHIELD_RADIUS, shield_n=Ns)
            Ic = solve_quad_coils(tpl)
            Ish = ls_shield_currents(tpl, Ic, gap_mm=dr, outer_mm=dr / 2.0,
                                     n_between=3)
            sh, _ = build_pair(tpl, Ic, Ish)
            row.append(ring_meanB(sh, DIAG[1]) * 1e6)   # unclamped residual
        disc[dr] = np.array(row)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for (Ns, nb), a in res.items():
        ls = "-" if Ns == 200 else "--"
        lbl = rf"$N_s$={Ns}, {nb}/gap"
        ax[0].plot(a[:, 0], a[:, 1], ls, marker="o", ms=3.5, lw=1.2, label=lbl)
        ax[1].plot(a[:, 0], a[:, 2], ls, marker="o", ms=3.5, lw=1.2, label=lbl)

    ax[0].axhline(1.2028, color=COL_REQ, ls=":", lw=1.2,
                  label="1.203 mT/mm")
    ax[0].set_ylabel(r"Central gradient $G$ (mT/mm)")
    ax[0].set_title("(a) Central-gradient convergence", fontsize=10)
    ax[1].set_ylabel(r"Shield sheet current $\mathcal{K}_s$ (A/mm)")
    ax[1].set_title("(b) Shield sheet-current convergence", fontsize=10)
    for a in ax[:2]:
        a.set_xlabel(r"Outer-ring offset $\Delta r$ (mm)")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=6.5, ncol=2)

    for dr, mk in ((5.0, "o"), (20.0, "s")):
        ax[2].loglog(Ns_series, disc[dr], marker=mk, ms=4.5, lw=1.4,
                     label=rf"$\Delta r$ = {dr:g} mm")
    ax[2].axhline(1e-12, color="0.5", lw=1.0, ls="--",
                 label="reference 1e-12 $\mu$T (float-precision scale)")
    ax[2].set_xlabel(r"Shield discretization $N_s$")
    ax[2].set_ylabel(r"Residual field at 419 mm ($\mu$T)")
    ax[2].set_title("(c) Exterior residual versus shield discretization",
                    fontsize=10)
    ax[2].grid(alpha=0.3, which="both")
    ax[2].legend(fontsize=7)

    fig.suptitle("Sensitivity to sampling layout and shield discretization")
    fig.tight_layout()
    save_fig(fig, "sampling_convergence.png")

    base = res[(200, 3)]
    return dict(Kbase=float(base[2, 2]),
                suppr_lo=float(base[0, 4] / base[0, 3]),
                suppr_hi=float(base[-1, 4] / base[-1, 3]),
                disc5=disc[5.0], Ns_series=Ns_series)


# =========================================================================
# 5. shield-radius scan, per mode
# =========================================================================
def fig_scan_combined(q, d, opt, opt_bl):
    """Four-panel shield-radius scan: field delivered and exterior residual,
    quadrupole in the top row and dipole in the bottom row."""
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))

    for row, (scan, mode) in enumerate(((q, "quad"), (d, "dipole"))):
        R = scan["R"]
        achieved = scan["G"] if mode == "quad" else scan["B0"]
        req = G_REQ if mode == "quad" else B_REQ
        unit = "mT/mm" if mode == "quad" else "mT"
        ylab = (r"Central gradient $G$ (mT/mm)" if mode == "quad"
                else r"Central dipole field $B_0$ (mT)")
        tags = ("a", "b") if row == 0 else ("c", "d")
        name = "Quadrupole" if mode == "quad" else "Dipole"

        ax[row, 0].plot(R, achieved, "-", color=COL_LS, lw=1.8,
                        label="least-squares design")
        ax[row, 0].axhline(req, color=COL_REQ, ls=":", lw=1.4,
                           label=rf"{req:g} {unit} working benchmark")
        if mode == "dipole":
            ax[row, 0].axhline(B_BL_ILL, color=COL_BL, ls="-.", lw=1.4,
                               label=rf"{B_BL_ILL:g} {unit} illustrative ($BL$/$L_\mathrm{{eff}}$)")
        ax[row, 0].set_ylabel(ylab)
        ax[row, 0].set_title(f"({tags[0]}) {name} shield-radius scan",
                             fontsize=10)
        ax[row, 0].legend(fontsize=8, loc="lower right")
        ax[row, 0].grid(alpha=0.3)

        ax[row, 1].semilogy(R, scan["leaknear"], "-", color=COL_SH, lw=1.8,
                            label="200 mm")
        ax[row, 1].semilogy(R, scan["leakfar"], "-", color=COL_LS, lw=1.8,
                            label="419 mm")
        ax[row, 1].axhline(LEAK_REQ, color=COL_REQ, ls="-.", lw=1.4,
                           label=r"$1\ \mu$T level (for scale)")
        ax[row, 1].set_ylabel(r"Ring-averaged residual field ($\mu$T)")
        ax[row, 1].set_title(f"({tags[1]}) {name} exterior residual",
                             fontsize=10)
        ax[row, 1].legend(fontsize=8, loc="lower right")
        ax[row, 1].grid(alpha=0.3, which="both")

        marks = [(opt if mode == "quad" else opt_bl[0], "green")]
        if mode == "dipole":
            marks.append((opt_bl[1], COL_BL))
        for a in ax[row]:
            a.set_xlabel(r"Shield radius $R_s$ (mm)")
            for x, c in marks:
                a.axvline(x, color=c, ls=":", lw=1.2)

    fig.tight_layout()
    save_fig(fig, "shield_radius_scan.png")


# =========================================================================
# 6. how the radius is selected, and how robust that selection is
# =========================================================================
def fig_selection(q, d, opt):
    """Left: what selects the radius.  Right: what does not.

    The left panel is the field capability of both modes against their
    requirements, which is what fixes the smallest feasible radius.  The right
    panel is the exterior residual at 419 mm for four sampling layouts: it
    spans ten orders of magnitude, so it cannot select anything, while the
    threshold radius it implies is the same in every case.
    """
    Rq, Rd, Rbl = opt
    rob = np.load("robust_scan.npz")
    keys = ("R", "G", "Ish", "leaknear", "leakfar", "barenear", "barefar")
    col = lambda a, k: a[:, keys.index(k)]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))

    ax[0].semilogy(q["R"], q["G"] / G_REQ, color=COL_LS, lw=1.8,
                   label=r"quadrupole:  $G\,/\,1$ mT/mm")
    ax[0].semilogy(d["R"], d["B0"] / B_REQ, color=COL_SH, lw=1.8,
                   label=r"dipole:  $B_0\,/\,1$ mT")
    ax[0].semilogy(d["R"], d["B0"] / B_BL_ILL, color=COL_BL, lw=1.4, ls="-.",
                   label=r"dipole:  $B_0\,/\,10$ mT  ($BL$/$L_\mathrm{eff}$, illustrative)")
    ax[0].axhline(1.0, color=COL_REQ, ls=":", lw=1.5, label="requirement met")
    ax[0].set_ylabel("Field capability / requirement")
    ax[0].set_title("(a) Field capability relative to requirement", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.3, which="both")
    for x, c in ((Rd, COL_SH), (Rbl, COL_BL), (Rq, COL_LS)):
        ax[0].axvline(x, color=c, ls=":", lw=1.1)
    ax[0].text(0.03, 0.97, "threshold $R_s$:\n"
               + f"  dipole, 1 mT        {Rd:.3f} mm\n"
               + f"  dipole, $BL$ ($L_\\mathrm{{eff}}$=0.1 m)  {Rbl:.1f} mm\n"
               + f"  quadrupole        {Rq:.3f} mm",
               transform=ax[0].transAxes, va="top", fontsize=7.5,
               bbox=dict(fc="white", ec="0.7", alpha=0.9))

    for dr, ls in ((2, "-"), (5, "--"), (10, "-."), (20, ":")):
        a = rob[f"dr{dr}"]
        ax[1].semilogy(col(a, "R"), np.maximum(col(a, "leakfar"), 1e-13), ls,
                       lw=1.5, label=rf"$\Delta r$ = {dr} mm")
    ax[1].axhline(LEAK_REQ, color=COL_REQ, ls="-.", lw=1.4,
                  label=r"$1\ \mu$T level (for scale)")
    ax[1].axhline(1e-12, color="0.5", lw=1.0, ls="--",
                 label="reference 1e-12 $\mu$T (float-precision scale)")
    ax[1].axvline(Rq, color=COL_LS, ls=":", lw=1.2)
    ax[1].set_ylabel(r"Residual field at 419 mm ($\mu$T)")
    ax[1].set_title("(b) Exterior residual for alternative sampling layouts",
                    fontsize=10)
    ax[1].legend(fontsize=7.5, loc="lower right", ncol=2)
    ax[1].grid(alpha=0.3, which="both")
    ax[1].text(0.03, 0.97, "same threshold radius in every layout:\n"
               r"  $R_s$ = 26.234, 26.233, 26.233, 26.233 mm",
               transform=ax[1].transAxes, va="top", fontsize=7.5,
               bbox=dict(fc="white", ec="0.7", alpha=0.9))

    for a in ax:
        a.set_xlabel(r"Shield radius $R_s$ (mm)")
        a.set_xlim(22.5, 45.0)
    fig.suptitle("Robustness of the shield-radius threshold")
    fig.tight_layout()
    save_fig(fig, "shield_radius_selection.png")


# =========================================================================
def latex_table(qt, dt, fname="table_scan.tex"):
    """The shield-radius table rows, both modes in one macro.

    Emits \\scantablecombined: R_s, the quadrupole gradient and its sheet
    current, the dipole central field and its sheet current, and the
    ring-averaged exterior residual at the far diagnostic radius for each
    mode.  Emitted as a macro definition rather than raw rows because
    \\input inside a tabular breaks the alignment, while a macro expanding
    to the rows does not.
    """
    import os
    from field_analysis import REPORT_DIR
    os.makedirs(REPORT_DIR, exist_ok=True)

    def sci(v):
        e = int(np.floor(np.log10(v)))
        m = v / 10.0 ** e
        if round(m, 1) >= 10.0:          # 9.96e-5 must print as 1.0e-4
            m, e = m / 10.0, e + 1
        return r"$%.1f\times10^{%d}$" % (m, e)

    def sheet(I, R):
        return I * SHIELD_N / (2 * np.pi * R)      # A/mm, N_s-independent

    rows = []
    for i in range(len(qt["R"])):
        assert abs(qt["R"][i] - dt["R"][i]) < 1e-9, "table radii must match"
        rows.append("    %.3f & %.3f & %.0f & %.2f & %.0f & %s & %s \\\\" % (
            qt["R"][i], qt["G"][i], sheet(qt["Ish"][i], qt["R"][i]),
            dt["B0"][i], sheet(dt["Ish"][i], dt["R"][i]),
            sci(qt["leakfar"][i]), sci(dt["leakfar"][i])))

    p = os.path.join(REPORT_DIR, fname)
    with open(p, "w", encoding="utf-8") as f:
        f.write("% auto-generated by report_figures.py -- do not edit\n")
        f.write("\\def\\scantablecombined{%\n")
        f.write("\n".join(rows) + "%\n}\n")
    print("wrote", p)


def main():
    q, d, qt, dt, opt = _load()
    cap = fig_capability(opt)
    rq = fig_rot_quad()
    prof = fig_exterior_profiles()
    lq, ld = prof["quad"], prof["dipole"]
    sp = fig_sampling()
    fig_scan_combined(q, d, opt[0], (opt[1], opt[2]))
    fig_selection(q, d, opt)
    latex_table(qt, dt)

    # net-current check of the baseline solved shield sheet (quadrupole
    # design at the default radius): the exact-cancellation argument of the
    # report requires this fraction to be numerically zero.
    tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                    shield_radius=SHIELD_RADIUS, shield_n=SHIELD_N)
    Ish0 = ls_shield_currents(tpl, solve_quad_coils(tpl))
    epsI = net_current_fraction(Ish0)

    # price of field quality, at the SAME orientation the design is solved at
    price_dip = 100 * cap["dip_bare"] / cap["lp_dip_at0"]
    price_quad = 100 * cap["quad_bare"] / cap["lp_quad_at0"]
    print(f"\nprice of field quality at the design's own orientation: "
          f"dipole {price_dip:.1f}%, quadrupole {price_quad:.1f}% of the ceiling")
    print(f"(against the ceiling at its most favourable orientation the "
          f"quadrupole would read {100*cap['quad_bare']/cap['lp_quad_max']:.0f}%, "
          f"which is the ripple, not a field-quality cost)")

    write_macros("results_scan.tex", {
        "MFdipceil": f"{cap['lp_dip_max']:.2f}",
        "MFquadceil": f"{cap['lp_quad_max']:.3f}",
        "MFripple": f"{cap['ripple']:.1f}",
        "MFdipbare": f"{cap['dip_bare']:.2f}",
        "MFquadbare": f"{cap['quad_bare']:.3f}",
        "MFpricedip": f"{price_dip:.0f}",
        "MFpricequad": f"{price_quad:.0f}",
        "MFquadripplefrac": f"{100*cap['quad_bare']/cap['lp_quad_max']:.0f}",
        "MFdipdefault": f"{cap['got'][1][1]:.3f}",
        "MFquaddefault": f"{cap['got'][1][2]:.3f}",
        # central-field change when the shield is added (bare ring -> default
        # Rs = 27.5 mm), at the same 1000 A operating point
        "SHdipratio": f"{cap['got'][0][1] / cap['got'][1][1]:.2f}",
        "SHquadratio": f"{cap['got'][0][2] / cap['got'][1][2]:.2f}",
        "SHdipred": f"{100 * (1 - cap['got'][1][1] / cap['got'][0][1]):.1f}",
        "SHquadred": f"{100 * (1 - cap['got'][1][2] / cap['got'][0][2]):.1f}",
        "MFdipeng": f"{cap['got'][3][1]:.2f}",
        "MFunif": f"{cap['unif10']:.1f}",
        "RQlinres": "\\num{%.1e}" % rq["lin_err"],
        "RQgmax": f"{rq['gmax']:.3f}", "RQgmin": f"{rq['gmin']:.3f}",
        "RQripple": f"{rq['ripple']:.1f}",
        "RQgshmax": f"{rq['gshmax']:.3f}", "RQgshmin": f"{rq['gshmin']:.3f}",
        "RQabsorb": f"{rq['absorb']:.0f}",
        "RQangerr": "\\num{%.1e}" % rq["angerr"],
        "RQpurity": "\\num{%.1e}" % rq["purity"],
        "RQpuritybare": "\\num{%.1e}" % rq["purity_bare"],
        "RQishmax": f"{rq['ishmax']:.0f}",
        "RQishsheet": f"{rq['ishmax'] * SHIELD_N / (2 * np.pi * SHIELD_RADIUS):.0f}",
        "RQshieldradius": f"{SHIELD_RADIUS:g}",
        "OPTquad": f"{opt[0]:.3f}", "OPTdip": f"{opt[1]:.3f}",
        "OPTdipBLeff": f"{opt[2]:.1f}",
        "EPSI": "\\num{%.1e}" % epsI,
        "LKquadnear": "\\num{%.1e}" % lq["leaknear"],
        "LKquadfar": "\\num{%.1e}" % lq["leakfar"],
        "LKdipnear": "\\num{%.1e}" % ld["leaknear"],
        "LKdipfar": "\\num{%.1e}" % ld["leakfar"],
        "SPK": f"{sp['Kbase']:.0f}",
        "SPsupprlo": "\\num{%.0e}" % sp["suppr_lo"],
        "SPsupprhi": "\\num{%.0e}" % sp["suppr_hi"],
    })


if __name__ == "__main__":
    main()
