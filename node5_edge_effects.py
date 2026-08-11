"""node5_edge_effects.py -- how far can the 2D model be trusted (node 5).

Everything in this repository models each coil leg as an *infinite*
straight wire.  Real legs have a finite axial length L, and the question
for the design report is not whether that matters -- it does -- but
*which conclusions it changes*.

The mid-plane field of a straight segment of length L centred on that
plane is known exactly,

    B = (mu0 I / 2 pi rho) * (L/2) / sqrt(rho^2 + (L/2)^2),

so the correction to the infinite-wire model is the pure geometry factor

    f(rho; L) = [1 + (2 rho / L)^2]^{-1/2}  <  1.

Applying f per wire gives an exact mid-plane ("2.5D") model, which is
what this script builds.  It captures the truncation of the straight
legs; it does not capture the field of the end turns that close each
circuit, which is the remaining genuinely-3D piece and is left to FEM.

Three results follow.

1.  **The 2D model always overestimates, and more so with distance.**
    f falls below 0.9 beyond rho ~ L/4, so the 2D model is a good
    approximation only inside a quarter of the coil length.

2.  **Central-field numbers survive; the shielding *factor* does not.**
    The centre sits at rho ~ R = 22.5 mm, where f is close to 1 for any
    plausible coil length, so the central gradient is only mildly
    optimistic.  The far benchmark sits at 419 mm, where f is small --
    but the point is not that the leakage shrinks.  It is that the
    shielding factor collapses: an infinite shield cancels the exterior
    multipoles almost exactly, and a finite one cannot, so the 2D model
    overestimates the suppression at 419 mm by orders of magnitude.  The
    absolute residual field stays small either way, which is what
    matters for the design, but the 10^6-scale suppression numbers of
    the 2D study must not be quoted as predictions.

3.  **The optimum shield radius barely moves.**  Coil ring and shield
    ring sit at almost the same rho, so f is nearly common to source and
    shield and largely cancels out of the ratio that sets the optimum.
    This is the reason the 2D design conclusion survives.

Run:  python node5_edge_effects.py
Outputs: figures/node5_edge_effects.png,
         ../PSXM_design_report/results_edge.tex
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver
from shield_common import (G, MAX_CURRENT, shield_zero_solver, build_pair,
                           ring_meanB)
from node5_common import (multipoles, length_factor, validity_radius,
                          FiniteCoils, RADIUS, save_fig, write_macros)

LENGTHS = np.array([20.0, 30.0, 45.0, 65.0, 90.0, 130.0, 164.0, 200.0,
                    300.0, 500.0, 1000.0])          # mm
L_REF = 164.0        # mm, the designed length of the inner-ASSM (Abe 2022);
                     # used as the representative length quoted in the report
L_2D = 1.0e9                                        # mm, stands in for infinity
RHO_MARKS = (RADIUS, 100.0, 419.0)                  # mm, the reported distances
SHIELD_N_SCAN = 120                                 # cheaper discretization for the R_s scan
R_SCAN = np.arange(23.0, 42.0, 0.5)                 # mm
L_FOR_RSCAN = (100.0, L_REF, 300.0, L_2D)           # mm


# --------------------------------------------------------------------------
# a solver whose kernel carries the finite-length factor
# --------------------------------------------------------------------------
class FiniteSolver(CurrentSolver):
    """``CurrentSolver`` with the mid-plane finite-length correction.

    The correction is diagonal in the (sample, wire) pair -- it depends
    only on their separation -- so it multiplies the 2D kernel
    element-wise, once for the Bx block and once for the By block.
    """

    length = L_2D

    def coefficient_matrix(self):
        K = super().coefficient_matrix()
        dx = self.sample_x[:, None] - self.current_x[None, :]
        dy = self.sample_y[:, None] - self.current_y[None, :]
        f = length_factor(np.hypot(dx, dy), self.length)
        return K * np.vstack([f, f])


def _as_finite(solver, L):
    solver.__class__ = FiniteSolver
    solver.length = L
    return solver


def finite_quad_coils(tpl, L, gradient=G):
    """Coil currents for the centre quadrupole in the finite-length model."""
    s = CurrentSolver.from_current_source(tpl)
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x, y = np.cos(a), np.sin(a)
        s.add_sample_point(x, y, Bx=gradient * y, By=gradient * x)
    _as_finite(s, L)
    K = (s.coefficient_matrix() @ tpl.group_matrix())[:, :PSXMCoils.N_COILS]
    I, *_ = np.linalg.lstsq(K, s.target_field(), rcond=None)
    return CurrentSolver.normalize_currents(I, MAX_CURRENT)


def finite_shield_currents(tpl, I_coil, L):
    """Least-squares shield response in the finite-length model."""
    s = _as_finite(shield_zero_solver(tpl), L)
    KM = s.coefficient_matrix() @ tpl.group_matrix()
    K6, Ksh = KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:]
    X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
    return (-X) @ I_coil


def evaluate(L, shield_radius=27.5, shield_n=200):
    """Centre gradient (T/mm) and leakage (uT) at 1000 A, length L."""
    tpl = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS), shield=True,
                    shield_radius=shield_radius, shield_n=shield_n)
    Ic = finite_quad_coils(tpl, L)
    Ish = finite_shield_currents(tpl, Ic, L)
    sh, un = build_pair(tpl, Ic, Ish)
    shf, unf = FiniteCoils(sh, L), FiniteCoils(un, L)
    m = multipoles(shf)
    l100, l419 = ring_meanB(shf, 100.0) * 1e6, ring_meanB(shf, 419.0) * 1e6
    r100, r419 = ring_meanB(unf, 100.0) * 1e6, ring_meanB(unf, 419.0) * 1e6
    return dict(
        Gmag=m["Gmag"],
        leak100=l100, leak419=l419, raw100=r100, raw419=r419,
        # the meaningful figure of merit: shielded vs unshielded at the
        # SAME coil length.  Comparing the shielded leakage against its
        # own 2D value divides by a number that is essentially zero in
        # 2D and says nothing useful.
        supp100=r100 / max(l100, 1e-30),
        supp419=r419 / max(l419, 1e-30),
        Ishmax=float(np.max(np.abs(Ish))),
    )


def optimum_radius(L, target=G):
    """Smallest shield radius still delivering ``target`` at 1000 A."""
    g = []
    for R in R_SCAN:
        tpl = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS), shield=True,
                        shield_radius=R, shield_n=SHIELD_N_SCAN)
        Ic = finite_quad_coils(tpl, L)
        Ish = finite_shield_currents(tpl, Ic, L)
        sh, _ = build_pair(tpl, Ic, Ish)
        g.append(multipoles(FiniteCoils(sh, L))["Gmag"])
    g = np.array(g)
    if g.max() < target:
        return np.nan, g
    i = int(np.argmax(g >= target))
    if i == 0:
        return float(R_SCAN[0]), g
    # linear interpolation between the bracketing radii
    r0, r1, g0, g1 = R_SCAN[i - 1], R_SCAN[i], g[i - 1], g[i]
    return float(r0 + (target - g0) * (r1 - r0) / (g1 - g0)), g


def main():
    # --- 1. the pure geometry factor --------------------------------------
    print("finite-length correction f(rho; L) = [1 + (2 rho/L)^2]^(-1/2)\n")
    print(f"{'L (mm)':>8} | " + " ".join(f"f({r:.0f}mm)".rjust(11) for r in RHO_MARKS)
          + " | rho_90% (mm)")
    for L in LENGTHS:
        fs = " ".join(f"{length_factor(r, L):11.4f}" for r in RHO_MARKS)
        print(f"{L:8.0f} | {fs} | {validity_radius(L):11.1f}")
    print("\n  rho_90%: radius out to which the 2D model is within 10% "
          "(= L/4 to within a few percent)\n")

    # --- 2. full re-evaluation of the design at finite length --------------
    ref = evaluate(L_2D)
    print("design quantities at 1000 A, shield R_s = 27.5 mm\n")
    print(f"{'L (mm)':>8} | {'|G| (mT/mm)':>12} {'rel 2D':>7} | "
          f"{'leak@419 uT':>12} {'suppr@419':>11} | {'suppr@100':>11}")
    rows = []
    for L in list(LENGTHS) + [L_2D]:
        e = ref if L >= L_2D else evaluate(L)
        rows.append((L, e))
        tag = "2D" if L >= L_2D else f"{L:.0f}"
        print(f"{tag:>8} | {e['Gmag']*1e3:12.4f} {e['Gmag']/ref['Gmag']:7.3f} | "
              f"{e['leak419']:12.6f} {e['supp419']:11.3g} | {e['supp100']:11.3g}")
    print("\n  suppression = unshielded / shielded leakage at the same coil"
          "\n  length.  The 2D model does not merely overestimate the field;"
          "\n  it overestimates the *shielding factor* by orders of magnitude,"
          "\n  because a finite shield cannot cancel the far field the way an"
          "\n  infinite one does.  The absolute residual stays small either way."
          "\n  (leak@100 is a small difference of large cancelling terms and is"
          "\n  correspondingly noisy in L; the suppression factor is the robust"
          "\n  quantity.)")

    # --- 3. does the optimum move? ------------------------------------------
    print("\noptimum shield radius (smallest R_s reaching 1 mT/mm at 1000 A)")
    opts = []
    for L in L_FOR_RSCAN:
        R, _ = optimum_radius(L)
        opts.append((L, R))
        tag = "2D" if L >= L_2D else f"{L:.0f} mm"
        print(f"  L = {tag:>8}:  R_s* = {R:.3f} mm")
    finite_opts = [o for o in opts if o[0] < L_2D and np.isfinite(o[1])]
    spread = (max(o[1] for o in finite_opts) - min(o[1] for o in finite_opts)
              if finite_opts else np.nan)
    print(f"  spread across the length scan: {spread:.3f} mm")

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

    rho = np.geomspace(5.0, 500.0, 300)
    for L in (20.0, 65.0, 200.0, 500.0):
        ax[0].loglog(rho, length_factor(rho, L), lw=1.3, label=f"L = {L:.0f} mm")
    for r in RHO_MARKS:
        ax[0].axvline(r, color="gray", ls="--", lw=0.7)
    ax[0].axhline(0.9, color="tab:red", ls=":", lw=1.0, label="10% error")
    ax[0].set_xlabel(r"distance $\rho$ (mm)")
    ax[0].set_ylabel(r"$f = B_{\rm finite}/B_{\rm 2D}$")
    ax[0].set_title("the 2D model always overestimates")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    fin = [r for r in rows if r[0] < L_2D]
    Ls = np.array([r[0] for r in fin])
    ax[1].semilogx(Ls, [r[1]["Gmag"] / ref["Gmag"] for r in fin], "o-", ms=3,
                   color="tab:blue", label="centre $|G|$ / 2D value")
    ax[1].axhline(1.0, color="tab:blue", ls="--", lw=0.8)
    ax[1].set_xlabel("coil axial length $L$ (mm)")
    ax[1].set_ylabel("centre $|G|$ / 2D value", color="tab:blue")
    ax[1].set_ylim(0.6, 1.05)
    ax[1].grid(alpha=0.3, which="both")

    axr = ax[1].twinx()
    axr.loglog(Ls, [r[1]["supp419"] for r in fin], "^-", ms=3, color="tab:red",
               label="suppression @ 419 mm")
    axr.axhline(ref["supp419"], color="tab:red", ls=":", lw=1.0)
    axr.set_ylabel("shielding factor @ 419 mm", color="tab:red")
    ax[1].set_title("centre survives; the shielding factor does not")

    lab = ["2D" if L >= L_2D else f"{L:.0f}" for L, _ in opts]
    ax[2].bar(np.arange(len(opts)), [o[1] for o in opts], 0.55)
    ax[2].set_xticks(np.arange(len(opts)))
    ax[2].set_xticklabels(lab)
    ax[2].set_xlabel("coil axial length $L$ (mm)")
    ax[2].set_ylabel("optimum $R_s$ (mm)")
    ax[2].set_ylim(min(o[1] for o in opts) - 1.0, max(o[1] for o in opts) + 1.0)
    ax[2].set_title("the optimum barely moves")
    ax[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Edge effects: what the infinite-wire approximation costs, "
                 "and which conclusions survive it")
    fig.tight_layout()
    save_fig(fig, "node5_edge_effects.png")

    # --- macros ---------------------------------------------------------------
    eref = dict(rows)[L_REF]
    write_macros("results_edge.tex", {
        "EEL": f"{L_REF:.0f}",
        "EEfcentre": f"{length_factor(RADIUS, L_REF):.3f}",
        "EEfhundred": f"{length_factor(100.0, L_REF):.3f}",
        "EEffar": f"{length_factor(419.0, L_REF):.4f}",
        "EEgrel": f"{eref['Gmag']/ref['Gmag']:.3f}",
        "EEleak": f"{eref['leak419']:.4f}",
        "EEsupp": f"{eref['supp419']:.0f}",
        "EEsupptwod": "\\num{%.1e}" % ref["supp419"],
        "EEspread": f"{spread:.2f}",
        "EEropt": " / ".join(f"{o[1]:.2f}" for o in opts),
    })


if __name__ == "__main__":
    main()
