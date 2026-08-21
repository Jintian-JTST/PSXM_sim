# -*- coding: utf-8 -*-
"""report_background.py -- field quality versus radius (reference-radius sigma)
and compatibility of the six-coil PSXM design with an external background field.

Two new results for the design report (Sec. 3.4 / 3.5):

  1. N_sigma(r): the higher-order field content at radius r, quoted in units
     of its value at the 1 mm reference ring (sigma = purity at 1 mm,
     N_sigma(r) = purity(r) / purity(1 mm)).  Computed for dipole and
     quadrupole, bare ring and shielded (R_s = 27.5 mm), together with the
     good-field radius at adopted tolerance levels of 1% and 5%.

  2. Background-field compatibility: the coil solve compensates an external
     background field by superposition (target -> target - B_background at
     the sample points).  The model is incremental: the shield's own
     transient response to the background is not modelled (Sec. 7).

     For each mode (dipole / quadrupole) and background kind (uniform /
     gradient) the scan reports the largest background amplitude, per
     direction, for which the magnet still delivers the adopted benchmark
     at the 1000 A operating point:

       * same-channel backgrounds (dipole + uniform, quadrupole + gradient)
         rescale the design's own current pattern: the delivered field and
         its quality are unchanged and the window is set by the 1000 A cap;
       * cross-channel backgrounds (quadrupole + uniform, dipole +
         gradient) share the current budget between the two patterns, so
         the delivered field at the cap falls toward the benchmark; with
         the shield the window closes earlier, at
         max|I| <= 1000 x (shielded/bare delivered ratio).

     The field quality (purity) of the delivered field is unchanged to
     machine precision in the same-channel case; in the cross-channel case
     the added pattern contributes its own higher-order content.

Run:  python report_background.py
Outputs: figures/background_field.png  and
         ../PSXM_design_report/{figures/background_field.png,
                                results_background.tex}
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver
from shield_common import SHIELD_N, MAX_CURRENT, ls_shield_currents, build_pair
from node5_common import multipoles, save_fig, write_macros

G_REQ = 1.0          # mT/mm, quadrupole working benchmark
B_REQ = 1.0          # mT, dipole working benchmark (central field)
SHIELD_RADIUS = 27.5
RADII_NSIG = (1.0, 2.0, 5.0, 10.0)     # N_sigma table radii
RADII_SCAN = np.arange(1.0, 15.01, 0.5)
TOLS = (0.01, 0.05)                    # adopted good-field tolerance levels
DIRS_UNIF = np.arange(0.0, 180.0, 15.0)   # uniform-bg direction scan
DIRS_GRAD = np.arange(0.0, 90.0, 15.0)    # gradient-bg direction scan (spin-2)
AMPS_UNIF = np.arange(0.0, 60.01, 0.5)    # mT
AMPS_GRAD = np.arange(0.0, 3.51, 0.05)    # mT/mm


# --------------------------------------------------------------------------
# coil solve with a background field
# --------------------------------------------------------------------------
def base_target(mode, x, y):
    if mode == "quad":
        return G_REQ * 1e-3 * y, G_REQ * 1e-3 * x
    return B_REQ * 1e-3, 0.0


def uniform_bg(Bx, By):
    return lambda x, y: (Bx, By)


def gradient_bg(G, phi_deg):
    """Normal quadrupole background rotated mechanically by phi (deg):
    (Gn, Gs) = |G| (cos 2phi, -sin 2phi), Bx = Gs x + Gn y, By = Gn x - Gs y.
    G in mT/mm, positions in mm."""
    c, s = np.cos(2 * np.radians(phi_deg)), np.sin(2 * np.radians(phi_deg))
    Gn, Gs = G * 1e-3 * c, -G * 1e-3 * s
    return lambda x, y: (Gs * x + Gn * y, Gn * x - Gs * y)


def raw_solve(mode, bg, target=None):
    """Raw (unnormalised) coil currents for target = benchmark - background
    (or an explicit target(x, y) field), sampled on the 1 mm ring exactly
    as the baseline solves."""
    tpl = PSXMCoils(currents=np.zeros(6))
    solver = CurrentSolver.from_current_source(tpl)
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x, y = np.cos(a), np.sin(a)
        if target is None:
            bx, by = base_target(mode, x, y)
        else:
            bx, by = target(x, y)
        gx, gy = bg(x, y)
        solver.add_sample_point(x, y, Bx=bx - gx, By=by - gy)
    K = (solver.coefficient_matrix() @ tpl.group_matrix())[:, :6]
    I, *_ = np.linalg.lstsq(K, solver.target_field(), rcond=None)
    return I


def build_obj(Ic, shielded):
    if not shielded:
        return PSXMCoils(currents=Ic)
    tpl = PSXMCoils(currents=np.zeros(6), shield=True,
                    shield_radius=SHIELD_RADIUS, shield_n=SHIELD_N)
    sh, _ = build_pair(tpl, Ic, ls_shield_currents(tpl, Ic))
    return sh


def delivered(mode, shielded, I_raw):
    """(delivered B0 in mT or G in mT/mm, purity at 1 mm) at the 1000 A point."""
    I = I_raw * (MAX_CURRENT / np.max(np.abs(I_raw)))
    m = multipoles(build_obj(I, shielded), r0=1.0, n=64)
    return (m["Gmag"] if mode == "quad" else m["B0mag"]) * 1e3, m["purity"]


# --------------------------------------------------------------------------
# 1. N_sigma vs radius and good-field radii
# --------------------------------------------------------------------------
def nsig_scan(mode):
    I0 = raw_solve(mode, uniform_bg(0.0, 0.0))
    I0 = I0 * (MAX_CURRENT / np.max(np.abs(I0)))
    out = {}
    for shielded in (False, True):
        obj = build_obj(I0, shielded)
        pur = np.array([multipoles(obj, r0=r, n=64)["purity"] for r in RADII_SCAN])
        sigma = float(pur[0])
        nsig = pur / sigma
        gf = {}
        for tol in TOLS:
            above = np.where(pur > tol)[0]
            if len(above) == 0:
                gf[tol] = float("nan")          # never reaches the level
            elif above[0] == 0:
                gf[tol] = float("nan")          # already above at 1 mm
            else:
                i = above[0]
                r0, r1 = RADII_SCAN[i - 1], RADII_SCAN[i]
                p0, p1 = pur[i - 1], pur[i]
                gf[tol] = float(r0 + (tol - p0) * (r1 - r0) / (p1 - p0))
        key = "bare" if not shielded else "sh"
        out[key] = dict(sigma=sigma, nsig=nsig, gf=gf)
    return out


# --------------------------------------------------------------------------
# 2. background scans: peak current required at the benchmark
# --------------------------------------------------------------------------
def ireq_uniform(mode, amp_mT, dir_deg):
    th = np.radians(dir_deg)
    I = raw_solve(mode, uniform_bg(amp_mT * 1e-3 * np.cos(th),
                                   amp_mT * 1e-3 * np.sin(th)))
    return float(np.max(np.abs(I)))


def ireq_gradient(mode, amp_mTmm, dir_deg):
    I = raw_solve(mode, gradient_bg(amp_mTmm, dir_deg))
    return float(np.max(np.abs(I)))


def scan(ireq, amps, dirs, threshold):
    """Grid of required peak currents, then per-direction windows where
    ireq <= threshold."""
    I_req = np.empty((len(amps), len(dirs)))
    for i, a in enumerate(amps):
        for j, d in enumerate(dirs):
            I_req[i, j] = ireq(a, d)
    wins = np.empty(len(dirs))
    for j in range(len(dirs)):
        feas = np.where(I_req[:, j] <= threshold)[0]
        if len(feas) == 0:
            wins[j] = 0.0
            continue
        i = feas[-1]
        if i == len(amps) - 1:
            wins[j] = float(amps[-1])
            continue
        lo, hi = amps[i], amps[i + 1]
        f = lambda a: ireq(a, dirs[j])
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if f(mid) <= threshold:
                lo = mid
            else:
                hi = mid
        wins[j] = 0.5 * (lo + hi)
    return I_req, wins


def main():
    print("== N_sigma scan ==")
    nsig = {m: nsig_scan(m) for m in ("quad", "dipole")}
    for m, d in nsig.items():
        for k, v in d.items():
            row = "  ".join(f"{x:8.3f}" for x in v["nsig"])
            print(f"  {m:6s} {k:4s} sigma={v['sigma']:.3e}  Nsig: {row}")
            print("        good-field: " +
                  ", ".join(f"{tol*100:g}% -> {r:.2f} mm"
                            for tol, r in v["gf"].items()))

    print("\n== baselines ==")
    baselines = {}
    for mode in ("quad", "dipole"):
        for sh in (False, True):
            D, P = delivered(mode, sh, raw_solve(mode, uniform_bg(0.0, 0.0)))
            baselines[(mode, sh)] = (D, P)
            print(f"  baseline {mode:6s} {'sh' if sh else 'bare'}: "
                  f"D = {D:.4f}, purity = {P:.3e}")
    red = {mode: baselines[(mode, True)][0] / baselines[(mode, False)][0]
           for mode in ("quad", "dipole")}
    print("  shield reduction of the delivered field:",
          {k: round(v, 4) for k, v in red.items()})

    print("\n== background scans ==")
    scans = (
        ("dipole", "unif",
         lambda a, d: ireq_uniform("dipole", a, d), AMPS_UNIF, DIRS_UNIF),
        ("quad", "unif",
         lambda a, d: ireq_uniform("quad", a, d), AMPS_UNIF, DIRS_UNIF),
        ("quad", "grad",
         lambda a, d: ireq_gradient("quad", a, d), AMPS_GRAD, DIRS_GRAD),
        ("dipole", "grad",
         lambda a, d: ireq_gradient("dipole", a, d), AMPS_GRAD, DIRS_GRAD),
    )
    SAME = (("dipole", "unif"), ("quad", "grad"))
    res = {}
    for mode, kind, ireq, amps, dirs in scans:
        I_req, wins_bare = scan(ireq, amps, dirs, MAX_CURRENT)
        if (mode, kind) in SAME:
            wins_sh = wins_bare
        else:
            _, wins_sh = scan(ireq, amps, dirs, MAX_CURRENT * red[mode])
        res[(mode, kind)] = dict(ireq=I_req, bare=wins_bare, sh=wins_sh,
                                 dirs=dirs, amps=amps)
        print(f"  {mode:6s} {kind:4s}: bare window worst {wins_bare.min():.2f} "
              f"(dir {dirs[int(np.argmin(wins_bare))]:.0f}), "
              f"best {wins_bare.max():.2f}")
        print(f"                  shielded window worst {wins_sh.min():.2f} "
              f"(dir {dirs[int(np.argmin(wins_sh))]:.0f}), "
              f"best {wins_sh.max():.2f}")

    # ---- same-channel verification ---------------------------------------
    print("\n== delivered field and purity at the 1000 A point ==")
    same_max_ddev = same_max_pdev = 0.0
    edge_dev = 0.0
    nchecked = 0
    for mode, kind in SAME:
        d_worst = int(np.argmin(res[(mode, kind)]["bare"]))
        amp = 0.5 * res[(mode, kind)]["bare"][d_worst]
        if kind == "unif":
            th = np.radians(res[(mode, kind)]["dirs"][d_worst])
            bg = uniform_bg(amp * 1e-3 * np.cos(th), amp * 1e-3 * np.sin(th))
            Dx = B_REQ * 1e-3 - amp * 1e-3 * np.cos(th)
            Dy = -amp * 1e-3 * np.sin(th)
            ref_tgt = lambda x, y: (Dx, Dy)
        else:
            ph = res[(mode, kind)]["dirs"][d_worst]
            bg = gradient_bg(amp, ph)
            c, s = np.cos(2 * np.radians(ph)), np.sin(2 * np.radians(ph))
            Gn_b, Gs_b = amp * 1e-3 * c, -amp * 1e-3 * s
            ref_tgt = (lambda Gn, Gs:
                       (lambda x, y: (Gs * x + Gn * y, Gn * x - Gs * y)))(
                           G_REQ * 1e-3 - Gn_b, -Gs_b)
        I_raw = raw_solve(mode, bg)
        I_ref = raw_solve(mode, uniform_bg(0.0, 0.0), target=ref_tgt)
        for sh in (False, True):
            D, P = delivered(mode, sh, I_raw)
            Dr, Pr = delivered(mode, sh, I_ref)
            same_max_ddev = max(same_max_ddev, abs(D - Dr) / Dr)
            same_max_pdev = max(same_max_pdev, abs(P - Pr) / Pr)
            nchecked += 1
            print(f"  SAME {mode:6s} {kind:4s} amp={amp:6.2f} "
                  f"{'sh' if sh else 'bare'}: D = {D:.4f} (ref {Dr:.4f}, "
                  f"dev {abs(D-Dr)/Dr:.1e}), purity = {P:.3e} "
                  f"(ref {Pr:.3e}, dev {abs(P-Pr)/Pr:.1e})")

    # ---- window-edge verification for cross-channel cases ----------------
    for mode, kind in (("quad", "unif"), ("dipole", "grad")):
        for sh, tag in ((False, "bare"), (True, "sh")):
            d_worst = int(np.argmin(res[(mode, kind)][tag]))
            amp = res[(mode, kind)][tag][d_worst]   # window edge
            if kind == "unif":
                th = np.radians(res[(mode, kind)]["dirs"][d_worst])
                bg = uniform_bg(amp * 1e-3 * np.cos(th), amp * 1e-3 * np.sin(th))
            else:
                bg = gradient_bg(amp, res[(mode, kind)]["dirs"][d_worst])
            I_raw = raw_solve(mode, bg)
            D, P = delivered(mode, sh, I_raw)
            target_val = G_REQ if mode == "quad" else B_REQ
            edge_dev = max(edge_dev, abs(D - target_val) / target_val)
            nchecked += 1
            print(f"  EDGE {mode:6s} {kind:4s} amp={amp:6.2f} {tag}: "
                  f"D = {D:.4f} (benchmark {target_val:g}, "
                  f"dev {abs(D-target_val)/target_val:.1e})")
    print(f"  checked {nchecked} cases: same-channel delivered dev "
          f"{same_max_ddev:.1e}, purity dev {same_max_pdev:.1e}, "
          f"cross-channel edge dev {edge_dev:.1e}")

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16.0, 5.2))
    styles = [("quad", "bare", "-", "tab:blue", "quadrupole, bare"),
              ("quad", "sh", "--", "tab:blue", "quadrupole, shielded"),
              ("dipole", "bare", "-", "tab:orange", "dipole, bare"),
              ("dipole", "sh", "--", "tab:orange", "dipole, shielded")]
    for mode, key, ls, col, lbl in styles:
        ax[0].plot(RADII_SCAN, nsig[mode][key]["nsig"], ls, color=col, lw=1.5,
                   label=lbl)
    ax[0].axhline(1.0, color="0.4", ls=":", lw=1.0)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"Sampling radius $r$ (mm)")
    ax[0].set_ylabel(r"$N_sigma(r)$  (purity / purity at 1 mm)")
    ax[0].set_title("(a) Field-quality distance versus radius", fontsize=10)
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=0.3, which="both")

    for a, (mode, kind), unit in (
            (ax[1], ("dipole", "unif"), "mT"),
            (ax[2], ("quad", "grad"), "mT/mm")):
        d = res[(mode, kind)]
        worst = d["ireq"].max(axis=1) / MAX_CURRENT
        best = d["ireq"].min(axis=1) / MAX_CURRENT
        a.plot(d["amps"], worst, "-", color="tab:red", lw=1.6,
               label="worst direction")
        a.plot(d["amps"], best, "-", color="tab:green", lw=1.6,
               label="best direction")
        a.axhline(1.0, color="0.4", ls=":", lw=1.2, label="1000 A cap")
        a.set_xlabel(rf"Background amplitude ({unit})")
        a.set_ylabel(r"Required peak current / 1000 A")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    ax[1].set_title("(b) Dipole mode: uniform background", fontsize=10)
    ax[2].set_title("(c) Quadrupole mode: background gradient", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "background_field.png")

    # ---- macros -----------------------------------------------------------
    def idx(r):
        return int(np.argmin(np.abs(RADII_SCAN - r)))

    m = {}
    for mode, key in (("quad", "bare"), ("quad", "sh"), ("dipole", "bare"),
                      ("dipole", "sh")):
        tag = {"quad": "Q", "dipole": "D"}[mode] + ("b" if key == "bare" else "s")
        d = nsig[mode][key]
        m[f"SIG{tag}"] = "\\num{%.1e}" % d["sigma"]
        rsfx = {1: "One", 2: "Two", 5: "Five", 10: "Ten"}
        for r in RADII_NSIG:
            m[f"NSI{tag}{rsfx[int(r)]}"] = "%.1f" % d["nsig"][idx(r)]
        for tol in TOLS:
            v = d["gf"][tol]
            m[f"GFR{tag}{rsfx[int(tol * 100)]}"] = "%.2f" % v
    wu_d = res[("dipole", "unif")]
    wu_q = res[("quad", "unif")]
    wg_q = res[("quad", "grad")]
    wg_d = res[("dipole", "grad")]
    m["BGWdipWorst"] = "%.1f" % wu_d["sh"].min()
    m["BGWdipBest"] = "%.1f" % wu_d["sh"].max()
    m["BGWquadWorst"] = "%.2f" % wu_q["sh"].min()
    m["BGWquadBest"] = "%.2f" % wu_q["sh"].max()
    m["BGGquadWorst"] = "%.2f" % wg_q["sh"].min()
    m["BGGquadBest"] = "%.2f" % wg_q["sh"].max()
    m["BGGdipWorst"] = "%.2f" % wg_d["sh"].min()
    m["BGGdipBest"] = "%.2f" % wg_d["sh"].max()
    m["BGWquadWorstBare"] = "%.1f" % wu_q["bare"].min()
    m["BGWquadBestBare"] = "%.1f" % wu_q["bare"].max()
    m["BGGdipWorstBare"] = "%.2f" % wg_d["bare"].min()
    m["BGGdipBestBare"] = "%.2f" % wg_d["bare"].max()
    m["BGSameDelivDev"] = "\\num{%.0e}" % same_max_ddev
    m["BGSamePurityDev"] = "\\num{%.0e}" % same_max_pdev
    m["BGEdgeDev"] = "\\num{%.0e}" % edge_dev
    m["BGchecked"] = str(nchecked)
    m["BGSHIELDRED"] = "%.4f" % red["quad"]
    write_macros("results_background.tex", m)
    print("\nwrote ../PSXM_design_report/results_background.tex")


if __name__ == "__main__":
    main()

