"""verify_ls.py -- validate the least-squares shield calculation, 4 tests.

T1 round-trip   : known currents -> field at samples -> LS solve -> must
                  recover the currents exactly (code correctness).
T2 forward check: K @ I  vs  PSXMCoils.B_field at the same points (the
                  matrix and the field routine must agree to float precision).
T3 conditioning : cond(K_shield) and singular-value spread of the shield
                  response problem (is it well-posed?).
T4 physics      : LS shield surface current K_LS(theta) vs the ANALYTIC
                  perfect-conductor solution K_ideal(theta); repeated for
                  several shield_n to show (non-)convergence.  This is the
                  test that answers "why do the two methods differ".

Run:  python verify_ls.py     (prints a PASS/FAIL summary, saves verify_ls.png)
"""

import numpy as np
import matplotlib.pyplot as plt

from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver

MU0 = 4e-7 * np.pi
G = 1e-3                 # T/mm quadrupole gradient (target)
I_COIL = 1000.0
SAMPLE_GAP_MM = 5.0      # radial gap between shield currents and B=0 samples
OUTER_MM = 5.0
N_BETWEEN = 3
M_MODES = 8


# ---------------------------------------------------------------- helpers
def quad_currents():
    return np.array([0, I_COIL, I_COIL, 0, -I_COIL, -I_COIL], float)


def build_shield_solver(tpl, gap_mm=SAMPLE_GAP_MM):
    """B=0 samples on (shield+gap) and (shield+OUTER) rings, in the azimuthal
    gaps between shield points. Returns solver and K blocks."""
    solver = CurrentSolver.from_current_source(tpl)
    gap = 360.0 / tpl.shield_n
    for radius in (tpl.shield_radius + gap_mm, tpl.shield_radius + OUTER_MM):
        for base in tpl.shield_angles:
            for j in range(1, N_BETWEEN + 1):
                a = np.radians(base + j * gap / (N_BETWEEN + 1))
                solver.add_sample_point(radius * np.cos(a), radius * np.sin(a), 0.0, 0.0)
    KM = solver.coefficient_matrix() @ tpl.group_matrix()
    return KM[:, :PSXMCoils.N_COILS], KM[:, PSXMCoils.N_COILS:]


def analytic_K(theta, a_m, coil_currents):
    """Perfect-conductor surface current K_ideal(theta) [A/m] (continuous shell)."""
    legs = PSXMCoils(currents=coil_currents)
    z = (np.asarray(legs.x) + 1j * np.asarray(legs.y)) * 1e-3
    Ileg = np.asarray(legs.I)
    K = np.zeros_like(theta)
    for m in range(1, M_MODES + 1):
        C = np.sum(Ileg * z ** m)
        K += -np.real(C * np.exp(-1j * m * theta)) / (np.pi * a_m ** (m + 1))
    return K


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
    K6, Ksh = build_shield_solver(tpl)
    I_c = quad_currents()
    rng = np.random.default_rng(0)
    I_s = rng.normal(0, 100, tpl.shield_n)
    # rebuild the same sample points to evaluate B_field directly
    pts = []
    gap = 360.0 / tpl.shield_n
    for radius in (tpl.shield_radius + SAMPLE_GAP_MM, tpl.shield_radius + OUTER_MM):
        for base in tpl.shield_angles:
            for j in range(1, N_BETWEEN + 1):
                a = np.radians(base + j * gap / (N_BETWEEN + 1))
                pts.append((radius * np.cos(a), radius * np.sin(a)))
    coils = PSXMCoils(currents=I_c, shield=True, shield_n=tpl.shield_n,
                      shield_currents=I_s)
    B_direct = np.array([coils.B_field(x, y) for x, y in pts])   # (n,2)
    B_matrix = K6 @ I_c + Ksh @ I_s                              # stacked [Bx..., By...]
    n = len(pts)
    err = np.max(np.abs(np.concatenate([B_direct[:, 0], B_direct[:, 1]]) - B_matrix))
    scale = np.max(np.abs(B_matrix))
    ok = err / scale < 1e-10
    print(f"T2 forward consistency : max rel err = {err/scale:.2e}   {'PASS' if ok else 'FAIL'}")
    return ok


def t3_conditioning():
    tpl = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=100)
    _, Ksh = build_shield_solver(tpl)
    s = np.linalg.svd(Ksh, compute_uv=False)
    cond = s[0] / s[-1]
    print(f"T3 conditioning        : cond(K_shield) = {cond:.2e}  "
          f"(sv range {s[-1]:.2e} .. {s[0]:.2e})")
    print( "                         (>1e10 would mean ill-posed; large but finite is OK with lstsq)")
    return True


def t4_physics(shield_ns=(60, 100, 200, 400)):
    """LS shield current vs analytic perfect-conductor K, for several shield_n."""
    I_c = quad_currents()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    th_plot = np.linspace(0, 2 * np.pi, 361)
    ratios = []
    for sn in shield_ns:
        tpl = PSXMCoils(currents=np.zeros(6), shield=True, shield_n=sn)
        K6, Ksh = build_shield_solver(tpl)
        X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
        I_s = (-X) @ I_c                                   # LS shield currents (A)
        seg = 2 * np.pi * (tpl.shield_radius * 1e-3) / sn  # m per point
        K_LS = I_s / seg                                   # A/m
        th_pts = np.radians(tpl.shield_angles)
        ax[0].plot(np.degrees(th_pts), K_LS, ".", ms=3, label=f"LS shield_n={sn}")
        Kan_pts = analytic_K(th_pts, tpl.shield_radius * 1e-3, I_c)
        ratios.append(np.max(np.abs(K_LS)) / np.max(np.abs(Kan_pts)))
    a_m = 27.5e-3
    ax[0].plot(np.degrees(th_plot), analytic_K(th_plot, a_m, I_c), "k-", lw=1.5,
               label="analytic (perfect conductor)")
    ax[0].set_xlabel("θ (deg)"); ax[0].set_ylabel("K (A/m)")
    ax[0].set_title("T4: LS surface current vs analytic")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].plot(shield_ns, ratios, "o-")
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8, label="perfect agreement")
    ax[1].set_xlabel("shield_n"); ax[1].set_ylabel("peak K_LS / peak K_analytic")
    ax[1].set_title("T4: convergence toward the analytic solution")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("verify_ls.png", dpi=150)
    print(f"T4 physics             : peak(K_LS)/peak(K_analytic) = "
          + ", ".join(f"{sn}:{r:.3f}" for sn, r in zip(shield_ns, ratios)))
    print( "                         -> 1.0 means the LS current IS the physical screening current")
    print( "saved verify_ls.png")
    return True


if __name__ == "__main__":
    t1_roundtrip()
    t2_forward()
    t3_conditioning()
    t4_physics()
