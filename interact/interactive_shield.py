"""Interactive shielded rotational-quadrupole viewer for the PSXM.

Controls (each a slider + a type-in box):

- theta      : orientation of the rotated quadrupole target; the 6 coil
               currents are re-solved for it.
- sample r   : radius of the ring where the quadrupole target is imposed.
- shield s   : shield response scale. Induced currents I_shield = s*S@I_coil
               (S precomputed by nulling the field on and just outside the
               shield). s = 0 -> no shield, s = 1 -> full induced screening.
- shield R   : radius of the conducting shield can (mm). Changing it
               re-derives the shield response, so you can watch how the
               shielding and leakage change with shield size (task-4
               geometry knob).

Hovering reads B at the cursor and draws two arrows (length prop to |B|,
capped): blue = B, red = muon Lorentz force for a mu+ into the screen.

Usage:
    python interactive_shield.py
    python interactive_shield.py --shield-n 60
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add PSXM_sim/ to path

from coils import MU0
from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver
from shield_common import shield_zero_solver, SAMPLE_GAP_MM
from interactive_plot import _remove_contour_set, _HoverReadout, format_B

G = 1e-3              # target field gradient, T/mm
N_SAMPLE = 12        # sample points on the target ring
MAX_CURRENT = 1000.0  # coil currents rescaled so max|I| = this (A) for display
SHIELD_R_MAX = 420  # max shield radius (mm) for the slider
VIEW_MM = 450.0      # plot half-extent (mm): field lines + gray fill this whole box.
                     #   raise it to reach further out (e.g. 450 for the 0.45 m beam),
                     #   but the magnet then looks smaller.

# hover-arrow scaling (blue B arrow and red muon-force arrow):
#   length (mm) = ARROW_MM_PER_TESLA * |B| ,  capped at ARROW_MAX_MM
ARROW_MM_PER_TESLA = 400.0
ARROW_MAX_MM = 15.0

BENCH_R_MM = 419.0   # mm: show ring-averaged |B| at this radius (nearest-beam distance)


def rotated_quad_target(x, y, theta):
    """(Bx, By) [T] of a quadrupole field rotated by theta (rad)."""
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return G * (y * c - x * s), G * (x * c + y * s)


class ShieldedRotationPanel:
    """theta / r / s / shield-R controls -> solve coil currents for the
    rotated quadrupole, drive the shield via I_shield = s*S@I_coil, redraw."""

    def __init__(self, n_grid=300, n_levels=40, extent=4.0 / 3.0, sample_r=1.0,
                 shield_radius=27.5, shield_n=100, **coil_kwargs):
        self.coil_kwargs = dict(coil_kwargs, shield=True,
                                shield_radius=shield_radius, shield_n=shield_n)
        self.n_levels = n_levels
        self.n_grid = n_grid
        self.shield_n = shield_n
        self.shield_radius = shield_radius
        self.theta = 0.0
        self.sample_r = sample_r
        self.shield_scale = 1.0
        self._updating = False

        # grid AND view both span +/- VIEW_MM, so the field lines and the
        # gray region fill the whole plot instead of a tiny central blob.
        self._L = VIEW_MM
        xs = np.linspace(-self._L, self._L, n_grid)
        self._X, self._Y = np.meshgrid(xs, xs)

        self.tpl = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS), **self.coil_kwargs)
        self._precompute_shield_response()
        self._compute_az_bases()
        self._rebuild_Kcoil()

        self.fig = plt.figure(figsize=(11, 6.8))
        self.ax = self.fig.add_axes([0.06, 0.10, 0.54, 0.82])
        self.fig.text(0.33, 0.02,
                      "Hover: blue = B, red = muon force (μ+ into screen), length ∝ |B|",
                      ha="center", fontsize=8, color="gray")
        self._field_artists = []
        self._draw_static()
        self.coils = None

        # live shield arrows (last shield_n points of the template)
        self._shield_x = self.tpl.x[-shield_n:].copy()
        self._shield_y = self.tpl.y[-shield_n:].copy()
        self._shield_quiver = self.ax.quiver(
            self._shield_x, self._shield_y, np.zeros(shield_n), np.zeros(shield_n),
            angles="xy", scale_units="xy", scale=1, width=0.003, color="tab:red", zorder=3)

        self._build_widgets()
        self._hover = _HoverReadout(
            self.fig, self.ax, lambda: self.coils, show_force=True,
            arrow_scale=ARROW_MM_PER_TESLA, arrow_max=ARROW_MAX_MM, annot_fixed=True)
        self._refresh()

    # ----------------------------------------------------------- precompute
    def _precompute_shield_response(self, n_between=3, outer_offset=5.0):
        """B=0 samples on two rings offset radially from the shield (at
        SAMPLE_GAP_MM and outer_offset mm), not directly on the shield
        conductors -- see shield_common.shield_zero_solver; sampling on
        the conductors themselves sits in their 1/r near-field singularity
        and gives spurious currents."""
        tpl = self.tpl
        solver = shield_zero_solver(tpl, gap_mm=SAMPLE_GAP_MM, outer_mm=outer_offset,
                                    n_between=n_between)
        KM = solver.coefficient_matrix() @ tpl.group_matrix()
        K6 = KM[:, :PSXMCoils.N_COILS]
        Ksh = KM[:, PSXMCoils.N_COILS:]
        X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
        self._S = -X                       # (shield_n, 6): I_shield = S @ I_coil

    def _compute_az_bases(self):
        """Az-on-grid bases for the coil field and the shield response,
        on the fixed grid. Recomputed when the shield geometry changes."""
        coef = -MU0 / (4.0 * np.pi)
        tpl = self.tpl
        M = tpl.group_matrix()
        M6 = M[:, :PSXMCoils.N_COILS]
        dx = (self._X.ravel()[:, None] - tpl.x) * 1e-3
        dy = (self._Y.ravel()[:, None] - tpl.y) * 1e-3
        log_r2 = np.log(np.maximum(dx ** 2 + dy ** 2, 1e-24))
        self._az_basis = coef * (log_r2 @ M6)
        lr0 = np.log(np.maximum((tpl.x * 1e-3) ** 2 + (tpl.y * 1e-3) ** 2, 1e-24))
        self._az0_basis = coef * (lr0 @ M6)
        MshS = M[:, PSXMCoils.N_COILS:] @ self._S
        self._az_shield = coef * (log_r2 @ MshS)
        self._az0_shield = coef * (lr0 @ MshS)

    def _rebuild_Kcoil(self):
        """Coefficient matrix (coil DOF only) for solving the coil currents
        from the rotated-quadrupole target on the sample ring."""
        ang = np.linspace(0, 2 * np.pi, N_SAMPLE, endpoint=False)
        self.sx, self.sy = self.sample_r * np.cos(ang), self.sample_r * np.sin(ang)
        solver = CurrentSolver.from_current_source(self.tpl)
        for x, y in zip(self.sx, self.sy):
            solver.add_sample_point(x, y, 0.0, 0.0)
        K = solver.coefficient_matrix() @ self.tpl.group_matrix()
        self._Kcoil = K[:, :PSXMCoils.N_COILS]

    def _rebuild_shield(self):
        """Rebuild everything that depends on the shield radius."""
        self.coil_kwargs["shield_radius"] = self.shield_radius
        self.tpl = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS), **self.coil_kwargs)
        self._precompute_shield_response()
        self._compute_az_bases()
        # coil-solve K is coil-only, unaffected by shield geometry; skip.
        self._shield_x = self.tpl.x[-self.shield_n:].copy()
        self._shield_y = self.tpl.y[-self.shield_n:].copy()
        self._shield_quiver.set_offsets(np.column_stack([self._shield_x, self._shield_y]))
        th = np.linspace(0, 2 * np.pi, 200)
        self._shield_circle.set_data(self.shield_radius * np.cos(th),
                                     self.shield_radius * np.sin(th))

    def _solve_currents(self):
        Bx, By = rotated_quad_target(self.sx, self.sy, self.theta)
        B = np.concatenate([Bx, By])
        I, *_ = np.linalg.lstsq(self._Kcoil, B, rcond=None)
        peak = float(np.max(np.abs(I)))
        return I * (MAX_CURRENT / peak) if peak > 0 else I

    # -------------------------------------------------------------- widgets
    def _add_slider_box(self, y, label, vmin, vmax, vinit, on_change):
        s_ax = self.fig.add_axes([0.68, y, 0.19, 0.03])
        slider = Slider(s_ax, label, vmin, vmax, valinit=vinit)
        slider.valtext.set_visible(False)
        b_ax = self.fig.add_axes([0.89, y - 0.006, 0.09, 0.042])
        box = TextBox(b_ax, "", initial=f"{vinit:.4g}")
        slider.on_changed(lambda v: self._sync_from_slider(slider, box, v, on_change))
        box.on_submit(lambda t: self._sync_from_box(slider, box, t, on_change))
        return slider, box

    def _sync_from_slider(self, slider, box, val, on_change):
        if self._updating:
            return
        self._updating = True
        try:
            box.set_val(f"{val:.4g}")
        finally:
            self._updating = False
        on_change(val)

    def _sync_from_box(self, slider, box, text, on_change):
        if self._updating:
            return
        try:
            val = float(text)
        except ValueError:
            self._updating = True
            try:
                box.set_val(f"{slider.val:.4g}")
            finally:
                self._updating = False
            return
        self._updating = True
        try:
            slider.eventson = False
            slider.set_val(float(np.clip(val, slider.valmin, slider.valmax)))
            slider.eventson = True
        finally:
            self._updating = False
        on_change(val)

    def _build_widgets(self):
        self.fig.text(0.68, 0.90, "Controls", fontsize=11, fontweight="bold")
        self.theta_slider, self.theta_box = self._add_slider_box(
            0.82, r"$\theta$ (deg)", 0.0, 180.0, 0.0, self._on_theta)
        r_max = 0.9 * self.tpl.radius
        self.r_slider, self.r_box = self._add_slider_box(
            0.72, "sample r (mm)", 0.2, r_max, self.sample_r, self._on_r)
        self.s_slider, self.s_box = self._add_slider_box(
            0.62, "shield s", 0.0, 1.0, self.shield_scale, self._on_s)
        sr_min = self.tpl.radius + 1.0
        self.sr_slider, self.sr_box = self._add_slider_box(
            0.52, "shield R (mm)", sr_min, SHIELD_R_MAX, self.shield_radius, self._on_shield_radius)
        self.fig.text(0.68, 0.47, "s: 0=no shield, 1=full | R: shield can radius",
                      fontsize=8, color="gray")
        self._bench_text = self.fig.text(0.06, 0.955, "", fontsize=11,
                                         fontweight="bold", color="tab:red", zorder=3)

    def _on_theta(self, v):
        self.theta = np.deg2rad(v)
        self._refresh()

    def _on_r(self, v):
        self.sample_r = max(1e-3, float(v))
        self._rebuild_Kcoil()
        self._refresh()

    def _on_s(self, v):
        self.shield_scale = float(v)
        self._refresh()

    def _on_shield_radius(self, v):
        self.shield_radius = float(v)
        self._rebuild_shield()
        self._refresh()

    # --------------------------------------------------------------- redraw
    def _draw_static(self):
        ax, L, tpl = self.ax, self._L, self.tpl
        n_leg = 2 * PSXMCoils.N_COILS
        ax.plot(tpl.x[:n_leg], tpl.y[:n_leg], "ks", markersize=4)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(tpl.radius * np.cos(th), tpl.radius * np.sin(th), "k-", lw=1.2)
        self._shield_circle, = ax.plot(
            tpl.shield_radius * np.cos(th), tpl.shield_radius * np.sin(th), "k--", lw=0.8)
        for k in range(PSXMCoils.N_COILS):
            a = np.radians(tpl.center_angles[k])
            ax.text(tpl.radius * 1.15 * np.cos(a), tpl.radius * 1.15 * np.sin(a),
                    f"$I_{k + 1}$", ha="center", va="center", fontsize=11, clip_on=True)
        ax.set_xlim(-self._L, self._L)
        ax.set_ylim(-self._L, self._L)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

    def _update_shield_arrows(self, I_shield):
        max_abs = float(np.max(np.abs(I_shield)))
        if max_abs == 0:
            z = np.zeros_like(I_shield)
            self._shield_quiver.set_UVC(z, z)
            return
        ux = self._shield_x / self.shield_radius
        uy = self._shield_y / self.shield_radius
        length = 0.15 * self.shield_radius * np.abs(I_shield) / max_abs
        direction = np.sign(I_shield)
        self._shield_quiver.set_UVC(direction * ux * length, direction * uy * length)

    def _bench_field(self, n=24):
        """Ring-averaged |B| (T) at BENCH_R_MM (well outside the view)."""
        r = BENCH_R_MM
        vals = []
        for a in np.linspace(0.01, 2 * np.pi, n, endpoint=False):
            try:
                vals.append(self.coils.B_magnitude(r * np.cos(a), r * np.sin(a)))
            except ValueError:
                pass
        return float(np.mean(vals)) if vals else np.nan

    def _refresh(self):
        I_coil = self._solve_currents()
        I_shield = self.shield_scale * (self._S @ I_coil)
        self.coils = PSXMCoils(currents=I_coil, shield_currents=I_shield, **self.coil_kwargs)
        self._update_shield_arrows(I_shield)

        for cs in self._field_artists:
            _remove_contour_set(cs)
        self._field_artists = []
        az_flat = self._az_basis @ I_coil + self.shield_scale * (self._az_shield @ I_coil)
        az0 = float(self._az0_basis @ I_coil + self.shield_scale * (self._az0_shield @ I_coil))
        Az = az_flat.reshape(self._X.shape)
        if az0 > Az.min():
            self._field_artists.append(self.ax.contourf(
                self._X, self._Y, Az, levels=[Az.min(), az0],
                colors=["0.6"], alpha=0.4, zorder=0))
        self._field_artists.append(self.ax.contour(
            self._X, self._Y, Az, levels=self.n_levels, colors="k", linewidths=0.6))

        currents = "  ".join(f"$I_{k + 1}$={I_coil[k]:.0f}" for k in range(PSXMCoils.N_COILS))
        self.ax.set_title(
            f"θ = {np.degrees(self.theta):.0f}°   |   r = {self.sample_r:.2f} mm   |   "
            f"s = {self.shield_scale:.2f}   |   shield R = {self.shield_radius:.1f} mm\n{currents}",
            fontsize=9)
        self._bench_text.set_text(
            f"|B| at {BENCH_R_MM/1000:.3f} m (ring-avg): {format_B(self._bench_field())}")
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Interactive shielded rotational-quadrupole PSXM viewer.")
    p.add_argument("--n-grid", type=int, default=300, help="contour grid resolution")
    p.add_argument("--radius", type=float, default=22.5, help="coil ring radius, mm")
    p.add_argument("--coil-length", type=float, default=20.0, help="coil leg chord length, mm")
    p.add_argument("--shield-radius", type=float, default=27.5, help="initial shield can radius, mm")
    p.add_argument("--shield-n", type=int, default=100, help="number of shield current points")
    p.add_argument("--sample-r", type=float, default=1.0, help="initial sample-ring radius, mm")
    args = p.parse_args()
    ShieldedRotationPanel(
        n_grid=args.n_grid, radius=args.radius, coil_length=args.coil_length,
        shield_radius=args.shield_radius, shield_n=args.shield_n, sample_r=args.sample_r).show()


if __name__ == "__main__":
    main()
