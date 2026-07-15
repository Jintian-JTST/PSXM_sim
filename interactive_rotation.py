"""Interactive rotational-quadrupole viewer for the PSXM.

Two controls (each a slider *and* a type-in box):

- theta: quadrupole orientation. For each theta the 6 coil currents are
  re-solved so the centre field is the quadrupole rotated by theta:

      (Bx, By) = G * ( y*cos2θ - x*sin2θ ,  x*cos2θ + y*sin2θ )

  θ = 0 is the upright quadrupole; θ = 45° is the skew quadrupole.
- radius: the radius of the ring of sample points where the quadrupole
  target is imposed. Push it outward to probe how far the field stays a
  clean quadrupole.

The title shows the solved I1..I6 and the fit **residual as a percentage**
(relative L2 error between the achieved and target field on the sample
ring). The blue dashed circle marks the current sample ring; hovering
reads B at the cursor.

θ / radius 滑块或输入框实时调节；标题显示解出的 I1..I6 和百分制 residual
（采样环上"实现场 vs 目标场"的相对 L2 误差）。蓝色虚线圈是当前采样环。

Only the target field / sample ring changes with the controls — geometry,
group_matrix and the solver are the same as example.py.

Usage:
    python interactive_rotation.py
    python interactive_rotation.py --n-grid 300

Note: needs an interactive matplotlib backend (a normal desktop session).
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

from coils import MU0
from PSXM_coils import PSXMCoils
from current_solver import CurrentSolver
from interactive_plot import _remove_contour_set, _HoverReadout

G = 1e-3              # target field gradient, T/mm
N_SAMPLE = 12        # number of sample points on the ring
MAX_CURRENT = 1000.0  # solved currents are rescaled so max|I| = this (A) for display

# hover-arrow scaling (both the blue B arrow and the red muon-force arrow):
#   length (mm) = ARROW_MM_PER_TESLA * |B| ,  capped at ARROW_MAX_MM
ARROW_MM_PER_TESLA = 400.0   # proportionality: arrow length in mm per tesla of |B|
ARROW_MAX_MM = 15.0          # hard upper limit on arrow length, mm


def rotated_quad_target(x, y, theta):
    """(Bx, By) [T] of a quadrupole field rotated by theta (rad)."""
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return G * (y * c - x * s), G * (x * c + y * s)


class RotationPanel:
    """
    theta / radius controls -> re-solve the 6 coil currents for the
    rotated quadrupole target on the sample ring -> redraw field + residual.

    coil_kwargs (radius, coil_length, start_angle, ...) go to PSXMCoils.
    Shield is off: this viewer is about the centre field, not leakage.
    """

    def __init__(self, n_grid=220, n_levels=40, extent=4.0 / 3.0,
                 sample_r=1.0, **coil_kwargs):
        self.coil_kwargs = coil_kwargs
        self.n_levels = n_levels
        self.theta = 0.0            # rad
        self.sample_r = sample_r    # mm
        self._updating = False      # guard slider <-> box feedback
        self.tpl = PSXMCoils(currents=np.zeros(PSXMCoils.N_COILS), **coil_kwargs)

        self._precompute_basis(n_grid, extent)
        self._rebuild_K()           # builds self.sx/self.sy/self._K from sample_r

        self.fig = plt.figure(figsize=(8, 7.6))
        self.ax = self.fig.add_axes([0.10, 0.26, 0.80, 0.66])
        self.fig.text(0.5, 0.005,
                      "Hover: blue = B, red = muon force (μ+ into screen), length ∝ |B|",
                      ha="center", fontsize=8, color="gray")
        self._field_artists = []
        self._draw_static()
        self.coils = None

        # sample ring (updates with radius) + sample points
        self._ring, = self.ax.plot([], [], "b--", lw=1.0, zorder=2)
        self._ring_pts, = self.ax.plot([], [], "b.", ms=5, zorder=2)

        # --- widgets: theta and radius, each slider + type-in box ---
        self.theta_slider, self.theta_box = self._add_slider_box(
            0.135, r"$\theta$ (deg)", 0.0, 180.0, 0.0, self._on_theta)
        r_max = 0.9 * self.tpl.radius
        self.radius_slider, self.radius_box = self._add_slider_box(
            0.065, "sample r (mm)", 0.2, r_max, self.sample_r, self._on_radius)

        # hover: blue arrow = B, red arrow = muon force (mu+ into screen);
        # both lengths prop. to |B| (capped); readout box pinned to a corner.
        self._hover = _HoverReadout(
            self.fig, self.ax, lambda: self.coils, show_force=True,
            arrow_scale=ARROW_MM_PER_TESLA, arrow_max=ARROW_MAX_MM, annot_fixed=True)
        self._refresh()

    # ----------------------------------------------------------- precompute
    def _precompute_basis(self, n_grid, extent):
        """Az on the grid is linear in the currents: precompute each coil's
        unit-current Az once, so redraws are a tiny (n_grid^2 x 6) product."""
        coef = -MU0 / (4.0 * np.pi)
        M6 = self.tpl.group_matrix()[:, :PSXMCoils.N_COILS]
        R = np.max(np.hypot(self.tpl.x, self.tpl.y))
        self._L = R * extent
        xs = np.linspace(-self._L, self._L, n_grid)
        self._X, self._Y = np.meshgrid(xs, xs)
        dx = (self._X.ravel()[:, None] - self.tpl.x) * 1e-3  # mm -> m
        dy = (self._Y.ravel()[:, None] - self.tpl.y) * 1e-3
        log_r2 = np.log(np.maximum(dx ** 2 + dy ** 2, 1e-24))
        self._az_basis = coef * (log_r2 @ M6)                # (n_grid^2, 6)
        lr0 = np.log(np.maximum((self.tpl.x * 1e-3) ** 2 + (self.tpl.y * 1e-3) ** 2, 1e-24))
        self._az0_basis = coef * (lr0 @ M6)                  # Az at origin

    def _rebuild_K(self):
        """Sample points on the ring of radius sample_r, and the fixed-per-r
        coefficient matrix K (field at samples per physical coil current)."""
        ang = np.linspace(0, 2 * np.pi, N_SAMPLE, endpoint=False)
        self.sx, self.sy = self.sample_r * np.cos(ang), self.sample_r * np.sin(ang)
        solver = CurrentSolver.from_current_source(self.tpl)
        for x, y in zip(self.sx, self.sy):
            solver.add_sample_point(x, y, 0.0, 0.0)
        self._K = solver.coefficient_matrix() @ self.tpl.group_matrix()

    # -------------------------------------------------------------- widgets
    def _add_slider_box(self, y, label, vmin, vmax, vinit, on_change):
        s_ax = self.fig.add_axes([0.16, y, 0.54, 0.03])
        slider = Slider(s_ax, label, vmin, vmax, valinit=vinit)
        slider.valtext.set_visible(False)
        b_ax = self.fig.add_axes([0.79, y - 0.006, 0.12, 0.042])
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

    def _on_theta(self, val_deg):
        self.theta = np.deg2rad(val_deg)
        self._refresh()

    def _on_radius(self, val_mm):
        self.sample_r = max(1e-3, float(val_mm))
        self._rebuild_K()
        self._refresh()

    # --------------------------------------------------------------- redraw
    def _draw_static(self):
        ax, L, tpl = self.ax, self._L, self.tpl
        ax.plot(tpl.x, tpl.y, "ks", markersize=4)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(tpl.radius * np.cos(th), tpl.radius * np.sin(th), "k-", lw=1.2)
        for k in range(PSXMCoils.N_COILS):
            a = np.radians(tpl.center_angles[k])
            ax.text(tpl.radius * 1.15 * np.cos(a), tpl.radius * 1.15 * np.sin(a),
                    f"$I_{k + 1}$", ha="center", va="center", fontsize=11, clip_on=True)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

    def _refresh(self):
        # target field on the sample ring, then solve the 6 coil currents
        Bx, By = rotated_quad_target(self.sx, self.sy, self.theta)
        B = np.concatenate([Bx, By])
        I, *_ = np.linalg.lstsq(self._K, B, rcond=None)

        # residual as a percentage: relative L2 error of achieved vs target
        # field, using the RAW least-squares currents (scale-honest).
        B_fit = self._K @ I
        denom = np.linalg.norm(B)
        resid_pct = 100.0 * np.linalg.norm(B_fit - B) / denom if denom > 0 else 0.0

        # rescale only for display so the field shape is visible at any theta
        peak = float(np.max(np.abs(I)))
        I_disp = I * (MAX_CURRENT / peak) if peak > 0 else I
        self.coils = PSXMCoils(currents=I_disp, **self.coil_kwargs)

        # sample ring + points
        th = np.linspace(0, 2 * np.pi, 200)
        self._ring.set_data(self.sample_r * np.cos(th), self.sample_r * np.sin(th))
        self._ring_pts.set_data(self.sx, self.sy)

        # field contours
        for cs in self._field_artists:
            _remove_contour_set(cs)
        self._field_artists = []
        Az = (self._az_basis @ I_disp).reshape(self._X.shape)
        az0 = float(self._az0_basis @ I_disp)
        if az0 > Az.min():
            self._field_artists.append(self.ax.contourf(
                self._X, self._Y, Az, levels=[Az.min(), az0],
                colors=["0.6"], alpha=0.4, zorder=0))
        self._field_artists.append(self.ax.contour(
            self._X, self._Y, Az, levels=self.n_levels, colors="k", linewidths=0.6))

        currents = "  ".join(f"$I_{k + 1}$={I_disp[k]:.0f}" for k in range(PSXMCoils.N_COILS))
        self.ax.set_title(
            f"θ = {np.degrees(self.theta):.0f}°   |   sample r = {self.sample_r:.2f} mm   |   "
            f"residual = {resid_pct:.2f} %\n{currents}",
            fontsize=9)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Interactive rotational-quadrupole PSXM viewer.")
    p.add_argument("--n-grid", type=int, default=220, help="contour grid resolution")
    p.add_argument("--radius", type=float, default=22.5, help="coil ring radius, mm")
    p.add_argument("--coil-length", type=float, default=20.0, help="coil leg chord length, mm")
    p.add_argument("--sample-r", type=float, default=1.0, help="initial sample-ring radius, mm")
    args = p.parse_args()
    RotationPanel(n_grid=args.n_grid, radius=args.radius,
                  coil_length=args.coil_length, sample_r=args.sample_r).show()


if __name__ == "__main__":
    main()
