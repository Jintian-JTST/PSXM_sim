"""Interactive field viewer with live current controls.

Opens the field-line plot in an interactive window with, for each of the
6 PSXM coils, a slider *and* a text box:

- drag a slider to change that coil's current — the plot redraws live;
- or type any number into the box next to it and press Enter (values
  outside the slider range are accepted; the slider just saturates);
- while the mouse is inside the plot, a tooltip follows the cursor
  showing the field at that point (|B|, direction angle, Bx, By) plus a
  blue arrow along the local field direction.

With shield=True (or the --shield flag), the conducting shield can is
simulated *live*: its induced currents are re-solved from the main coil
currents on every change (via a precomputed response matrix, so it's just
a matrix-vector product), the red per-point shield arrows update in real
time, and an extra slider + text box controls the shield response scale
s (0 = shield absent, 1 = fully induced currents that null the field on
and just outside the can).

交互式磁场查看器：每个线圈电流对应一个滑块和一个输入框；鼠标悬停实时读场。
开启屏蔽层（--shield）后，屏蔽层感应电流随主线圈电流实时反解、红色箭头实时
更新，并有独立的滑块和输入框控制屏蔽响应强度 s（0 = 无屏蔽，1 = 完全屏蔽）。

Performance notes:
- Az is linear in all currents, so per-coil Az basis fields on the grid —
  including the shield's *response* field per unit coil current — are
  precomputed once; every redraw is a tiny (n_grid^2 x 6) product plus
  contour tracing.
- The hover tooltip is blitted (only the tooltip + arrow region is
  repainted per mouse move) and rate-limited to 120 Hz so high-polling
  mice can't build up an event backlog.

Usage:
    python interactive_plot.py                       # demo currents, no shield
    python interactive_plot.py --shield              # live shield simulation
    python interactive_plot.py [--shield] I1 I2 I3 I4 I5 I6

From code:
    from interactive_plot import PSXMControlPanel, interactive_plot
    PSXMControlPanel(currents=[...], shield=True).show()
    interactive_plot(any_coils_object)               # hover readout only

Note: requires an interactive matplotlib backend (a normal desktop
session); it will not work in a headless environment.
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.widgets import Slider, TextBox

from coils import MU0
from current_solver import CurrentSolver
from PSXM_coils import PSXMCoils

DEMO_CURRENTS = [729.3, 1000.0, 270.7, -729.3, -1000.0, -270.7]


def format_B(value):
    """Format a field value (T) with an auto-scaled unit, sign preserved."""
    sign = "-" if value < 0 else ""
    mag = abs(value)
    for scale, unit in ((1.0, "T"), (1e-3, "mT"), (1e-6, "µT"), (1e-9, "nT")):
        if mag >= scale:
            return f"{sign}{mag / scale:.4g} {unit}"
    return f"{sign}{mag / 1e-9:.4g} nT"


def _readout_artists(ax, animated=False):
    """Create the (hidden) tooltip and direction-arrow artists on ax."""
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        fontsize=9, zorder=10, visible=False, animated=animated,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="0.5", alpha=0.9),
    )
    arrow = ax.quiver(
        [0.0], [0.0], [1.0], [0.0], angles="xy", scale_units="xy", scale=1,
        color="tab:blue", width=0.006, zorder=9, visible=False, animated=animated,
    )
    return annot, arrow


def _update_readout(event, ax, coils, annot, arrow, arrow_frac=0.08,
                    proportional=False, b_ref=None, cap=3.0, force_arrow=None):
    """
    Update the tooltip + arrow from a mouse-motion event.
    Returns True if the readout changed (needs re-blit/redraw).

    proportional: if True (and b_ref > 0), the arrow length is scaled by
    |B| / b_ref (capped at `cap` times the reference length) so the arrow
    grows/shrinks with the field magnitude. Default False keeps the
    fixed-length, direction-only arrow.
    """
    if event.inaxes is not ax or event.xdata is None:
        force_vis = force_arrow is not None and force_arrow.get_visible()
        if annot.get_visible() or arrow.get_visible() or force_vis:
            annot.set_visible(False)
            arrow.set_visible(False)
            if force_arrow is not None:
                force_arrow.set_visible(False)
            return True
        return False

    x, y = event.xdata, event.ydata
    try:
        Bx, By = coils.B_field(x, y)
    except ValueError:  # cursor sitting exactly on a conductor
        annot.xy = (x, y)
        annot.set_text(f"({x:.2f}, {y:.2f}) mm\non a conductor (field singular)")
        annot.set_visible(True)
        arrow.set_visible(False)
        if force_arrow is not None:
            force_arrow.set_visible(False)
        return True

    B_mag = float(np.hypot(Bx, By))
    angle = np.degrees(np.arctan2(By, Bx))  # direction, deg CCW from +x

    annot.xy = (x, y)
    annot.set_text(
        f"({x:.2f}, {y:.2f}) mm\n"
        f"|B| = {format_B(B_mag)}\n"
        f"dir = {angle:.1f}°\n"
        f"Bx = {format_B(Bx)},  By = {format_B(By)}"
    )
    annot.set_visible(True)

    if B_mag > 0:
        xlim = ax.get_xlim()
        length = arrow_frac * (xlim[1] - xlim[0])
        if proportional and b_ref and b_ref > 0:
            length *= min(B_mag / b_ref, cap)  # length now prop. to |B| (capped)
        arrow.set_offsets([[x, y]])
        arrow.set_UVC([length * Bx / B_mag], [length * By / B_mag])
        arrow.set_visible(True)
        if force_arrow is not None:
            # muon mu+ moving into the screen (v = -z): F = q v x B = e(By, -Bx),
            # perpendicular to B; |F| prop to |B|, so reuse the same length.
            force_arrow.set_offsets([[x, y]])
            force_arrow.set_UVC([length * By / B_mag], [length * (-Bx) / B_mag])
            force_arrow.set_visible(True)
    else:
        arrow.set_visible(False)
        if force_arrow is not None:
            force_arrow.set_visible(False)

    return True


class _HoverReadout:
    """
    Blitted hover readout: caches the static plot as a bitmap (recaptured
    automatically after every full draw) and, on mouse moves, restores it
    and redraws only the tooltip + arrow. Mouse motion therefore never
    triggers a full canvas redraw.

    get_coils: zero-argument callable returning the current Coils object
    (so the readout stays correct when the panel swaps coils).
    """

    # cap handler rate: high-polling mice (500-1000 Hz) would otherwise
    # queue up more motion events than we can draw, and the tooltip would
    # trail further and further behind the cursor
    MIN_INTERVAL = 1.0 / 120.0

    def __init__(self, fig, ax, get_coils, arrow_frac=0.08,
                 proportional=False, get_bref=None, show_force=False, cap=3.0):
        self.fig = fig
        self.ax = ax
        self.get_coils = get_coils
        self.arrow_frac = arrow_frac
        self.proportional = proportional
        self.get_bref = get_bref
        self.cap = cap
        self.annot, self.arrow = _readout_artists(ax, animated=True)
        if show_force:
            self.force_arrow = ax.quiver(
                [0.0], [0.0], [0.0], [0.0], angles="xy", scale_units="xy", scale=1,
                color="tab:red", width=0.006, zorder=9, visible=False, animated=True,
            )
        else:
            self.force_arrow = None
        self._bg = None
        self._last_region = None  # display-space bbox drawn last frame
        self._last_time = 0.0
        fig.canvas.mpl_connect("draw_event", self._on_draw)
        fig.canvas.mpl_connect("motion_notify_event", self._on_move)

    def _on_draw(self, event):
        # animated artists are excluded from full draws, so this snapshot
        # is always the clean background
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._last_region = None

    def _readout_region(self):
        """
        Display-space bbox covering the tooltip + arrow this frame, union
        the previous frame (so the old position gets erased on screen).
        Returns None if nothing needs blitting or it can't be computed.
        """
        try:
            renderer = self.fig.canvas.get_renderer()
        except AttributeError:  # non-Agg canvas; caller falls back to full blit
            return None

        boxes = []
        if self.annot.get_visible():
            boxes.append(Bbox(self.annot.get_window_extent(renderer).get_points()))
        if self.arrow.get_visible():
            # quiver window extent is unreliable; build a box around the
            # arrow base with radius = max arrow length in pixels
            px, py = self.ax.transData.transform(self.arrow.get_offsets()[0])
            reach = self.arrow_frac * (self.cap if self.proportional else 1.0)
            r = self.ax.bbox.width * reach * 1.6
            boxes.append(Bbox([[px - r, py - r], [px + r, py + r]]))

        new = Bbox.union(boxes).padded(12) if boxes else None
        if new is not None and self._last_region is not None:
            dirty = Bbox.union([new, self._last_region])
        else:
            dirty = new if new is not None else self._last_region
        self._last_region = new
        if dirty is None:
            return None
        clipped = Bbox.intersection(dirty, Bbox(self.fig.bbox.get_points()))
        return clipped if clipped is not None else dirty

    def _on_move(self, event):
        coils = self.get_coils()
        if coils is None:
            return

        # rate-limit only while inside the axes; leave/hide events always run
        now = time.perf_counter()
        if event.inaxes is self.ax:
            if now - self._last_time < self.MIN_INTERVAL:
                return
            self._last_time = now

        b_ref = self.get_bref() if self.get_bref is not None else None
        if not _update_readout(event, self.ax, coils, self.annot, self.arrow,
                               self.arrow_frac, self.proportional, b_ref,
                               self.cap, self.force_arrow):
            return
        if self._bg is None:  # no full draw yet; fall back
            self.fig.canvas.draw_idle()
            return

        canvas = self.fig.canvas
        canvas.restore_region(self._bg)
        self.ax.draw_artist(self.annot)
        self.ax.draw_artist(self.arrow)
        if self.force_arrow is not None:
            self.ax.draw_artist(self.force_arrow)
        # push only the small dirty region to the screen -- copying the
        # whole ~1100x650 px figure per mouse move is what feels "laggy"
        # on slower backends (especially TkAgg)
        region = self._readout_region()
        canvas.blit(region if region is not None else self.fig.bbox)
        canvas.flush_events()  # paint now, don't wait for the event loop


def _remove_contour_set(cs):
    """Remove a ContourSet from its axes (compatible across mpl versions)."""
    try:
        cs.remove()
    except (AttributeError, NotImplementedError):
        for coll in getattr(cs, "collections", []):
            coll.remove()


def interactive_plot(coils, arrow_frac=0.08, **draw_kwargs):
    """
    Hover-readout-only viewer for any Coils-like object (no current
    controls). draw_kwargs are forwarded to coils.draw() (n_grid,
    n_levels, extent, ...).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.text(0.5, 0.01, "Move the mouse inside the plot to read B at the cursor",
             ha="center", fontsize=8, color="gray")

    coils.draw(ax, **draw_kwargs)
    _HoverReadout(fig, ax, lambda: coils, arrow_frac)
    plt.show()


class PSXMControlPanel:
    """
    Interactive PSXM viewer: one slider + one text box per coil current,
    live plot redraw, and a blitted hover field readout at the cursor.

    With shield=True, the shield can is simulated live: on every change
    the induced shield currents are recomputed as I_shield = s * S @ I,
    where S is a response matrix (precomputed once by least-squares,
    requiring zero field on the can and 5 mm outside it) and s is the
    shield-response scale set by its own slider / text box. The red
    shield arrows and the field plot update in real time. Do not pass
    shield_currents — they are solved, not prescribed.

    currents: initial I1..I6 (A); defaults to DEMO_CURRENTS.
    n_grid: contour grid resolution. Thanks to the precomputed basis
        fields, higher values mainly cost contour tracing, not physics.
    slider_max: coil-slider range is [-slider_max, +slider_max]; defaults
        to max(1000, 1.2 * max|initial current|). Text boxes are not
        limited by this range.
    coil_kwargs: forwarded to PSXMCoils (radius, coil_length, start_angle,
        shield=..., shield_radius=..., shield_n=..., etc.).
    """

    def __init__(self, currents=None, n_grid=200, n_levels=40, arrow_frac=0.08,
                 slider_max=None, extent=4.0 / 3.0, **coil_kwargs):
        currents = DEMO_CURRENTS if currents is None else currents
        self.currents = np.asarray(currents, dtype=float).copy()
        if len(self.currents) != PSXMCoils.N_COILS:
            raise ValueError(f"expected {PSXMCoils.N_COILS} currents, got {len(self.currents)}")
        if "shield_currents" in coil_kwargs:
            raise ValueError("PSXMControlPanel solves shield currents live; don't pass shield_currents")

        self.coil_kwargs = coil_kwargs
        self.n_levels = n_levels
        self.shield_scale = 1.0
        self._updating = False  # guard against slider <-> textbox feedback loops

        tpl = PSXMCoils(currents=self.currents, **coil_kwargs)  # geometry template
        self.live_shield = bool(tpl.shield)

        self.fig = plt.figure(figsize=(11, 6.5))
        self.ax = self.fig.add_axes([0.06, 0.09, 0.50, 0.84])
        self.fig.text(0.66, 0.895, "Coil currents (A)", fontsize=11, fontweight="bold")
        self.fig.text(0.31, 0.015,
                      "Hover inside the plot to read B  |  drag a slider, or type a value and press Enter",
                      ha="center", fontsize=8, color="gray")

        if self.live_shield:
            self._precompute_shield_response(tpl)
        self._precompute_basis(tpl, n_grid, extent)
        self._draw_static_decorations(tpl)
        self._field_artists = []

        # fixed arrow-length reference so the shield arrows visibly grow /
        # shrink with the actual current magnitudes (incl. the s slider)
        if self.live_shield:
            ref = float(np.max(np.abs(self._S @ self.currents))) * self.shield_scale
            self._arrow_ref = ref  # 0 falls back to per-frame normalization

        self._build_widgets(slider_max)
        self.coils = None
        self._refresh()

        self._hover = _HoverReadout(self.fig, self.ax, lambda: self.coils, arrow_frac)

    # ---------------------------------------------------------------- setup

    def _precompute_shield_response(self, tpl, n_between=3, outer_offset=5.0):
        """
        Precompute the shield response matrix S (shield_n x 6) such that
        I_shield = S @ I_coils is the least-squares solution nulling the
        field at points on the shield circle and outer_offset mm outside
        it (n_between sample points between every pair of adjacent shield
        current points, as in example.py).
        """
        solver = CurrentSolver.from_current_source(tpl)
        gap = 360.0 / tpl.shield_n
        for radius in (tpl.shield_radius, tpl.shield_radius + outer_offset):
            for base_angle in tpl.shield_angles:
                for j in range(1, n_between + 1):
                    a = np.radians(base_angle + j * gap / (n_between + 1))
                    solver.add_sample_point(radius * np.cos(a), radius * np.sin(a), Bx=0.0, By=0.0)

        KM = solver.coefficient_matrix() @ tpl.group_matrix()
        K6 = KM[:, :PSXMCoils.N_COILS]     # field at samples per unit coil current
        Ksh = KM[:, PSXMCoils.N_COILS:]    # field at samples per unit shield current
        # minimize ||Ksh @ X - K6|| column-wise, then I_shield = -X @ I6
        X, *_ = np.linalg.lstsq(Ksh, K6, rcond=None)
        self._S = -X  # (shield_n, 6)

    def _precompute_basis(self, tpl, n_grid, extent):
        """
        Az on the grid is linear in the currents: precompute, once, the Az
        field of each coil at unit current — and, in live-shield mode, the
        Az of the shield's induced response per unit coil current — so
        every redraw is just a small matrix product.
        """
        coef = -MU0 / (4.0 * np.pi)
        M = tpl.group_matrix()
        M6 = M[:, :PSXMCoils.N_COILS]  # free vars -> per-point leg currents

        R = np.max(np.hypot(tpl.x, tpl.y))
        L = R * extent
        self._L = L
        xs = np.linspace(-L, L, n_grid)
        self._X, self._Y = np.meshgrid(xs, xs)

        dx = (self._X.ravel()[:, None] - tpl.x) * 1e-3  # mm -> m
        dy = (self._Y.ravel()[:, None] - tpl.y) * 1e-3
        log_r2 = np.log(np.maximum(dx**2 + dy**2, 1e-24))
        self._az_basis = coef * (log_r2 @ M6)  # (n_grid^2, 6)

        lr0 = np.log(np.maximum((tpl.x * 1e-3) ** 2 + (tpl.y * 1e-3) ** 2, 1e-24))
        self._az0_basis = coef * (lr0 @ M6)    # Az at origin (for the shading)

        if self.live_shield:
            # shield response folded down to per-coil basis: (points x 6)
            MshS = M[:, PSXMCoils.N_COILS:] @ self._S
            self._az_shield = coef * (log_r2 @ MshS)
            self._az0_shield = coef * (lr0 @ MshS)

        del dx, dy, log_r2  # free the big intermediate arrays

    def _draw_static_decorations(self, tpl):
        """Everything that doesn't depend on the currents, drawn once."""
        ax, L = self.ax, self._L

        ax.plot(tpl.x, tpl.y, "ks", markersize=4)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(tpl.radius * np.cos(theta), tpl.radius * np.sin(theta), "k-", linewidth=1.2)

        label_r = tpl.radius * 1.15
        for k in range(PSXMCoils.N_COILS):
            angle = np.radians(tpl.center_angles[k])
            ax.text(label_r * np.cos(angle), label_r * np.sin(angle),
                    f"$I_{k + 1}$", ha="center", va="center", fontsize=11, clip_on=True)

        if self.live_shield:
            ax.plot(tpl.shield_radius * np.cos(theta), tpl.shield_radius * np.sin(theta),
                    "k--", linewidth=0.8)
            self._shield_x = tpl.x[-tpl.shield_n:].copy()
            self._shield_y = tpl.y[-tpl.shield_n:].copy()
            self._shield_radius = tpl.shield_radius
            zeros = np.zeros(tpl.shield_n)
            # live-updated arrows: length prop. to |I|, pointing inward for
            # current into the page, outward for current out of the page
            self._shield_quiver = ax.quiver(
                self._shield_x, self._shield_y, zeros, zeros,
                angles="xy", scale_units="xy", scale=1, width=0.003,
                color="tab:red", zorder=3,
            )

        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Magnetic field lines (Az contours)")

    def _build_widgets(self, slider_max):
        if slider_max is None:
            slider_max = max(1000.0, 1.2 * float(np.max(np.abs(self.currents))))

        self.sliders, self.boxes = [], []
        for k in range(PSXMCoils.N_COILS):
            y = 0.80 - k * 0.09

            s_ax = self.fig.add_axes([0.66, y, 0.20, 0.035])
            slider = Slider(
                s_ax, f"$I_{k + 1}$", -slider_max, slider_max,
                valinit=float(np.clip(self.currents[k], -slider_max, slider_max)),
            )
            slider.valtext.set_visible(False)  # the text box shows the number
            slider.on_changed(lambda val, k=k: self._on_slider(k, val))

            b_ax = self.fig.add_axes([0.885, y - 0.006, 0.10, 0.047])
            box = TextBox(b_ax, "", initial=f"{self.currents[k]:.6g}")
            box.on_submit(lambda text, k=k: self._on_text(k, text))

            self.sliders.append(slider)
            self.boxes.append(box)

        if self.live_shield:
            self.fig.text(0.66, 0.30, "Shield response s  (0 = no shield, 1 = full)",
                          fontsize=10, fontweight="bold")
            s_ax = self.fig.add_axes([0.66, 0.24, 0.20, 0.035])
            self.shield_slider = Slider(s_ax, "$s$", 0.0, 1.0, valinit=self.shield_scale)
            self.shield_slider.valtext.set_visible(False)
            self.shield_slider.on_changed(self._on_shield_slider)

            b_ax = self.fig.add_axes([0.885, 0.234, 0.10, 0.047])
            self.shield_box = TextBox(b_ax, "", initial=f"{self.shield_scale:.4g}")
            self.shield_box.on_submit(self._on_shield_text)

    # ------------------------------------------------------------- redraw

    def _refresh(self):
        """Recompute shield currents, rebuild coils, redraw field + arrows."""
        if self.live_shield:
            I_shield = self.shield_scale * (self._S @ self.currents)
            self.coils = PSXMCoils(currents=self.currents, shield_currents=I_shield,
                                   **self.coil_kwargs)
            self._update_shield_arrows(I_shield)
        else:
            self.coils = PSXMCoils(currents=self.currents, **self.coil_kwargs)
        self._redraw_field()

    def _update_shield_arrows(self, I_shield):
        ref = self._arrow_ref if self._arrow_ref > 0 else float(np.max(np.abs(I_shield)))
        if ref == 0:
            zeros = np.zeros_like(I_shield)
            self._shield_quiver.set_UVC(zeros, zeros)
            return
        ux = self._shield_x / self._shield_radius
        uy = self._shield_y / self._shield_radius
        length = 0.15 * self._shield_radius * np.abs(I_shield) / ref
        direction = np.sign(I_shield)
        self._shield_quiver.set_UVC(direction * ux * length, direction * uy * length)

    def _redraw_field(self):
        """Swap in the contours for the current current vector (fast path)."""
        for cs in self._field_artists:
            _remove_contour_set(cs)
        self._field_artists = []

        az_flat = self._az_basis @ self.currents
        az0 = float(self._az0_basis @ self.currents)
        if self.live_shield:
            az_flat = az_flat + self.shield_scale * (self._az_shield @ self.currents)
            az0 += self.shield_scale * float(self._az0_shield @ self.currents)
        Az = az_flat.reshape(self._X.shape)

        if az0 > Az.min():
            self._field_artists.append(self.ax.contourf(
                self._X, self._Y, Az, levels=[Az.min(), az0],
                colors=["0.6"], alpha=0.4, zorder=0,
            ))
        self._field_artists.append(self.ax.contour(
            self._X, self._Y, Az, levels=self.n_levels, colors="k", linewidths=0.6,
        ))
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ callbacks

    def _on_slider(self, k, val):
        if self._updating:
            return
        self._updating = True
        try:
            self.boxes[k].set_val(f"{val:.6g}")  # keep the box in sync
        finally:
            self._updating = False
        self.currents[k] = val
        self._refresh()

    def _on_text(self, k, text):
        if self._updating:
            return
        try:
            val = float(text)
        except ValueError:
            # invalid input: restore the box to the current value
            self._updating = True
            try:
                self.boxes[k].set_val(f"{self.currents[k]:.6g}")
            finally:
                self._updating = False
            return

        self._updating = True
        try:
            slider = self.sliders[k]
            slider.eventson = False
            slider.set_val(float(np.clip(val, slider.valmin, slider.valmax)))
            slider.eventson = True
        finally:
            self._updating = False
        self.currents[k] = val
        self._refresh()

    def _on_shield_slider(self, val):
        if self._updating:
            return
        self._updating = True
        try:
            self.shield_box.set_val(f"{val:.4g}")
        finally:
            self._updating = False
        self.shield_scale = float(val)
        self._refresh()

    def _on_shield_text(self, text):
        if self._updating:
            return
        try:
            val = float(text)
        except ValueError:
            self._updating = True
            try:
                self.shield_box.set_val(f"{self.shield_scale:.4g}")
            finally:
                self._updating = False
            return

        self._updating = True
        try:
            self.shield_slider.eventson = False
            self.shield_slider.set_val(
                float(np.clip(val, self.shield_slider.valmin, self.shield_slider.valmax)))
            self.shield_slider.eventson = True
        finally:
            self._updating = False
        self.shield_scale = val
        self._refresh()

    def show(self):
        plt.show()


def main(argv):
    argv = list(argv)
    shield = "--shield" in argv
    if shield:
        argv.remove("--shield")

    if len(argv) == 0:
        currents = None
    elif len(argv) == PSXMCoils.N_COILS:
        currents = [float(a) for a in argv]
    else:
        sys.exit("usage: python interactive_plot.py [--shield] [I1 I2 I3 I4 I5 I6]")

    PSXMControlPanel(currents=currents, shield=shield).show()


if __name__ == "__main__":
    main(sys.argv[1:])
