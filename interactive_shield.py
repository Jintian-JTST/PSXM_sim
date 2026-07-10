"""Interactive PSXM viewer with the live conducting shield always on.

Builds on interactive_plot.PSXMControlPanel (shield mode):

- the shield can's induced currents are re-solved from the 6 main coil
  currents on every change — I_shield = s * S @ I, with the response
  matrix S precomputed once by least squares (zero field required on the
  can and just outside it), so updates are a single matrix-vector product;
- the red per-point shield arrows update in real time (length ∝ |I|,
  pointing inward for current into the page, outward for out of the page);
- an extra slider + text box controls the shield response scale s
  (slider spans 0..1: 0 = shield absent, 1 = fully induced currents;
  the text box accepts any value, e.g. 1.2 for an over-driven shield);
- each of the 6 coil currents keeps its own slider + text box, and the
  hover readout (|B|, direction, Bx, By at the cursor) includes the
  shield's contribution.

带实时屏蔽层的交互式查看器：主线圈电流一变，屏蔽层感应电流立即反解，红色
箭头实时更新；s 滑块/输入框控制屏蔽响应强度；悬停读数包含屏蔽层的贡献。

Usage:
    python interactive_shield.py                              # demo currents
    python interactive_shield.py 729.3 1000 270.7 -729.3 -1000 -270.7
    python interactive_shield.py --shield-radius 30 --shield-n 60
    python interactive_shield.py --n-grid 300                 # finer contours

From code:
    from interactive_shield import ShieldedPSXMPanel
    ShieldedPSXMPanel(currents=[...], shield_radius=27.5, shield_n=100).show()
"""

import argparse

from PSXM_coils import PSXMCoils
from interactive_plot import PSXMControlPanel


class ShieldedPSXMPanel(PSXMControlPanel):
    """
    PSXMControlPanel with the live shield simulation always enabled.

    shield_radius: radius (mm) of the conducting shield can.
    shield_n: number of discrete current points modeling the can. More
        points = smoother shield response; the response matrix and Az
        bases are precomputed, so runtime interactivity is unaffected.
    All other arguments are forwarded to PSXMControlPanel / PSXMCoils.
    """

    def __init__(self, currents=None, shield_radius=27.5, shield_n=100, **kwargs):
        if not kwargs.pop("shield", True):
            raise ValueError("ShieldedPSXMPanel always has shield=True; "
                             "use PSXMControlPanel for the unshielded viewer")
        super().__init__(currents=currents, shield=True,
                         shield_radius=shield_radius, shield_n=shield_n, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive PSXM viewer with a live-solved conducting shield.")
    parser.add_argument("currents", nargs="*", type=float, metavar="I",
                        help=f"initial coil currents I1..I{PSXMCoils.N_COILS} in A "
                             "(omit for demo values)")
    parser.add_argument("--shield-radius", type=float, default=27.5,
                        help="shield can radius in mm (default: 27.5)")
    parser.add_argument("--shield-n", type=int, default=100,
                        help="number of shield current points (default: 100)")
    parser.add_argument("--n-grid", type=int, default=200,
                        help="contour grid resolution (default: 200)")
    args = parser.parse_args()

    if args.currents and len(args.currents) != PSXMCoils.N_COILS:
        parser.error(f"expected {PSXMCoils.N_COILS} currents, got {len(args.currents)}")

    ShieldedPSXMPanel(
        currents=args.currents or None,
        shield_radius=args.shield_radius,
        shield_n=args.shield_n,
        n_grid=args.n_grid,
    ).show()


if __name__ == "__main__":
    main()
