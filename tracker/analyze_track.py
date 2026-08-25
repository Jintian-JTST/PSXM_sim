"""Injection-track analysis (trk20000.root): turn-to-turn spacing near the
PSXM and the true 3-D distance to the nearest other beam segment.

Near z = 1.1 m the beam is still on the incoming arc (full 2*pi turns only
start below z ~ 0.7 m), so the closest *other* beam segment found here is on
the same incoming arc, not on a later stored turn.  EXCLUDE_TURNS is expressed
in the full-trajectory average sampling interval (n / n_turns), so at z = 1.1 m
it covers only ~0.27 turns of the arc; the reported distance is therefore a
near-field diagnostic of the model (an "other segment on the incoming arc"),
not the distance to the storage region.  The true distance to the storage plane
(z < 0.05 m) is about 1.06 m.  The exterior-field values built on this radius
are numerical residuals of the discrete ideal-sheet model and do not enter the
design conclusion (see Sec. 5 of the report).

Setup:  pip install uproot awkward numpy matplotlib
Run:    python analyze_track.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import uproot

# trk20000.root lives in PSXM_sim/ (parent of this script's folder)
FNAME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trk20000.root")
PSXM_Z = 1.10           # m, approximate PSXM location
EXCLUDE_TURNS = 1.0     # ignore +/- this many turns around the PSXM point (same pass)


def main():
    tree = uproot.open(FNAME)["trk"]
    a = tree.arrays(["r", "theta", "z", "b_x", "b_y", "b_z"], library="np")
    r, th, z = a["r"].ravel(), a["theta"].ravel(), a["z"].ravel()
    bx, by, bz = a["b_x"].ravel(), a["b_y"].ravel(), a["b_z"].ravel()
    n = len(z)
    print(f"{n} points,  z [{z.min():.3f}, {z.max():.3f}] m,  r [{r.min():.3f}, {r.max():.3f}] m")

    # ---- turns: z at each full 2*pi of unwrapped theta ----
    thu = np.unwrap(th)
    turn_z, target = [], thu[0] + 2 * np.pi
    for i in range(1, n):
        while thu[i] >= target:
            d = thu[i] - thu[i - 1]
            frac = (target - thu[i - 1]) / d if d != 0 else 0.0
            turn_z.append(z[i - 1] + frac * (z[i] - z[i - 1]))
            target += 2 * np.pi
    turn_z = np.array(turn_z)
    n_turns = max(len(turn_z), 1)
    per_turn = n / n_turns
    print(f"{len(turn_z)} full turns,  ~{per_turn:.0f} samples/turn")
    if len(turn_z) >= 2:
        dz = np.abs(np.diff(turn_z))
        mid = 0.5 * (turn_z[:-1] + turn_z[1:])
        k = int(np.argmin(np.abs(mid - PSXM_Z)))
        print(f"nearest full-turn pitch to z={PSXM_Z} m: Δz = {dz[k]:.3f} m "
              f"(turns at z={turn_z[k]:.3f}, {turn_z[k+1]:.3f})")

    # ---- 3-D nearest other beam segment to the PSXM point ----
    x = r * np.cos(th); y = r * np.sin(th)
    P = np.column_stack([x, y, z])
    i0 = int(np.argmin(np.abs(z - PSXM_Z)))
    p0 = P[i0]
    print(f"\nPSXM point idx={i0}: r={r[i0]:.3f}, z={z[i0]:.3f} m,  "
          f"|B|_beam={np.sqrt(bx[i0]**2+by[i0]**2+bz[i0]**2)*1e3:.1f} mT (main storage field)")
    d = np.linalg.norm(P - p0, axis=1)
    excl = int(EXCLUDE_TURNS * per_turn)
    d = np.where(np.abs(np.arange(n) - i0) > excl, d, np.inf)
    j = int(np.argmin(d))
    print(f">>> nearest OTHER beam segment (index exclusion +/-{EXCLUDE_TURNS} turn): "
          f"{d[j]:.3f} m  (at z={z[j]:.3f}, r={r[j]:.3f} m, turn={thu[j]/(2*np.pi):.2f})")
    print(">>> NOTE: this lies on the incoming arc, not in the storage region;")
    print(">>> distance to the first storage-plane samples (z < 0.05 m) is %0.3f m"
          % np.min(d[z < 0.05]))
    print(">>> evaluate the PSXM leakage field around this near-field diagnostic radius.")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].plot(thu / (2 * np.pi), z, ".", ms=1.5)
    ax[0].axhline(PSXM_Z, color="r", ls="--", label=f"PSXM z={PSXM_Z} m")
    ax[0].set_xlabel("turn number"); ax[0].set_ylabel("z (m)")
    ax[0].set_title("injection spiral: z vs turn"); ax[0].legend()

    ax[1].plot(r, z, ".", ms=1.5, color="0.6", label="beam")
    ax[1].plot(r[i0], z[i0], "r*", ms=14, label="PSXM")
    ax[1].plot(r[j], z[j], "bo", ms=8, mfc="none", label="nearest other segment (arc)")
    ax[1].annotate(f"{d[j]:.3f} m", (r[j], z[j]), textcoords="offset points",
                   xytext=(8, 8), fontsize=8)
    ax[1].set_xlabel("r (m)"); ax[1].set_ylabel("z (m)")
    ax[1].set_title(f"nearest other segment on the arc = {d[j]:.3f} m"); ax[1].legend()
    fig.tight_layout()
    fig.savefig("figures/track_turn_spacing.png", dpi=150)
    print("\nsaved figures/track_turn_spacing.png")


if __name__ == "__main__":
    main()
