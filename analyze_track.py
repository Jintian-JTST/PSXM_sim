"""Read the injection track (trk20000.root) and measure the turn-to-turn
spacing near the PSXM location (z ~ 1.1 m).

This is step 1 of the week-2 leakage study: we first need the physical
distance to the *neighbouring* turn of the beam near the PSXM, because
that is where the PSXM's leakage field must be evaluated.

Setup:
    pip install uproot awkward numpy matplotlib
Run (from the PSXM_sim folder, next to trk20000.root):
    python analyze_track.py

It prints the ROOT structure (trees / branches) and auto-detects r, z,
theta. If auto-detection fails, edit the candidate names in find_branch()
using the printed branch list.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import uproot
except ImportError:
    raise SystemExit("Need uproot:  pip install uproot awkward")

FNAME = "trk20000.root"
PSXM_Z = 1.10   # m, approximate PSXM location


def main():
    f = uproot.open(FNAME)
    print("Top-level keys:", f.keys())

    # pick the first TTree-like object
    tree, treename = None, None
    for k in f.keys():
        obj = f[k]
        if hasattr(obj, "keys") and getattr(obj, "num_entries", None) is not None:
            tree, treename = obj, k
            break
    if tree is None:
        raise SystemExit("No TTree found; inspect the keys printed above.")
    print(f"\nUsing tree: {treename}  ({tree.num_entries} entries)")
    print("Branches:", list(tree.keys()))

    def find_branch(cands):
        low = {b.lower(): b for b in tree.keys()}
        for c in cands:
            if c in low:
                return low[c]
        return None

    rb = find_branch(["r", "rho", "r_m", "rr"])
    zb = find_branch(["z", "z_m", "zz"])
    tb = find_branch(["theta", "phi", "th", "angle"])
    print(f"\nDetected  r: {rb}   z: {zb}   theta: {tb}")
    if not (rb and zb and tb):
        raise SystemExit("Could not auto-detect r/z/theta — edit find_branch() "
                         "candidates using the Branches list above.")

    arr = tree.arrays([rb, zb, tb], library="np")
    r = np.asarray(arr[rb], dtype=float).ravel()
    z = np.asarray(arr[zb], dtype=float).ravel()
    th = np.asarray(arr[tb], dtype=float).ravel()
    print(f"\nn points: {len(z)},  z range [{z.min():.3f}, {z.max():.3f}] m,  "
          f"r range [{r.min():.3f}, {r.max():.3f}] m")

    # unwrap theta and find z at each full 2*pi turn
    thu = np.unwrap(th)
    turn_z = []
    target = thu[0] + 2 * np.pi
    for i in range(1, len(thu)):
        while thu[i] >= target:
            denom = thu[i] - thu[i - 1]
            frac = (target - thu[i - 1]) / denom if denom != 0 else 0.0
            turn_z.append(z[i - 1] + frac * (z[i] - z[i - 1]))
            target += 2 * np.pi
    turn_z = np.array(turn_z)
    print(f"\n{len(turn_z)} full turns detected.")
    if len(turn_z) >= 2:
        dz = np.abs(np.diff(turn_z))
        mid = 0.5 * (turn_z[:-1] + turn_z[1:])
        k = int(np.argmin(np.abs(mid - PSXM_Z)))
        print(f"turn z (m): {np.round(turn_z, 3)}")
        print(f"\n>>> turn-to-turn spacing near z={PSXM_Z} m:  Δz = {dz[k]:.3f} m")
        print(f"    (between turns at z={turn_z[k]:.3f} and z={turn_z[k+1]:.3f} m)")
        print(">>> evaluate the PSXM leakage field at this distance.")

    # plot z vs turn number, mark PSXM
    plt.figure(figsize=(7, 5))
    plt.plot(thu / (2 * np.pi), z, ".", ms=2)
    plt.axhline(PSXM_Z, color="r", ls="--", label=f"PSXM  z = {PSXM_Z} m")
    plt.xlabel("turn number")
    plt.ylabel("z (m)")
    plt.title("Injection spiral: z vs turn")
    plt.legend()
    plt.tight_layout()
    plt.savefig("track_turns.png", dpi=150)
    print("\nsaved track_turns.png")


if __name__ == "__main__":
    main()
