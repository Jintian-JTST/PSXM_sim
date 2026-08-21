"""PSXM 自检脚本 —— 不依赖 PSXM_sim，只用 numpy，从 Biot-Savart 重建报告的核心数字。
   python3 selfcheck.py
"""
import numpy as np

mu0 = 4e-7 * np.pi
R, chord, N = 22.5, 20.0, 6            # mm, mm, coils
delta = np.arcsin(chord / 2 / R)       # 半张角 26.3878 deg
alpha = np.arange(N) * 2 * np.pi / N   # 线圈中心角
beta = np.concatenate([alpha + delta, alpha - delta])   # 12 根导线
sgn = np.concatenate([np.ones(N), -np.ones(N)])         # 每个线圈 (+I, -I)
wx, wy = R * np.cos(beta), R * np.sin(beta)


def Kmat(px, py):
    """(2n, 6) 系数矩阵, T/A. 位置单位 mm."""
    n = len(px)
    K = np.zeros((2 * n, N))
    for j in range(2 * N):
        dx, dy = px - wx[j], py - wy[j]
        r2 = dx**2 + dy**2
        K[:n, j % N] += sgn[j] * mu0 / (2 * np.pi) * (-dy) / r2 * 1e3
        K[n:, j % N] += sgn[j] * mu0 / (2 * np.pi) * (dx) / r2 * 1e3
    return K


nth = 48
th = np.arange(nth) * 2 * np.pi / nth
px, py = np.cos(th), np.sin(th)        # 1 mm 采样环
K = Kmat(px, py)
solve = lambda T: np.linalg.lstsq(K, T, rcond=None)[0]

print("=== 1. 场能力（最小二乘，1000 A 归一）===")
Id = solve(np.concatenate([np.ones(nth), np.zeros(nth)]))
Id = Id / np.abs(Id).max() * 1000
print("  dipole currents (A) :", np.round(Id, 1), "  <- 1000*cos(alpha_k)")
print("  B0  (mT)            :", (K @ Id)[:nth].mean() * 1e3, "  报告 23.7037")

Iq = solve(np.concatenate([py, px]))
Iq = Iq / np.abs(Iq).max() * 1000
Bq = K @ Iq
G = np.mean(Bq[nth:] * px + Bq[:nth] * py) / np.mean(px**2 + py**2) * 1e3
print("  quad currents (A)   :", np.round(Iq, 1), "  <- 1000*sin(2 alpha_k)/sin120")
print("  G   (mT/mm)         :", G, "  报告 2.179451")

print("\n=== 2. box ceiling（纯手算，不用求解器）===")
c1 = mu0 * 1000 / (2 * np.pi * R * 1e-3) * 2 * np.sin(delta) * 1e3
c2 = mu0 * 1000 / (2 * np.pi * (R * 1e-3) ** 2) * 2 * np.sin(2 * delta)
print("  dipole box  (mT)    :", c1 * 4, "        sum|cos a_k| = 4")
print("  dipole LS   (mT)    :", c1 * 3, "        sum cos^2 a_k = 3   -> 75.000 %")
print("  quad   box  (mT/mm) :", c2 * 4, "        (at phi = 15 deg)")
print("  quad   LS   (mT/mm) :", c2 * 2 * np.sqrt(3), "  = box at phi = 0 -> 100.000 %")
print("  ripple              :", (2 / np.sqrt(3) - 1) * 100, "%   报告 15.47")

print("\n=== 3. 场品质：污染是纯 m=4，按 (r/R)^2 长 ===")
for rr in (1.0, 5.0, 10.0):
    p = np.exp(1j * th) * rr
    B = Kmat(p.real, p.imag) @ Iq
    c = np.fft.fft(B[nth:] + 1j * B[:nth]) / nth
    rel = lambda m: abs(c[m - 1]) / abs(c[1])
    print(f"  r={rr:5.1f} mm  m4={rel(4):.4e}  m8={rel(8):.1e}"
          f"  (m3={rel(3):.0e}, m5={rel(5):.0e}, m6={rel(6):.0e})")

print("\n=== 4. 闭式解（完全不用求解器）===")
f = lambda m, Rs: 1 - (R / Rs) ** (2 * m)
print("  @ Rs=27.5: dipole (mT)    :", 23.703704 * f(1, 27.5), "  报告 7.836")
print("  @ Rs=27.5: quad (mT/mm)   :", 2.179451 * f(2, 27.5), "  报告 1.2028")
inv = lambda m, Ft, Fb: R * (1 - Ft / Fb) ** (-1 / (2 * m))
print("  quad  1 mT/mm -> Rs (mm)  :", inv(2, 1.0, 2.179451), "  报告 26.2331")
print("  dip   1 mT    -> Rs (mm)  :", inv(1, 1.0, 23.703704), "  报告 22.9902")
print("  dip   10 mT   -> Rs (mm)  :", inv(1, 10.0, 23.703704), "  报告 29.5918")

print("\n=== 5. 设计图 R_s,min(G*, I_max) ===")
print("  G*\\I     " + "".join(f"{i:>9d} A" for i in (750, 1000, 1500, 2000)))
for Gs in (0.5, 0.75, 1.0, 1.5):
    row = [inv(2, Gs, 2.179451 * I / 1000) for I in (750, 1000, 1500, 2000)]
    print(f"  {Gs:4.2f}   " + "".join(f"{v:11.1f}" for v in row))
