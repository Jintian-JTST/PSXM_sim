# PSXM_sim

2D magnetostatics toolkit for the inner PSXM (Passive Shielding auXiliary
Magnet), the pulsed steering magnet in the J-PARC muon $g-2$/EDM spiral
injection.  It simulates the six-coil ring with an optional conducting
shield can and inverse-solves the currents needed to hit a target field.
This repository is the code behind the design report
`Design Study of the Inner PSXM for Spiral Injection: Field Capability
and Shield-Radius Constraint`.

一个二维静磁工具包，用于内 PSXM（被动屏蔽辅助磁铁）——J-PARC 缪子 $g-2$/EDM
螺旋注入中的脉冲偏转磁铁。它仿真六线圈环（可选导体屏蔽层），并反解出能实现目标
磁场的电流。本仓库是设计报告《Design Study of the Inner PSXM for Spiral
Injection: Field Capability and Shield-Radius Constraint》的配套代码。

---

## Language / 语言
- [English](#english)
- [中文](#中文)

---

## English

### Physics model

Every current-carrying element is an **infinite straight wire
perpendicular to the x-y plane**, carrying current `I` (A) and piercing
the plane at `(x, y)` (mm).  The field of each wire follows the Biot-Savart
result `B = mu0 I / (2 pi rho)`, superposed over all wires.  This is a 2D
approximation: it ignores the finite axial length of real coils, so it is
accurate near the mid-plane over a transverse scale of order the coil
dimensions.  Units are consistent: **position in mm, current in A, field
in T**; gradients are quoted in mT/mm (= T/m).

### Repository layout

- **`coils.py`** — `Coils` base class and `as_array` / `MU0`: stores
  per-point `(x, y, I)` arrays and provides `B_field`, `B_magnitude`,
  `A_z` (vector potential, whose contours are the field lines) and `plot`.
- **`psxm_coils.py`** — `PSXMCoils(Coils)`: the six-coil PSXM ring
  (canonical geometry constants `RADIUS_MM`, `COIL_LENGTH_MM`).  Each
  physical coil pierces the plane at two legs (`+I`, `-I`); `group_matrix`
  maps the 6 physical currents to the 12 per-leg currents so the solver
  works on the real degrees of freedom.  `shield=True` adds `shield_n`
  current points around a circle of radius `shield_radius`, modelling the
  shield can.
- **`current_solver.py`** — `CurrentSolver`: builds the coefficient
  matrix `K` (`B = K @ I`) and solves the weighted least-squares inverse
  problem for the currents.  `solve()` returns the raw, unnormalised
  solution; `normalize_currents(I, max_current)` rescales to a hardware
  cap as a separate explicit step.
- **`config.py`** — the report's *adopted working assumptions* in one
  place: `MAX_CURRENT`, `G` (1 mT/mm quadrupole benchmark), `DIPOLE_TARGET`
  (provisional 1 mT dipole benchmark), `SHIELD_N`, and the B=0 sample-ring
  layout.
- **`shield.py`** — the ideal-sheet shield response: the quadrupole /
  dipole coil solves, the offset B=0 ring layout, the least-squares
  response matrix `S = -K_s^+ K_6`, and the ring-averaged `|B|`
  measurement.
- **`field_analysis.py`** — shared measurement helpers: `multipoles`
  (dipole + quadrupole fit on a sampling ring — every "achieved field"
  number in the report), `analytic_ceilings` / `unit_response` /
  `lp_ceiling` (hardware ceilings), plus `save_fig` / `write_macros`
  plumbing that writes figures and LaTeX macros for the report.
- **`field_capability.py`** — maximum reachable central field (hardware
  ceiling vs. field-quality-constrained design), and the cost of the
  shield at the centre.
- **`rot_quad.py`** — rotational quadrupole: currents at roll angle
  `phi` are `cos 2phi I_normal + sin 2phi I_skew` (verified against a full
  re-solve), the 15.5 % amplitude ripple, and the shield's effect on it.
- **`induction.py`** — the physical induced-eddy model (multipole
  moments x thin-shell L/R response).  This is the report's section 7.1
  "next step" and is **not** part of the current report's numbers.
- **`report_scan.py`**, **`report_robust.py`**, **`report_background.py`**,
  **`report_figures.py`** — the report pipeline; see below.
- **`verify_ls.py`** — checks the least-squares shield solve: round trip,
  forward map, conditioning, and the analytic ideal-shell cross-check.
- **`selfcheck.py`** — an independent, numpy-only first-principles check
  that rebuilds the report's core numbers directly from Biot-Savart
  (no solver modules).
- **`tracker/analyze_track.py`** — reads the spiral-injection tracking
  data (`trk20000.root`, not in the repository) and produces report
  Figure 1.  Requires `uproot`.

### Reproducing the design report

The report lives in the sibling directory `../PSXM_design_report/`; the
`report_*.py` scripts write their LaTeX macro files (`results_*.tex`,
`table_scan.tex`) and figures into it.  From the repository root:

```bash
pip install -r requirements.txt
python report_scan.py          # shield-radius scan      -> scan_1000A.npz
python report_robust.py        # convergence / robustness -> robust_scan.npz
python report_background.py    # field quality vs radius  -> results_background.tex
python report_figures.py       # all report figures       -> results_scan.tex, table_scan.tex
```

or run `./run_report.sh`.  `report_figures.py` draws every figure from
the cached `scan_1000A.npz` scan and from the same solver modules, so
figures and tables cannot disagree.  Figure 1 additionally needs
`tracker/analyze_track.py` + `trk20000.root`.

The analysis scripts `field_capability.py` and `rot_quad.py` may also be
run standalone for their own figures; the report's figures and macros come
from `report_figures.py`.

### Verification

- `python verify_ls.py` — solver self-checks and the analytic ideal-shell
  cross-check of report section 2.5.
- `python selfcheck.py` — rebuilds the core numbers (23.70 mT dipole,
  2.179 mT/mm quadrupole, the 26.233 / 22.990 / 29.6 mm thresholds, the
  15.5 % ripple) from first principles.

### Data files

- `scan_1000A.npz` — shield-radius scan cache written by `report_scan.py`.
- `robust_scan.npz` — convergence/robustness cache written by
  `report_robust.py`.

### Known limitations

- The 2D infinite-wire model has no axial (z) dependence: no end effects,
  and the exterior-field values it produces are discretisation residuals
  of the ideal flux-excluding shell, not a physical leakage prediction.
- The shield is an ideal current sheet; the physical induced-eddy response
  (finite conductivity, finite length, longitudinal slit) is deferred to
  the induction model in `induction.py`.
- `PSXMCoils`' mapping of the `I1..I6` labels / winding sign convention
  to the physical reference drawing is a documented best-effort
  approximation; adjust `start_angle` or the sign convention if it does
  not match real hardware.

---

## 中文

### 物理模型

每个带电流的元素都建模为一根**垂直于 x-y 平面的无限长直导线**，携带电流 `I`（A），
在平面上穿过 `(x, y)`（mm）。每根导线的场用毕奥-萨伐尔公式 `B = mu0 I / (2 pi rho)`
计算，再线性叠加。这是 2D 近似：忽略真实线圈的有限轴向长度，因此在中平面附近、
尺度与线圈尺寸相当的区域最准确。全代码库统一单位：**位置 mm、电流 A、磁场 T**；
梯度以 mT/mm（= T/m）给出。

### 仓库结构

- **`coils.py`** —— `Coils` 基类与 `as_array` / `MU0`：存储逐点 `(x, y, I)`
  数组，提供 `B_field`、`B_magnitude`、`A_z`（磁矢势，其等高线即磁场线）与 `plot`。
- **`psxm_coils.py`** —— `PSXMCoils(Coils)`：六线圈 PSXM 环（规范几何常量
  `RADIUS_MM`、`COIL_LENGTH_MM`）。每个物理线圈在平面上穿过两条"腿"（`+I`、`-I`）；
  `group_matrix` 把 6 个物理电流映射到 12 个逐腿电流，使求解器只解真实自由度。
  `shield=True` 在半径 `shield_radius` 的圆上额外加 `shield_n` 个电流点，模拟屏蔽壳。
- **`current_solver.py`** —— `CurrentSolver`：构造系数矩阵 `K`（`B = K @ I`），
  用加权最小二乘反解电流。`solve()` 返回**未归一化**的原始解；
  `normalize_currents(I, max_current)` 作为单独一步按硬件上限缩放。
- **`config.py`** —— 报告的"采纳工作假设"集中在一处：`MAX_CURRENT`、
  `G`（1 mT/mm 四极基准）、`DIPOLE_TARGET`（暂定的 1 mT 偶极基准）、`SHIELD_N`、
  以及 B=0 采样环布局。
- **`shield.py`** —— 理想电流片屏蔽响应：四极/偶极线圈求解、外移 B=0 环布局、
  最小二乘响应矩阵 `S = -K_s^+ K_6`、环平均 `|B|` 测量。
- **`field_analysis.py`** —— 共享测量工具：`multipoles`（采样环上的偶极+四极拟合——
  报告里每一个"实测场"数字都来自它）、`analytic_ceilings` / `unit_response` /
  `lp_ceiling`（硬件上限），以及 `save_fig` / `write_macros`（向报告写入图与 LaTeX 宏）。
- **`field_capability.py`** —— 最大可达中心场（硬件上限 vs 场品质约束设计），
  以及屏蔽在中心造成的代价。
- **`rot_quad.py`** —— 转角四极：转角 `phi` 处的电流为 `cos 2phi I_normal +
  sin 2phi I_skew`（逐角重解验证）、15.5% 幅度纹波、屏蔽对其的影响。
- **`induction.py`** —— 物理诱导涡流模型（多极矩 × 薄壳 L/R 响应）。这是报告
  7.1 节的"下一步工作"，**不属于**当前报告的数字。
- **`report_scan.py`**、**`report_robust.py`**、**`report_background.py`**、
  **`report_figures.py`** —— 报告管线，见下。
- **`verify_ls.py`** —— 校验最小二乘屏蔽求解：往返、正演、条件数、解析理想壳对照。
- **`selfcheck.py`** —— 独立的、仅用 numpy 的从头验证脚本，直接从毕奥-萨伐尔重建
  报告核心数字（不依赖求解器模块）。
- **`tracker/analyze_track.py`** —— 读螺旋注入跟踪数据（`trk20000.root`，不在仓库内），
  生成报告图 1。需要 `uproot`。

### 复现设计报告

报告在兄弟目录 `../PSXM_design_report/`；`report_*.py` 会把 LaTeX 宏文件
（`results_*.tex`、`table_scan.tex`）和图写入该目录。在仓库根目录：

```bash
pip install -r requirements.txt
python report_scan.py          # 屏蔽半径扫描        -> scan_1000A.npz
python report_robust.py        # 收敛 / 鲁棒性        -> robust_scan.npz
python report_background.py    # 场品质 vs 半径       -> results_background.tex
python report_figures.py       # 全部报告图           -> results_scan.tex, table_scan.tex
```

或直接运行 `./run_report.sh`。`report_figures.py` 的所有图都来自缓存的
`scan_1000A.npz` 扫描和同一批求解器模块，因此图与表不会互相矛盾。图 1 还需要
`tracker/analyze_track.py` + `trk20000.root`。

分析脚本 `field_capability.py` 和 `rot_quad.py` 也可以单独运行以生成各自的图；
报告的图与宏统一由 `report_figures.py` 产出。

### 验证

- `python verify_ls.py` —— 求解器自检与报告 2.5 节的解析理想壳对照。
- `python selfcheck.py` —— 从头重建核心数字（23.70 mT 偶极、2.179 mT/mm 四极、
  26.233 / 22.990 / 29.6 mm 阈值、15.5% 纹波）。

### 数据文件

- `scan_1000A.npz` —— `report_scan.py` 写出的屏蔽半径扫描缓存。
- `robust_scan.npz` —— `report_robust.py` 写出的收敛/鲁棒性缓存。

### 已知局限

- 2D 无限长导线模型没有轴向（z 方向）变化：无端部效应，其外场值只是理想磁通
  排除壳的离散化残差，不是物理漏场预言。
- 屏蔽是理想电流片；物理诱导涡流响应（有限电导率、有限长度、纵向缝隙）推迟到
  `induction.py` 中的感应模型。
- `PSXMCoils` 中 `I1..I6` 标签、绕线正负号约定与实物参考图的对应关系是文档化的
  "尽力而为"近似；若与硬件不符，可调整 `start_angle` 或正负号约定。
