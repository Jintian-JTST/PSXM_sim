# PSXM_sim

A small 2D magnetostatics toolkit for simulating planar coil arrays (in
particular a "PSXM" 6-coil sensor with an optional conducting shield can),
and for inverse-solving the currents needed to hit a target field.

一个平面线圈阵列的 2D 磁场仿真小工具包（重点是一个 6 线圈的 "PSXM" 传感器，外加可选的导体屏蔽层），
并支持反解出能实现目标磁场分布的电流。

---

## Language / 语言
- [English](#english)
- [中文](#中文)

---

## English

### Physics model

Every current-carrying element (a "coil" in `Coils`) is modeled as an
**infinite straight wire perpendicular to the x-y plane**, carrying current
`I` (A) and piercing the plane at `(x, y)` (mm). The field of each wire
follows the standard Biot-Savart result for a straight wire,
`B = mu0*I / (2*pi*rho)`, superposed over all wires. This is a 2D
approximation: it ignores the finite axial length of real coils, so it's
most accurate near the mid-plane of a coil assembly whose axial extent is
large compared to the region of interest.

Units are consistent throughout the codebase: **position in mm, current in
A, field in Tesla**, magnetic vector potential `A_z` in T*m.

### Modules

- **`coils.py`** — `Coils`: the base class. Stores per-point `(x, y, I)`
  arrays and provides:
  - `B_field(x, y)` / `B_magnitude(x, y)` — magnetic flux density (T).
  - `A_z(x, y)` — magnetic vector potential; its contour lines are exactly
    the field lines, which is what `plot()` draws.
  - `plot(...)` — field-line plot (Az contours) with the region below the
    origin's potential shaded, saved to `path` or shown interactively.

- **`PSXM_coils.py`** — `PSXMCoils(Coils)`: the 6-coil PSXM sensor ring.
  Each of the 6 physical coils pierces the plane at two points (its "+"
  and "-" legs); `group_matrix()` exposes this as a linear map from 6
  physical currents to 12 per-leg currents, so a `CurrentSolver` can solve
  for the 6 real degrees of freedom instead of one DOF per leg. Optional
  `shield=True` adds `shield_n` extra current points around a
  `shield_radius` circle, modeling a conducting shield can whose currents
  are independent unknowns (also picked up by `group_matrix()`). `plot()`
  draws the coil ring, `I1..I6` labels and legend, and — if a shield is
  present — the shield boundary plus an arrow per shield current (length
  ∝ |I|, pointing inward for current into the page, outward for current
  out of the page).

- **`current_solver.py`** — `CurrentSolver`: given a set of **sample
  points** (positions with a prescribed target `Bx, By`) and a set of
  **current points** (positions with an unknown current), builds the
  linear coefficient matrix `K` (`B = K @ I`) and solves for `I` by
  weighted least squares. `from_current_source(source)` builds a solver
  directly from any `Coils`-like object that defines `group_matrix()`
  (e.g. `PSXMCoils`), so you solve for physical DOFs instead of raw
  per-point currents. `solve()` always returns the raw, unnormalized
  solution; use the static `normalize_currents(I, max_current)` to rescale
  for a hardware current limit as an explicit, separate step (so
  `predicted_field()` / `to_coils()` given the same `I_free` always stay
  consistent with what was actually solved).

- **`example.py`** — end-to-end demo: solves jointly for the 6 PSXM coil
  currents *and* 100 shield-can currents (106 unknowns), requiring a
  quadrupole field near the center and near-zero field on/just outside the
  shield can, then plots the result.

### Quick start

```python
from PSXM_coils import PSXMCoils

psxm = PSXMCoils(currents=[729.3, 1000, 270.7, -729.3, -1000, -270.7])
psxm.plot(path="fields.png")          # field-line plot with I1..I6 legend
print(psxm.B_field(0, 0))             # (Bx, By) in Tesla at the origin
```

```python
from current_solver import CurrentSolver

solver = CurrentSolver.from_current_source(psxm)
solver.add_sample_point(x=1, y=0, Bx=0.0, By=1e-3)   # target field at (1, 0) mm
I_free = solver.solve()                               # solve for I1..I6
solved = PSXMCoils(currents=I_free, radius=psxm.radius, coil_length=psxm.coil_length)
solved.plot(path="solved.png")
```

Run `python example.py` for the full shield + solver walkthrough.

### Known limitations

- The 2D infinite-wire model has no axial (z) dependence — it does not
  capture end effects of finite-length coils.
- `PSXMCoils`'s exact mapping from `I1..I6` labels / winding sign
  convention to the physical reference drawing is a documented
  best-effort approximation (see the class docstring); adjust
  `start_angle` or the sign convention if it doesn't match real hardware.
- Jointly fitting targets at very different physical scales (e.g. a small
  quadrupole target vs. a much larger natural field near a shield) is a
  real trade-off, not a solver bug — use `add_sample_point(..., weight=)`
  to balance it (see the comments in `example.py`).

---

## 中文

### 物理模型

每个带电流的点（`Coils` 里的一个"线圈"）都被建模成一根**垂直于 x-y 平面的无限长直导线**，
携带电流 `I`（单位 A），在平面上穿过位置 `(x, y)`（单位 mm）。每根导线的场用标准的
毕奥-萨伐尔直导线公式 `B = mu0*I / (2*pi*rho)` 计算，再线性叠加所有导线的贡献。这是一个
2D 近似模型：忽略了真实线圈在轴向（z 方向）的有限长度，因此在线圈组件轴向尺寸远大于关注区域时最准确。

全代码库统一单位：**位置用 mm，电流用 A，磁场用特斯拉（T）**，磁矢势 `A_z` 单位为 T·m。

### 各文件说明

- **`coils.py`** —— `Coils`：基类。存储每个点的 `(x, y, I)` 数组，提供：
  - `B_field(x, y)` / `B_magnitude(x, y)` —— 磁通密度（T）。
  - `A_z(x, y)` —— 磁矢势；它的等高线正是磁场线，这也是 `plot()` 画图所依据的原理。
  - `plot(...)` —— 磁场线图（Az 等高线），并将电势低于原点处的区域标灰阴影，可保存到 `path`
    或直接弹窗显示。

- **`PSXM_coils.py`** —— `PSXMCoils(Coils)`：6 线圈的 PSXM 传感器环。6 个物理线圈中的每一个
  都在平面上穿过两个点（它的 "+"、"-" 两条"腿"）；`group_matrix()` 把这个关系表示成一个从
  6 个物理电流到 12 个逐腿电流的线性映射，这样 `CurrentSolver` 就能求解真正的 6 个自由度，
  而不是每个点各算一个自由度。可选参数 `shield=True` 会在半径 `shield_radius` 的圆上额外加
  `shield_n` 个电流点，模拟导体屏蔽壳，这些电流是独立未知量（同样会被 `group_matrix()` 识别）。
  `plot()` 会画出线圈环、`I1..I6` 标签和图例，如果开启了屏蔽层，还会画出屏蔽层边界，以及每个
  屏蔽电流点的箭头（长度正比于 |I|，电流流入纸面箭头指向圆心，流出纸面箭头指向外侧）。

- **`current_solver.py`** —— `CurrentSolver`：给定一组**采样点**（指定位置上期望的目标
  `Bx, By`）和一组**电流点**（位置已知、电流未知），构造线性系数矩阵 `K`（满足 `B = K @ I`），
  用加权最小二乘求解 `I`。`from_current_source(source)` 可以直接从任何定义了 `group_matrix()`
  方法的 `Coils` 类对象（比如 `PSXMCoils`）构造求解器，这样解出的就是物理自由度而不是逐点电流。
  `solve()` 始终返回**未归一化**的原始解；如果需要按硬件电流上限缩放，用静态方法
  `normalize_currents(I, max_current)` 作为单独、显式的一步来处理（这样只要把同一个 `I_free`
  传给 `predicted_field()` / `to_coils()`，结果就始终和实际求解出的解一致，不会因为反复调用而
  悄悄解出不同的电流）。

- **`example.py`** —— 完整示例：联合求解 6 个 PSXM 线圈电流和 100 个屏蔽层电流（共 106 个
  未知量），要求中心处为四极场，且屏蔽层表面及外侧附近场接近零，最后画图展示结果。

### 快速上手

```python
from PSXM_coils import PSXMCoils

psxm = PSXMCoils(currents=[729.3, 1000, 270.7, -729.3, -1000, -270.7])
psxm.plot(path="fields.png")          # 画磁场线图，带 I1..I6 图例
print(psxm.B_field(0, 0))             # 原点处的 (Bx, By)，单位 T
```

```python
from current_solver import CurrentSolver

solver = CurrentSolver.from_current_source(psxm)
solver.add_sample_point(x=1, y=0, Bx=0.0, By=1e-3)   # (1, 0)mm 处的目标磁场
I_free = solver.solve()                               # 解出 I1..I6
solved = PSXMCoils(currents=I_free, radius=psxm.radius, coil_length=psxm.coil_length)
solved.plot(path="solved.png")
```

运行 `python example.py` 查看包含屏蔽层的完整求解流程。

### 已知局限

- 2D 无限长导线模型没有轴向（z 方向）的变化，无法反映有限长度线圈的端部效应。
- `PSXMCoils` 里 `I1..I6` 标签、绕线正负号约定与实物参考图的精确对应关系，是文档中说明过的
  "尽力而为"的近似（详见类的 docstring）；如果和实际硬件对不上，可以调整 `start_angle` 或正负号约定。
- 同时拟合物理尺度差异很大的目标（比如很小的中心四极场 vs. 屏蔽层附近大得多的自然场）本身
  是一个真实的权衡取舍，不是求解器的 bug——用 `add_sample_point(..., weight=)` 来平衡
  （可参考 `example.py` 里的注释）。
