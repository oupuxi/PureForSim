"""
calculate_blast_parameters.py

该模块用于计算爆炸波形相关的物理参数，例如 Friedlander 波形中的时间常数、衰减系数，
冲击波正相位持续时间、峰值超压等。
"""
import math
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar

from src.utils.Kingery import KingeryBulmashModel


# β
def calculate_friedlander_beta(charge_weight: float,
                               path: List[Tuple[float, float]],
                               decay_factor: float = 0.85,
                               beta_min: float = 0.25) -> float:
    """根据路径获取 Friedlander 衰减系数 β。

    * **自由场**: 当 ``path`` 仅含两个节点(爆心→自由场点)，
      直接从 KB 表读取 ``Ps, td, I`` 数值反推 β (Brent 根求解)。
    * **多段反射**: 首段自由场反推得到 β₀，随后每次反射 β 乘以
      ``decay_factor``，并强制不低于 ``beta_min``。

    参数
    ------
    charge_weight: 爆炸当量（kg TNT）。
    path: 坐标序列 ``[(x0,y0), (x1,y1), ...]``。
           - len==2 ⇒ 自由场  (β₀ 数值反推)
           - len>2  ⇒ 多次反射 (β = β₀ × decay_factor^(n‑1))
    decay_factor: 每次反射的 β 衰减系数(默认 0.85)。
    beta_min:     β 最小值下限，避免过小(默认 0.25)。

    返回
    ------
    float: 终点处 β (dimensionless)。
    """
    if len(path) < 2:
        raise ValueError("路径至少应包含爆心与一个目标点")

    # ── 首段自由场距离 ──
    R_free = math.hypot(path[1][0] - path[0][0], path[1][1] - path[0][1])
    kb = KingeryBulmashModel(neq=charge_weight, distance=R_free)
    Ps_kpa = kb.incident_pressure       # kPa
    td_ms  = kb.positive_phase_duration # ms
    I_kpams = kb.incident_impulse       # kPa·ms

    # ---- 数值反推 β₀ ----
    def residual(beta):
        if beta <= 0:
            return np.inf
        return (Ps_kpa * td_ms / beta) * (1 - 1 / np.exp(beta)) - I_kpams

    sol = root_scalar(residual, bracket=(0.1, 5.0), method="brentq")
    if not sol.converged:
        raise RuntimeError("β 反求未收敛，请检查 KB 输入区间")
    beta0 = sol.root

    if len(path) == 2:
        # 只有自由场
        return beta0

    # ---- 多段反射：按反射次数衰减 ----
    reflection_count = len(path) - 2  # 后续段数 = 反射次数
    beta_final = beta0 * (decay_factor ** reflection_count)
    return max(beta_final, beta_min)
# td
def calculate_friedlander_td(charge_weight: float,
                             prev_distance: float,
                             distance: float,
                             prev_td: float = None,
                             is_free_field: bool = True,
                             td_max: float = 200.0) -> float:
    """
    计算正相位持续时间 td（单位：ms）

    参数:
        charge_weight (float): 炸药当量（kg TNT），仅自由场时需要
        prev_distance (float): 上一段传播距离（m）
        distance (float): 当前传播距离（m）
        prev_td (float): 上一段 td，仅反射传播时需要
        is_free_field (bool): 是否为自由场，默认为 True

    返回:
        float: 当前传播位置的 td（正相位持续时间，单位ms）
    """
    if is_free_field:
        if charge_weight is None:
            raise ValueError("自由场计算必须提供 charge_weight")
        if distance <= 0:
            raise ValueError("distance 必须为正数")
        model = KingeryBulmashModel(neq=charge_weight, distance=distance, unit_system='metric')
        return model.positive_phase_duration

    else:
        if prev_td is None or prev_distance is None or distance is  None:
            raise ValueError("反射传播计算 td 需要提供 prev_td 和 prev_distance和distance")
        if prev_distance <= 0 or distance <= 0:
            raise ValueError("传播距离必须为正数")

        ratio = distance / prev_distance
        td = prev_td * math.log1p(ratio)
        return min(td, td_max)



# 到达时间ta
def calculate_friedlander_time_arrive(charge_weight: float,
                                      path: List[Tuple[float, float]],
                                      sound_speed: float = 343.0,
                                      speed_scale: float = 1.1) -> float:
    """
    计算 Friedlander 模型的冲击波到达时间（单位：ms）

    参数:
        path (List[Tuple[float, float]]): 坐标路径，起点为爆炸点，第一个点为自由场终点
        charge_weight (float): 炸药当量（kg TNT），用于查询 KB 表
        sound_speed (float): 基础声速（m/s），默认 343
        speed_scale (float): 超声速系数，默认 1.1 倍声速

    返回:
        float: 总到达时间（单位 ms）
    """

    if len(path) < 2:
        raise ValueError("路径必须至少包含两个点")

    # --- 自由场段 ---
    x0, y0 = path[0]
    x1, y1 = path[1]
    dist0 = math.hypot(x1 - x0, y1 - y0)

    try:
        model = KingeryBulmashModel(neq=charge_weight, distance=dist0)
        t0_ms = model.time_of_arrival  # 单位 ms
    except Exception as e:
        raise RuntimeError(f"KB 表查找失败：{e}")

    # --- 反射段 ---
    t_extra_ms = 0.0
    v = sound_speed * speed_scale  # m/s

    for i in range(1, len(path) - 1):
        xa, ya = path[i]
        xb, yb = path[i + 1]
        segment_dist = math.hypot(xb - xa, yb - ya)
        t_extra_ms += (segment_dist / v) * 1000  # s → ms

    return t0_ms + t_extra_ms
# Ps
def calculate_friedlander_peak_overpressure(charge_weight: float,
                                            path: List[Tuple[float, float]],
                                            decay_exponent: float = 1.5,
                                            wall_reflect_factor: float = 1.0) -> float:
    """根据多段路径计算峰值超压 Ps（单位：kPa）。

    思路说明
    --------
    1. **第一段（爆心 → 第一反射点）**
       - 直接调用 Kingery‑Bulmash 表，获取 "反射峰压" `Ps0`。
    2. **后续每一段（反射传播）**
       - 用距离比例衰减公式：
         ``Ps_next = Ps_prev × (R_prev / R_curr) ** n``
       - 同时乘以一次墙面反射能量保持系数 ``wall_reflect_factor``。
       - 其中 ``n = decay_exponent``，经验取 1.3–1.8。
    3. **循环衰减到终点**，最终得到末端峰压。

    参数
    ------
    charge_weight: 爆炸当量（kg TNT）。
    path: 传播节点坐标序列 ``[(x0,y0), (x1,y1), ...]``。
          - `path[0]` 为爆心；
          - `path[1]` 为第一次反射点；
          - 其余为连续反射后的传播节点，按顺序排列。
    decay_exponent: 距离衰减指数 *n*，默认为 1.5。
    wall_reflect_factor: 每次反射的墙面衰减系数，默认 1.0 表示无能量损失。

    返回
    ------
    float: 终点处峰值超压（kPa）。
    """
    # 路径至少包含爆心和一个目标点
    if len(path) < 2:
        raise ValueError("路径必须至少包含两个点（爆心 + 目标）")

    # ── 第 1 段：爆心 → 第 1 个反射点 ─────────────────────────────────────────
    R_prev = math.hypot(path[1][0] - path[0][0], path[1][1] - path[0][1])

    # 查询 Kingery‑Bulmash 表，获取首段反射峰压 Ps0（单位 kPa）
    kb_model = KingeryBulmashModel(neq=charge_weight, distance=R_prev)
    Ps = kb_model.reflected_pressure

    # 若仅有两点，表示无反射段，直接返回入射峰压或反射峰压
    if len(path) == 2:
        return Ps

    # ── 后续段：逐段衰减 ────────────────────────────────────────────────────
    cumulative_dist = R_prev  # 已传播的累计距离

    # 从第二段开始遍历：path[i] → path[i+1]
    for i in range(1, len(path) - 1):
        seg_dist = math.hypot(path[i + 1][0] - path[i][0],
                              path[i + 1][1] - path[i][1])
        if seg_dist <= 0:
            # 跳过重合节点
            continue

        cumulative_next = cumulative_dist + seg_dist  # 最新累计距离 R_curr

        # 距离比例衰减
        ratio = cumulative_dist / cumulative_next  # R_prev / R_curr
        Ps *= ratio ** decay_exponent

        # 墙面能量保持衰减
        Ps *= wall_reflect_factor

        # 更新累计距离，进入下一循环
        cumulative_dist = cumulative_next

    return Ps
def generate_friedlander_pressure_time(
        Ps_kpa: float,
        td_ms: float,
        beta: float,
        ta_ms: float = 0.0,
        dt_ms: float = 0.1,
        clip_to_zero: bool = True,
):
    """生成 Friedlander 压力‑时间历程 P(t)。

    公式
    ------
    P(t) = P_s * (1 - (t - t_a) / t_d) * exp(-β (t - t_a) / t_d),
    其中 t_a ≤ t ≤ t_a + t_d。

    参数
    ------
    Ps_kpa: 峰值超压 kPa
    td_ms: 正相位持续时间 ms
    beta:  衰减系数 (无量纲)
    ta_ms: 到达时间 ms，默认 0（相对时间）
    dt_ms: 采样间隔 ms，默认 0.1
    clip_to_zero: True → 区间外返回 0；False → 不裁剪(仅生成区间内)

    返回
    ------
    t: np.ndarray, 时间轴 (ms)
    P: np.ndarray, 对应压力 (kPa)
    """
    if Ps_kpa <= 0 or td_ms <= 0 or beta <= 0:
        raise ValueError("Ps, td, beta 必须为正数且大于 0")

    import numpy as _np  # 局部引用避免顶层污染

    t = _np.arange(ta_ms, ta_ms + td_ms + dt_ms * 0.5, dt_ms)
    x = (t - ta_ms) / td_ms  # 0 → 1
    P = Ps_kpa * (1 - x) * _np.exp(-beta * x)

    return t, P

# ─────────────────────────────────────────────────────────────────────────────
# 示例运行（仅当作为脚本执行时）
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """示例：生成 50kg TNT、50m 自由场的 Friedlander 波形并绘图"""

    # 1) 基本输入
    charge_weight = 50               # kg TNT
    path = [(0, 0), (50, 0)]         # 爆心 → 50m（自由场）

    # 2) 关键参数
    Ps   = calculate_friedlander_peak_overpressure(charge_weight, path)   # kPa
    td   =  calculate_friedlander_td(charge_weight, 0, 50)              # ms
    beta = calculate_friedlander_beta(charge_weight, path)              # —
    ta   = calculate_friedlander_time_arrive(charge_weight, path)       # ms

    print(f"Ps  = {Ps:.1f} kPa")
    print(f"td  = {td:.1f} ms")
    print(f"β   = {beta:.3f}")
    print(f"ta  = {ta:.1f} ms")

    # 3) Friedlander 波形
    t, P = generate_friedlander_pressure_time(Ps, td, beta, ta_ms=ta, dt_ms=0.2)

    # 4) 绘图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3.5))
    plt.plot(t, P)
    plt.title("Friedlander Pressure History 50 kg TNT, 50 m (free field)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Overpressure (kPa)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

