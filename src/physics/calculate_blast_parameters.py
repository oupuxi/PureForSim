"""
calculate_blast_parameters.py

该模块用于计算爆炸波形相关的物理参数，例如 Friedlander 波形中的时间常数、衰减系数，
冲击波正相位持续时间、峰值超压等。
"""
import math
import warnings
from typing import List, Tuple
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar

from src.utils.Kingery import KingeryBulmashModel
from src.data_struct.data_structs import RayPath, NodeType, WallNode, ProbeNode, SOUND_C, BaseNode


def _calc_Cr(cos_theta: float) -> float:
    """近似正入射反射系数"""
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if theta_deg < 60.0:
        return 2.0 + cos_theta ** 2 + 3.0 * cos_theta
    return 1.0
def _cos_theta(prev_pt: np.ndarray, curr_pt: np.ndarray, normal: np.ndarray) -> float:
    dir_in = (curr_pt - prev_pt)
    dir_in /= (np.linalg.norm(dir_in) + 1e-12)
    normal /= (np.linalg.norm(normal) + 1e-12)
    return abs(dir_in @ normal)


# ---------- 动态寻找括区的辅助函数 ----------
def _find_bracket(f, x_lo=0.05, x_hi=1.0,
                  factor=2.0, max_iter=20):
    """
    逐次把 x_hi *= factor，直到 f(x_lo)*f(x_hi)<0。
    若始终同号，返回 None。
    """
    f_lo, f_hi = f(x_lo), f(x_hi)
    for _ in range(max_iter):
        if f_lo * f_hi < 0:          # 已异号
            return (x_lo, x_hi)
        x_hi *= factor               # 向右扩张
        f_hi = f(x_hi)
    return None                      # bracket 未找到

# β
def calculate_friedlander_beta(raypath,
                               charge_weight: float,
                               decay_factor: float = 0.85,
                               beta_min: float = 0.25,
                               beta_fallback: float = 0.30) :
    """
递推计算 RayPath 路径各节点的 Friedlander 衰减系数 β，并返回终点节点的 β。

算法说明：
    - 首段（爆心→首命中点，节点0→1）：
        采用 Kingery-Bulmash 表，根据爆炸当量和距离数值反求获得自由场 β₀。
    - 其后每次遇到障碍物（即每过一个节点）：
        β 按固定衰减因子 decay_factor 递减（即 β = β₀ × decay_factor^n）。
        最小不低于 beta_min。

参数:
    raypath      : RayPath 路径对象，nodes 为节点链表，节点.coord 为三维坐标
    charge_weight: 炸药当量（kg TNT），用于 KB 查表
    decay_factor : β 每经过一次反射的递减系数，默认 0.85
    beta_min     : β 最小允许值，默认 0.25

过程:
    - 自动将各节点的 friedlander_beta 字段填入递推值（节点0、1为 β₀，其后依次衰减）

返回:
    float        : 路径最后一个节点的 Friedlander β 系数

示例:
    beta_end = calculate_friedlander_beta(path, charge_weight=10)

"""
    nodes = raypath.nodes
    if len(nodes) < 2:
        raise ValueError("路径节点数不足 2")
    p0, p1 = map(np.asarray, (nodes[0].coord, nodes[1].coord))
    R_free = np.linalg.norm(p1 - p0)

    kb = KingeryBulmashModel(neq=charge_weight, distance=R_free)
    Ps_kpa = kb.incident_pressure
    td_ms = kb.positive_phase_duration
    I_kpams = kb.incident_impulse

    # ---------- 2. 构造残差函数 ----------
    def residual(beta: float) -> float:
        if beta <= 0:
            return np.inf
        I_fried = (Ps_kpa * td_ms / beta) * (1 - np.exp(-beta))
        return I_fried - I_kpams

    # ---------- 3. 动态括区 + 求根 ----------
    bracket = _find_bracket(residual, x_lo=0.05, x_hi=1.0)

    if bracket is None:  # 始终同号 ⇒ 用后备值
        warnings.warn("β root 未括住，使用 fallback", RuntimeWarning)
        beta0 = beta_fallback
    else:
        sol = root_scalar(residual, bracket=bracket, method="brentq")
        if not sol.converged:
            warnings.warn("β 反求未收敛，使用 fallback", RuntimeWarning)
            beta0 = beta_fallback
        else:
            beta0 = sol.root

    # ---------- 4. 写 β 到各节点 ----------
    nodes[1].friedlander_beta = beta0  # 自由场节点
    for i in range(2, len(nodes)):
        beta_i = max(beta0 * decay_factor ** (i - 1), beta_min)
        nodes[i].friedlander_beta = beta_i

    return nodes[-1].friedlander_beta
# td
def calculate_friedlander_td(raypath: RayPath,
                             charge_weight: float,
                             td_max: float = 200.0) :
    """
    给 RayPath 所有节点递推填充正相位持续时间 td (ms) 到节点字段 constant_time_ms。
    - 节点1（自由场终点）用 KB 查表；
    - 后续节点用前一段的 td 和距离，递推。
    """
    nodes = raypath.nodes
    if len(nodes) < 2:
        raise ValueError("路径节点数不足2")
    # 节点0为爆心，无意义（可设为0或None）
    nodes[0].constant_time_ms = 0.0

    # 节点1（自由场终点）
    p0 = np.array(nodes[0].coord)
    p1 = np.array(nodes[1].coord)
    dist1 = np.linalg.norm(p1 - p0)
    model = KingeryBulmashModel(neq=charge_weight, distance=dist1)
    td1 = model.positive_phase_duration
    nodes[1].constant_time_ms = td1

    # 后续节点递推
    for i in range(2, len(nodes)):
        pa = np.array(nodes[i - 1].coord)
        pb = np.array(nodes[i].coord)
        prev_td = nodes[i - 1].constant_time_ms
        prev_distance = np.linalg.norm(pa - np.array(nodes[i - 2].coord))
        distance = np.linalg.norm(pb - pa)
        # 递推公式（与你的 calculate_friedlander_td 保持一致）
        ratio = distance / prev_distance if prev_distance > 0 else 1.0
        td = prev_td * math.log1p(ratio)
        td = min(td, td_max)
        nodes[i].constant_time_ms = td



# 到达时间ta
def calculate_friedlander_time_arrive(
        raypath: RayPath,
        charge_weight: float,
        sound_speed: float = 343.0,
        speed_scale: float = 1.1
) :
    """
    递推计算 RayPath 路径中每个节点的冲击波到达时间（ms），并自动填入各节点的 arrive_time_ms 字段。

    算法说明:
        - 首段（爆心 → 第1个命中点）：查 Kingery-Bulmash 表获取自由场传播时延（ms）。
        - 其余段（多次反射/障碍物/探测面）：按速度 v = sound_speed × speed_scale（默认1.1倍声速）递推累加，delta_t = 段长 / v。
        - 到达时间按节点顺序递推，所有节点的 arrive_time_ms 字段被依次写入。

    参数:
        raypath      : RayPath 对象，包含节点链表 nodes（每个节点需有 coord 字段，三维坐标）
        charge_weight: 炸药当量（kg TNT），用于 KB 表查自由场传播时间
        sound_speed  : 空气中声速（m/s），默认 343
        speed_scale  : 障碍物段的超声速比例系数（默认 1.1）

    过程:
        - 自动将各节点的 arrive_time_ms 字段填入递推值（节点0为0，节点1查KB，节点2及以后用公式递推）

    返回:
        无返回值（到达时间直接填入每个节点的 arrive_time_ms 字段）

    用法示例:
        calculate_friedlander_time_arrive(path, charge_weight=10)
        for n in path.nodes:
            print(n.arrive_time_ms)

    """

    nodes = raypath.nodes
    if len(nodes) < 2:
        raise ValueError("路径节点数不足2")
    # 节点0（爆源）设为0
    nodes[0].arrive_time_ms = 0.0
    # 节点1（第一个命中点）：查 KB 得到 t0
    p0 = np.array(nodes[0].coord)
    p1 = np.array(nodes[1].coord)
    dist0 = np.linalg.norm(p1 - p0)
    try:
        model = KingeryBulmashModel(neq=charge_weight, distance=dist0)
        t0_ms = model.time_of_arrival
    except Exception as e:
        raise RuntimeError(f"KB表查找失败：{e}")
    nodes[1].arrive_time_ms = t0_ms

    # 其余节点递推
    v = sound_speed * speed_scale
    for i in range(2, len(nodes)):
        pa = np.array(nodes[i - 1].coord)
        pb = np.array(nodes[i].coord)
        segment_dist = np.linalg.norm(pb - pa)
        delta_t_ms = (segment_dist / v) * 1000
        nodes[i].arrive_time_ms = nodes[i - 1].arrive_time_ms + delta_t_ms
# Ps
def calculate_friedlander_peak_overpressure(
        raypath: RayPath,
        charge_weight: float,
        *,
        decay_exponent: float = 1.1,
        alpha: float = 0.0                            # 可选线性吸收系数
) -> float:
    """
    为 RayPath 中的每个 Wall / Probe 节点写入
        • incident_peak_pressure_kpa
        • reflected_peak_pressure_kpa   (Probe 节点置 0)
    并返回终点(Probe 或最后墙)的 incident_peak_pressure_kpa。

    规则
    ----
    - 第一面：直接查 KB 表入射峰压 → 依 Cr 求反射峰压
    - 后续段：以上一面“反射峰压”为起点，按 (R_prev/R_now)^n·e^{-αΔR} 衰减到入射峰压
              再乘 Cr 得新的反射峰压
    """
    """
        为 RayPath 写 incident / reflected 峰压 (kPa)，返回终点 incident 值。
        - 传播使用上一面 *入射* 峰压
        - 墙面反射仅影响当前节点，不影响后续
        """
    nodes = raypath.nodes
    if len(nodes) < 2:
        raise ValueError("RayPath 至少需 1 个 WALL/PROBE 节点")

    # ---------- 第一面 ----------
    A, B = np.asarray(nodes[0].coord), np.asarray(nodes[1].coord)
    R_prev = np.linalg.norm(B - A)
    kb = KingeryBulmashModel(neq=charge_weight, distance=R_prev)
    P_inc = float(kb.incident_pressure)

    # 第一面反射
    normal1 = getattr(nodes[1], "normal", None)
    if normal1 is not None:
        cos_th = _cos_theta(A, B, np.asarray(normal1))
    else:
        cos_th = 1.0
    C_r = _calc_Cr(cos_th)
    nodes[1].incident_peak_pressure_kpa = P_inc
    nodes[1].reflected_peak_pressure_kpa = P_inc * C_r

    # ---------- 后续面逐段传播 ----------
    for idx in range(1, len(nodes) - 1):
        prev_pt = np.asarray(nodes[idx].coord)
        curr_pt = np.asarray(nodes[idx + 1].coord)
        seg = np.linalg.norm(curr_pt - prev_pt)
        if seg <= 0:
            continue
        R_now = R_prev + seg
        # 只用入射峰压递推
        P_inc *= (R_prev / R_now) ** decay_exponent * np.exp(-alpha * seg)

        node = nodes[idx + 1]
        node.incident_peak_pressure_kpa = P_inc

        if node.node_type is NodeType.WALL:
            nrm = getattr(node, "normal", None)
            if nrm is not None:
                cos_th = _cos_theta(prev_pt, curr_pt, np.asarray(nrm))
            else:
                cos_th = 1.0
            C_r = _calc_Cr(cos_th)
            node.reflected_peak_pressure_kpa = P_inc * C_r
        else:  # Probe
            node.reflected_peak_pressure_kpa = 0.0

        R_prev = R_now

    return nodes[-1].incident_peak_pressure_kpa

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

def compute_physics_for_raypath(raypath, charge_weight, sound_speed=343.0):
    calculate_friedlander_beta(raypath, charge_weight)
    calculate_friedlander_td(raypath, charge_weight)
    calculate_friedlander_peak_overpressure(raypath, charge_weight)
    calculate_friedlander_impulse(raypath)
    calculate_friedlander_time_arrive(raypath, charge_weight, sound_speed=sound_speed)


def friedlander_impulse(Ps: float, td: float, beta: float) -> float:
    """Friedlander 波形的正相位冲量（kPa·ms）"""
    if Ps <= 0 or td <= 0 or beta <= 0:
        return 0.0
    return Ps * td / beta * (1 - np.exp(-beta) * (1 + beta))

def calculate_friedlander_impulse(raypath: RayPath) -> None:
    """
    自动为每个节点写入 incident_impulse、reflected_impulse (kPa·ms)。
    依赖节点已写入 incident_peak_pressure_kpa, constant_time_ms, friedlander_beta。
    """
    nodes = raypath.nodes
    for node in nodes:
        # 入射冲量
        Ps = getattr(node, "incident_peak_pressure_kpa", 0.0)
        td = getattr(node, "constant_time_ms", 0.0)
        beta = getattr(node, "friedlander_beta", 0.0)
        node.incident_impulse = friedlander_impulse(Ps, td, beta)
        # 反射冲量
        if node.node_type is NodeType.WALL:
            Ps_ref = getattr(node, "reflected_peak_pressure_kpa", 0.0)
            node.reflected_impulse = friedlander_impulse(Ps_ref, td, beta)
        else:
            node.reflected_impulse = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 示例运行（仅当作为脚本执行时）
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def make_simple_raypath(points):
        """从点序列生成一个简单的 RayPath（无物理反射，只直线）"""
        raypath = RayPath(rid=0)
        prev_node = None
        for i, p in enumerate(points):
            if i == 0:
                node = BaseNode(node_id=0, coord=tuple(p), node_type=NodeType.SOURCE)
            else:
                node = BaseNode(node_id=i, coord=tuple(p), node_type=NodeType.WALL)
            node.prev = prev_node
            if prev_node:
                prev_node.next = node
            raypath.nodes.append(node)
            prev_node = node
        return raypath
    """示例：生成 50kg TNT、50m 自由场的 Friedlander 波形并绘图（RayPath风格）"""
    # 1) 构造简单 RayPath（爆心→50m）
    charge_weight = 50
    points = [(0, 0, 0), (50, 0, 0)]  # 爆心和目标点
    raypath = make_simple_raypath(points)

    # 2) 各节点递推物理参数
    calculate_friedlander_beta(raypath, charge_weight)
    calculate_friedlander_td(raypath, charge_weight)
    calculate_friedlander_time_arrive(raypath, charge_weight)
    calculate_friedlander_peak_overpressure(raypath, charge_weight)

    # 3) 取末端节点（目标点）各物理参数
    node = raypath.nodes[-1]
    Ps = node.incident_peak_pressure_kpa
    td = node.constant_time_ms
    beta = node.friedlander_beta
    ta = node.arrive_time_ms

    print(f"Ps  = {Ps:.1f} kPa")
    print(f"td  = {td:.1f} ms")
    print(f"β   = {beta:.3f}")
    print(f"ta  = {ta:.1f} ms")

    # 4) Friedlander 波形
    t, P = generate_friedlander_pressure_time(Ps, td, beta, ta_ms=ta, dt_ms=0.2)

    # 5) 绘图
    plt.figure(figsize=(6, 3.5))
    plt.plot(t, P)
    plt.title("Friedlander Pressure History 50 kg TNT, 50 m (free field)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Overpressure (kPa)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

