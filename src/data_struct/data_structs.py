from __future__ import annotations
from dataclasses import dataclass,field
from typing import List, Tuple, Dict, Optional,Callable
from enum import Enum, auto

import numpy as np
SOUND_C = 343.0          # 默认声速
Vec3 = Tuple[float, float, float]	# (x,y,z)
PathNodeID = int  # 节点编号索引

def _default_kb_factory(W: float, R: float):
    from src.utils.Kingery import KingeryBulmashModel
    return KingeryBulmashModel(neq=W, distance=R)

# ----------------------------------------
# 数据结构
# ----------------------------------------

@dataclass
class Material:
    """表示建筑物的材质属性"""
    name: str                       # 材质名称
    reflection_factor: float = 0.2  # 反射系数(冲击波)0–1
    max_pressure_kpa: float = 1e9   # 破坏阈值
class NodeType(Enum):
    SOURCE = auto()
    WALL   = auto()
    PROBE  = auto()

@dataclass
class BaseNode:
    node_id: int  # 在 RayPath 内唯一，表示光路中第几个节点
    coord: Vec3   # 物理坐标
    node_type: NodeType # 类型标签"source" | "wall" | "probe"
    prev: BaseNode|None =None # 上一个节点
    next: BaseNode|None =None # 下一个节点

@dataclass
class SourceNode(BaseNode):
    node_type: NodeType = NodeType.SOURCE   # 固定
    prev =  None

@dataclass
class WallNode(BaseNode):
    node_type: NodeType = NodeType.WALL

    seg_len: float = 0.0 # 本段长度(m)，首节点可为 0，即到前一个节点的长度
    incident_peak_pressure_kpa: float = 0.0
    reflected_peak_pressure_kpa: float = 0.0
    incident_impulse: float = 0.0
    reflected_impulse: float = 0.0
    arrive_time_ms: float = 0.0
    constant_time_ms: float = 0.0
    friedlander_beta: float = 0.0
    building_id: int = -1 #geom_id
    material_name: str = ""
    reflect_coeff: float = 0.0
    incident_angle_deg: float = 0.0  # 入射角度，单位°，相对于法线的夹角
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # 命中面的法线

@dataclass
class ProbeNode(BaseNode):
    node_type: NodeType = NodeType.PROBE

    seg_len: float = 0.0
    incident_peak_pressure_kpa: float = 0.0
    arrive_time_ms: float = 0.0
    constant_time_ms: float = 0.0
    friedlander_beta: float = 0.0

@dataclass
class RayPath:
    rid: int
    nodes: List[BaseNode] = field(default_factory=list)      # 每个 RayPath 实例都有自己独立的 nodes 列表。
    wall_factors: List[float] = field(default_factory=list)  # len = 反射次数,每次反射记录一次
    @property
    def total_len(self) -> float:
        return sum(n.seg_len for n in self.nodes[1:])
    @property
    def reflect_count(self) -> int:
        return max(0, len(self.nodes) - 2)

    def compute_physics(self, *args, **kwargs):
        """
        占位接口。实际物理递推应由外部高层函数调用。
        """
        raise NotImplementedError("请在物理计算模块（如 physics.workflow）外部调用相关递推函数")


@dataclass
class ProbeGrid:
    origin: Vec3                         # 探测面左下角 world 坐标
    u_vec: Vec3                          # 网格 X 方向 (单位向量，平面内)
    v_vec: Vec3                          # 网格 Y 方向
    du: float                            # 单元尺寸 (u 方向, m)
    dv: float                            # 单元尺寸 (v 方向, m)
    nu: int                              # 列数
    nv: int                              # 行数
    cells: Dict[Tuple[int, int], List[ProbeNode]] = field(default_factory=dict)

    # 将命中点 (world 坐标) 映射到 (i,j) 单元索引
    def coord_to_index(self, p: Vec3) -> Optional[Tuple[int, int]]:
        rel = np.array(p) - np.array(self.origin)
        u = rel.dot(self.u_vec)
        v = rel.dot(self.v_vec)
        i, j = int(u // self.du), int(v // self.dv)
        return (i, j) if 0 <= i < self.nu and 0 <= j < self.nv else None

    # 记录一次命中
    def add_probe_node(self, idx: Tuple[int, int], node: ProbeNode):
        self.cells.setdefault(idx, []).append(node)

    # (i,j) 单元索引得到格子的世界坐标中心点
    def index_to_center(self, i: int, j: int) -> Tuple[float, float, float]:
        """

        """
        if not (0 <= i < self.nu and 0 <= j < self.nv):
            raise IndexError(f"ProbeGrid index out of range: {(i, j)}")
        base = np.asarray(self.origin, dtype=np.float64)
        u = np.asarray(self.u_vec, dtype=np.float64) * (i + 0.5) * self.du
        v = np.asarray(self.v_vec, dtype=np.float64) * (j + 0.5) * self.dv
        return tuple(base + u + v)

# @dataclass
# class BlastScene:
#     """
#     全局容器：
#       • buildings  : geom_id → Building
#       • rays       : rid     → RayPath
#       • probe_grid : 单一探测面；若有多面可改成字典
#     """
#     buildings: Dict[int, Building] = field(default_factory=dict)
#     rays: Dict[int, RayPath] = field(default_factory=dict)
#     probe_grid: Optional[ProbeGrid] = None
#
#     # —— 建筑 / 材质辅助 —— #
#     def add_building(self, b: Building):
#         """在加载 mesh 并获取 geom_id 后，调用此函数注册建筑。"""
#         self.buildings[b.mesh_id] = b     # mesh_id == geom_id 推荐
#
#     def get_material(self, geom_id: int) -> Optional[Material]:
#         b = self.buildings.get(geom_id)   # geom_id 就是 mesh_id
#         return b.material if b else None
#
#     def get_reflect_coeff(self, geom_id: int) -> float:
#         m = self.get_material(geom_id)
#         return m.reflect_coeff if m else 1.0
#
#     # —— 射线辅助 —— #
#     def new_ray(self, rid: int) -> RayPath:
#         """创建一条新的射线并注册到场景."""
#         ray = RayPath(rid=rid)
#         self.rays[rid] = ray
#         return ray
#
#     def add_node_to_ray(self, rid: int, node: BaseNode):
#         """向既有 RayPath 追加节点（假设按顺序调用）"""
#         ray = self.rays[rid]
#         if ray.nodes:
#             ray.nodes[-1].next = node
#             node.prev = ray.nodes[-1]
#         ray.nodes.append(node)
#
#     # —— 探测面辅助 —— #
#     def register_probe_hit(self, probe_node: BaseNode):
#         """如果 probe_grid 存在，则把探测面节点写入对应网格单元"""
#         if self.probe_grid is None:
#             return
#         idx = self.probe_grid.coord_to_index(probe_node.coord)
#         if idx:
#             self.probe_grid.add_probe_node(idx, probe_node)
#
#     # —— 结果查询示例 —— #
#     def get_grid_max_pressure(self, idx) -> float:
#         """
#         返回指定网格单元的最大峰压（kPa）。
#         若未命中则返回 0。
#         """
#         if self.probe_grid is None:
#             return 0.0
#         hits: List[BaseNode] = self.probe_grid.cells.get(idx, [])
#         return max((h.peak_pressure_kpa for h in hits), default=0.0)