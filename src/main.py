from __future__ import annotations
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Tuple, List,Dict,Set
import numpy as np
import open3d as o3d

from src.physics.calculate_blast_parameters import compute_physics_for_raypath
from utils.Kingery import *
from src.data_struct.data_structs import RayPath, ProbeNode, ProbeGrid, Material,SourceNode,NodeType,WallNode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SceneT = o3d.t.geometry.RaycastingScene
PathNodeID = int  # 节点编号索引
RayDirArray = np.ndarray
Vec3 = Tuple[float, float, float]	#(x,y,z)





# ----------------------------------------
# 场景参数
# ----------------------------------------
HOUSE_CENTERS   = [(5, 15), (15, -15), (-15, 15), (-15, -15)] # 房屋中心点坐标列表，每个元组表示一个房屋在X-Z平面上的位置。
HOUSE_SIZE      = 10.0 # 房屋的边长
HOUSE_HEIGHT    = 20.0 # 建筑物在Y轴方向的高度。
GROUND_SIZE     = 100.0 # 地面尺寸

DETECTOR_X      = 30.0 # 探测器在X轴上的位置
DETECTOR_HEIGHT = 50 # 探测器所在高度
DETECTOR_DEPTH  = 16.0  # 探测器的深度

MAX_BOUNCES     = 3  # 光线最大反射次数
ORIGIN_R        = 0.05 # “爆源球壳”半径,用 uniform_hemisphere() 之类函数时，常把射线起点推到球壳上，以避开自相交.origin = BLAST_CENTER + dir * ORIGIN_R，大于EPS ×10³，远小于最小房屋尺寸；一般占场景尺度
EPS             = 1e-6 # 极小值常量，用于浮点数比较或防止除零错误等数值稳定性处理
# 准备你的物理材质库
MAT = {
    "concrete": Material("concrete", reflection_factor=0.85, max_pressure_kpa=8000),
    # ……
}

# 创建探测面的函数，以 O 点为原点，u_vec/v_vec 为边，长宽为 nu*du, nv*dv
def create_probe_plane(origin, u_vec, v_vec, nu, nv, du, dv):
    # 构造四个角点
    p0 = np.array(origin)
    p1 = p0 + np.array(u_vec) * du * nu
    p2 = p0 + np.array(u_vec) * du * nu + np.array(v_vec) * dv * nv
    p3 = p0 + np.array(v_vec) * dv * nv
    # 两个三角面
    vertices = np.stack([p0, p1, p2, p3])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color([0.9, 0.9, 0.9])  # 明显可见
    return mesh
def add_probe_plane_to_scene(scene, mesh):
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    geom_id = scene.add_triangles(tmesh)
    return geom_id

def create_ground_plane(size: float,
                        y: float = 0.0) -> o3d.geometry.TriangleMesh:
    """
    生成位于 y 平面的正方形地板，由两片三角面组成。
    """
    half = size * 0.5
    vertices = np.array([
        [-half, y, -half],  # 0
        [ half, y, -half],  # 1
        [ half, y,  half],  # 2
        [-half, y,  half],  # 3
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2],
                          [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

# ----------------------------------------
# 场景构建，返回 RaycastingScene 和 探测面几何 ID
# ----------------------------------------
def build_scene():
    scene = o3d.t.geometry.RaycastingScene()
    vis_meshes = []  # ← 存 legacy 网格，供可视化
    material_map: dict[int, Material] = {}
    probe_geom_ids:set[int] = set()

    # 地面
    mesh_ground = o3d.geometry.TriangleMesh.create_box(
        width=GROUND_SIZE, height=0.1, depth=GROUND_SIZE)
    mesh_ground.translate((-GROUND_SIZE/2, -0.1, -GROUND_SIZE/2))
    tmesh_ground = o3d.t.geometry.TriangleMesh.from_legacy(mesh_ground)
    scene.add_triangles(tmesh_ground)
    vis_meshes.append(mesh_ground.paint_uniform_color([0.7, 0.7, 0.7]))

    # 房屋
    for (cx, cy) in HOUSE_CENTERS:
        mat=MAT["concrete"]
        mesh_house = o3d.geometry.TriangleMesh.create_box(
            width=HOUSE_SIZE, height=HOUSE_HEIGHT, depth=HOUSE_SIZE)
        mesh_house.translate((cx - HOUSE_SIZE/2, 0, cy - HOUSE_SIZE/2))
        tmesh_house = o3d.t.geometry.TriangleMesh.from_legacy(mesh_house)
        vis_meshes.append(mesh_house.paint_uniform_color([0.3, 0.6, 0.8]))
        geom_id =scene.add_triangles(tmesh_house)
        material_map[geom_id] = mat

    # 探测面
    probe_grid = ProbeGrid(
        origin = (30.0, 0.0, -24.0),  # z 从 -24 到 +24，总宽 48m
        u_vec   = (0.0, 0.0, 1.0),
        v_vec   = (0.0, 1.0, 0.0),
        du      = 6.0,   # 48m / 8 = 6m
        dv      = 5.0,   # 40m / 8 = 5m
        nu      = 9,     # 列数不变
        nv      = 9,     # 行数不变
    )
    probe_mesh = create_probe_plane(
        origin=probe_grid.origin,
        u_vec=probe_grid.u_vec,
        v_vec=probe_grid.v_vec,
        nu=probe_grid.nu,
        nv=probe_grid.nv,
        du=probe_grid.du,
        dv=probe_grid.dv)
    geom_id = add_probe_plane_to_scene(scene, probe_mesh)
    vis_meshes.append(probe_mesh)
    probe_geom_ids.add(geom_id)

    return scene,vis_meshes,material_map,probe_geom_ids,probe_grid



def uniform_hemisphere(n: int, *, up: Vec3 = (0,1,0), method: str = "fibonacci") -> RayDirArray:
    """
    只是生成方向向量（单位向量数组，表示“哪边是上半球”），不涉及“从哪里发射”或者“起点在哪里”。
    生成 n 条均匀分布于 up 方向半球的单位向量（形状 (n,3)）。
    支持 method=["fibonacci", "random"]。
    """
    up = np.array(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    if method == "fibonacci":
        # Fibonacci 螺旋半球采样
        # 公式见 Ray Tracing Gems I, Chapter 5
        # https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/Sampling_Uniformly_on_Spheres_and_Hemispheres
        i = np.arange(n)
        phi = (np.sqrt(5.0) - 1.0) / 2.0   # 黄金比例
        theta = 2 * np.pi * phi * i
        z = 1 - i / (n - 1)     # 从 1 到 0
        r = np.sqrt(1 - z**2)
        # 半球限制
        dirs = np.stack([r * np.cos(theta), z, r * np.sin(theta)], axis=1)  # up=(0,1,0)半球

    elif method == "random":
        # 随机半球采样
        u = np.random.rand(n)
        theta = 2 * np.pi * np.random.rand(n)
        z = u   # [0,1]
        r = np.sqrt(1 - z**2)
        dirs = np.stack([r * np.cos(theta), z, r * np.sin(theta)], axis=1)  # up=(0,1,0)

    else:
        raise ValueError(f"Unknown method: {method}")

    # 若 up 不是 y+，则需将 [0,1,0]→up 旋转
    def align_vector(vecs, target):
        # 计算把[0,1,0]转到target的旋转矩阵
        y = np.array([0,1,0], dtype=np.float64)
        v = np.cross(y, target)
        c = np.dot(y, target)
        if np.allclose(v, 0):
            if c > 0: return vecs
            else: return -vecs
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * (1/(1+c))
        return vecs @ rot.T

    if not np.allclose(up, [0,1,0]):
        dirs = align_vector(dirs, up)

    return dirs.astype(np.float64)  # 形状(n,3)

def trace_single_ray(
        scene: SceneT,
        origin: Vec3,
        dir: Vec3,
        *,
        material_map: Dict[int, Material],
        probe_geom_ids: Set[int],
        max_bounces: int = 3,
        max_dist: float = 500.0,
        origin_offset: float = 0.05,# 必须 >EPS；<墙厚/重要最小距离。
        eps: float = 1e-6
) -> RayPath:
    """
    追踪一条射线，遇到障碍物生成WallNode，遇到探测面生成ProbeNode，返回完整RayPath。
    """
    # 初始化起点和方向
    p0 = np.array(origin, dtype=np.float64) + np.array(dir, dtype=np.float64) * origin_offset # 射线起点，不是炸点位置
    dir = np.array(dir, dtype=np.float64)
    total_len = 0.0 # 累计行进距离
    bounce_id = 0 # 已反射次数计数
    path = RayPath(rid=-1) # 新建一个 RayPath 对象，rid 暂用 -1（可后续设置真正 ID）

    # 先放入 SourceNode
    src_node = SourceNode(node_id=0, coord=tuple(p0), node_type=NodeType.SOURCE)
    path.nodes.append(src_node)
    prev_node = src_node # 以便后续生成节点时可以 prev=prev_node 进行链式连接

    while bounce_id <= max_bounces:
        # 构造Open3D t型射线：o3d.core.Tensor (B,6) 或 (6,)
        ray_np = np.hstack([p0, dir]) # 拼接出 [origin + direction] 的 6 维数组[ox,oy,oz, dx,dy,dz]
        ray_tensor = o3d.core.Tensor(ray_np, dtype=o3d.core.Dtype.Float32).reshape((1,6)) # 转为 Open3D Tensor，并 reshape((1,6))，表示一条射线。
        # Cast
        out = scene.cast_rays(ray_tensor) # 求出最近交点的位置、命中哪个几何体，输出结构化字段（t_hit, geometry_ids, normal 等）
        # hit = ans[0]   # 因为单射线，所以第0个结果
        # distance = float(hit['t_hit'].item()) # 沿 dir 方向的距离
        # geom_id = int(hit['geometry_ids'].item()) # 命中的 mesh_id
        distance  = float(out['t_hit'][0].item())
        geom_id   = int(out['geometry_ids'][0].item())

        if distance == float('inf') or geom_id < 0:
            # 离场，终止
            break

        # 计算本段长度和累计距离
        seg_len = distance
        total_len += seg_len
        if total_len >= max_dist:
            # 超距离，终止（可在这里插值终点）
            break

        # 计算命中点、法线
        hit_point = p0 + dir * seg_len # 命中点坐标
        normal = out['primitive_normals'][0].numpy() # 因为只有一条射线，故从 hit['primitive_normals'] 中拿到面片法向数组，取第一个

        node_id = len(path.nodes) # 用目前 path.nodes 长度作为新节点的 node_id，保证唯一递增

        if geom_id in probe_geom_ids:
            # 命中探测面，生成 ProbeNode
            probe_node = ProbeNode(
                node_id = node_id,
                coord = tuple(hit_point),
                node_type = NodeType.PROBE,
                prev = prev_node,
                seg_len = seg_len
            )
            path.nodes.append(probe_node)
            prev_node.next = probe_node #
            prev_node = probe_node
            # 重要：射线继续沿原方向穿透探测面，不反射
            # 注意：1、若是墙面->探测面->墙面，找这个代码来，应该是有问题的，因为击中探测面，不影响射线传递，故一些参数是按照探测面的前一个结点来计算的
            step_n = np.sign(np.dot(dir, normal)) * normal * 1e-3   # 1 mm
            step_t = dir * 1e-4
            p0 = hit_point + step_n + step_t
            # 不加反射次数，继续直行
            continue
        else:
            # 计算入射角和单位法向量
            dir_in = -dir / np.linalg.norm(dir)
            normal_unit = normal / (np.linalg.norm(normal) + 1e-12)
            cos_theta = np.dot(dir_in, normal_unit)
            incident_angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            # 注意：这里的geom_id就能确保在构建的 material_map 中吗，会不会有可能是地面
            # 命中障碍物（墙面）
            material = material_map.get(geom_id, None)
            wall_node = WallNode(
                node_id = node_id,
                coord = tuple(hit_point),
                node_type = NodeType.WALL,
                prev = prev_node,
                seg_len = seg_len,
                building_id = geom_id,
                material_name = material.name if material else "",
                reflect_coeff = material.reflection_factor if material else 1.0,
                incident_angle_deg=incident_angle_deg,  # 新增
                normal=tuple(normal_unit)  # 新增
            )
            path.nodes.append(wall_node)
            prev_node.next = wall_node
            prev_node = wall_node
            path.wall_factors.append(material.reflection_factor if material else 1.0)

            # 反射：新方向 = dir - 2*(dir·n)*n，单位化
            dir = dir - 2 * np.dot(dir, normal) * normal
            dir /= np.linalg.norm(dir)

            step_n = np.sign(np.dot(dir, normal)) * normal * 1e-3   # 1 mm
            step_t = dir * 1e-4
            p0 = hit_point + step_n + step_t

            bounce_id += 1

    return path

def raypath_to_lineset(raypath: RayPath, color=(1, 0, 0)):
    """
    将 RayPath 渲染为 Open3D LineSet，color 为 RGB 三元组（0-1），默认红色。
    """
    points = [np.array(n.coord) for n in raypath.nodes]
    if len(points) < 2:
        raise ValueError("RayPath 节点数不足，无法成线")
    # 连续相邻点连线
    lines = [[i, i+1] for i in range(len(points)-1)]
    colors = [color] * len(lines)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def register_probe_hits_for_path(path: RayPath, grid: ProbeGrid) -> int:
    """
    遍历 RayPath，找到所有 ProbeNode，落格登记到 grid.cells。
    返回登记条数（命中次数）。
    """
    count = 0
    for node in path.nodes:
        if isinstance(node, ProbeNode):
            idx = grid.coord_to_index(node.coord)
            if idx is not None:
                grid.add_probe_node(idx, node)
                count += 1
    return count

def make_probe_grid_lineset(grid: ProbeGrid) -> o3d.geometry.LineSet:
    """生成探测面的网格线 LineSet（便于可视化对齐格子）"""
    O = np.asarray(grid.origin, float)
    U = np.asarray(grid.u_vec, float)
    V = np.asarray(grid.v_vec, float)
    # 顶点：按行列生成网格交点
    verts = []
    for j in range(grid.nv + 1):
        for i in range(grid.nu + 1):
            p = O + U * (i * grid.du) + V * (j * grid.dv)
            verts.append(p)
    verts = np.asarray(verts, float)

    # 连线：横线 + 竖线
    lines = []
    # 横线 (沿 u 方向)
    for j in range(grid.nv + 1):
        row_start = j * (grid.nu + 1)
        for i in range(grid.nu):
            a = row_start + i
            b = row_start + i + 1
            lines.append([a, b])
    # 竖线 (沿 v 方向)
    for i in range(grid.nu + 1):
        for j in range(grid.nv):
            a = j * (grid.nu + 1) + i
            b = (j + 1) * (grid.nu + 1) + i
            lines.append([a, b])

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(verts),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, int))
    )
    # 网格线颜色（淡青）
    ls.colors = o3d.utility.Vector3dVector([[0.2, 0.8, 0.8]] * len(lines))
    return ls

def make_probe_hits_pointcloud(grid: ProbeGrid,
                               color_by: str = "incident_peak_pressure_kpa",
                               vmax: float | None = None) -> o3d.geometry.PointCloud:
    """
    将所有 Probe 命中点渲染为点云；按指定字段映射颜色（默认按入射峰压）。
    颜色映射：红(高)→蓝(低)
    """
    pts, cols = [], []
    # 收集所有命中
    for hits in grid.cells.values():
        for node in hits:
            pts.append(np.asarray(node.coord, float))
            val = float(getattr(node, color_by, 0.0))
            cols.append(val)
    if not pts:
        return o3d.geometry.PointCloud()

    pts = np.asarray(pts, float)
    cols = np.asarray(cols, float)
    if vmax is None:
        vmax = max(cols.max(), 1e-6)
    x = np.clip(cols / vmax, 0.0, 1.0)
    # 简单红蓝配色：x=1→红, x=0→蓝
    colors = np.stack([x, 0.0 * x, 1.0 - x], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

import csv

def export_probe_hits_csv(grid: ProbeGrid, path: str) -> None:
    """
    把所有 Probe 命中点逐条导出（行=一次命中）。
    列包含：i,j, x,y,z, Ps_inc, Ps_ref, I_inc, I_ref, ta, td, beta
    """
    fields = [
        "i","j","x","y","z",
        "incident_peak_pressure_kpa",
        "reflected_peak_pressure_kpa",
        "incident_impulse",
        "reflected_impulse",
        "arrive_time_ms",
        "constant_time_ms",
        "friedlander_beta",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for (i,j), hits in grid.cells.items():
            for n in hits:
                x,y,z = n.coord
                row = [
                    i, j, x, y, z,
                    getattr(n, "incident_peak_pressure_kpa", 0.0),
                    getattr(n, "reflected_peak_pressure_kpa", 0.0),
                    getattr(n, "incident_impulse", 0.0),
                    getattr(n, "reflected_impulse", 0.0),
                    getattr(n, "arrive_time_ms", 0.0),
                    getattr(n, "constant_time_ms", 0.0),
                    getattr(n, "friedlander_beta", 0.0),
                ]
                w.writerow(row)

def export_probe_cell_stats_csv(grid: ProbeGrid, path: str) -> None:
    """
    对每个格子做简单聚合：命中次数、Pmax、最早到达时间。
    """
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i","j","hits","Pmax_inc_kPa","t_first_ms"])
        for (i,j), hits in grid.cells.items():
            if not hits:
                continue
            pmax = max(getattr(h, "incident_peak_pressure_kpa", 0.0) for h in hits)
            tmin = min(getattr(h, "arrive_time_ms", float("inf")) for h in hits)
            w.writerow([i, j, len(hits), pmax, tmin])

if __name__ == "__main__":
    scene, vis_meshes, material_map, probe_geom_ids, probe_grid = build_scene()

    origin = (0.0, 0.01, 0.0)

    # ───────────────────────────────────────
    # 7 条测试方向：
    # 3 条打到同一格 (i=4, j=4)；4 条打到左右下上的邻格
    # ───────────────────────────────────────
    d_same_1 = np.array([0.797, 0.598,  0.0797])   # 命中 (4,4)
    d_same_2 = np.array([0.815, 0.570,  0.1085])   # 命中 (4,4)
    d_same_3 = np.array([0.774, 0.632,  0.0258])   # 命中 (4,4)

    d_left   = np.array([0.797, 0.598, -0.0797])   # 命中 (3,4)
    d_right  = np.array([0.778, 0.583,  0.2330])   # 命中 (5,4)
    d_down   = np.array([0.861, 0.502,  0.0860])   # 命中 (4,3)
    d_up     = np.array([0.735, 0.674,  0.0736])   # 命中 (4,5)

    test_dirs = [d_same_1, d_same_2, d_same_3, d_left, d_right, d_down, d_up]

    # 逐条发射、计算物理量、登记到 ProbeGrid
    for k, d in enumerate(test_dirs, 1):
        print(f"\n=== 发射射线 #{k} ===")
        path = trace_single_ray(
            scene=scene,
            origin=origin,
            dir=d,                         # 直接使用上面的单位方向
            material_map=material_map,
            probe_geom_ids=probe_geom_ids,
            max_bounces=3
        )
        # 计算该条射线上的 Ps/td/β/ta/冲量等
        compute_physics_for_raypath(path, 500)

        # 登记到探测网格
        hit_count = register_probe_hits_for_path(path, probe_grid)
        print(f"登记命中次数: {hit_count}")

        # 打印该条射线的 Probe 命中，以及各命中格子的 (i,j)
        for n in path.nodes:
            if isinstance(n, ProbeNode):
                idx = probe_grid.coord_to_index(n.coord)
                print(f"  Probe 命中坐标 {n.coord} → cell{idx}  "
                      f"Ps={getattr(n,'incident_peak_pressure_kpa',0.0):.1f} kPa, "
                      f"ta={getattr(n,'arrive_time_ms',0.0):.2f} ms")

        # 可选：把单条路径的折线也加到可视化里
        vis_meshes.append(raypath_to_lineset(path, color=(1, 0, 0)))

    # 汇总打印每个被命中的格子的统计
    print("\n【探测网格命中汇总】")
    for (i, j), hits in probe_grid.cells.items():
        pmax = max(getattr(h, "incident_peak_pressure_kpa", 0.0) for h in hits)
        tmin = min(getattr(h, "arrive_time_ms", 1e9) for h in hits)
        print(f"  cell({i},{j}): hits={len(hits)}  Pmax={pmax:.1f} kPa  t_first={tmin:.2f} ms")

    # 可视化：网格线 + 命中点云（按入射峰压着色）
    vis_meshes.append(make_probe_grid_lineset(probe_grid))
    pcd_hits = make_probe_hits_pointcloud(probe_grid, color_by="incident_peak_pressure_kpa")
    if len(pcd_hits.points) > 0:
        vis_meshes.append(pcd_hits)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=(0,0,0))
    vis_meshes.append(axes)
    o3d.visualization.draw_geometries(vis_meshes)

    # 导出 CSV
    export_probe_hits_csv(probe_grid, "probe_hits_points.csv")
    export_probe_cell_stats_csv(probe_grid, "probe_hits_cells.csv")
    print("\n已导出: probe_hits_points.csv / probe_hits_cells.csv")

# if __name__ == "__main__":
#     scene, vis_meshes,material_map,probe_geom_ids,probe_grid= build_scene()
#     #----发射射线、追踪每条射线生成射线数个RayPath
#
#     ray_dir = np.array([0.9, 0.8, 1.5])
#     ray_dir = ray_dir / np.linalg.norm(ray_dir)
#     path = trace_single_ray(
#         scene=scene,
#         origin=(0.0, 0.01, 0.0),
#         dir=ray_dir,
#         material_map=material_map,
#         probe_geom_ids=probe_geom_ids,
#         max_bounces=3
#     )
#     print("节点数:", len(path.nodes))
#     for i, n in enumerate(path.nodes):
#         print(f"节点{i} ({n.node_type.name}): 坐标 = {n.coord}")
#     # 由路径进行冲击波量的计算
#     compute_physics_for_raypath(path,500)
#
#     print("\n【所有节点参数快照】")
#     for i, n in enumerate(path.nodes):
#         print(f"\n节点{i} ({n.node_type.name}):")
#         for field in n.__dataclass_fields__:
#             value = getattr(n, field)
#             print(f"  {field}: {value}")
#
#     # 把 Probe 命中登记进网格
#     num_hits = register_probe_hits_for_path(path, probe_grid)
#     print(f"\n登记到探测网格的命中次数: {num_hits}")
#     # 5) 打印每个被命中的格子
#     for (i, j), hits in probe_grid.cells.items():
#         print(f"cell({i},{j}): hits={len(hits)}  "
#               f"Pmax={max(getattr(h, 'incident_peak_pressure_kpa', 0.0) for h in hits):.1f} kPa  "
#               f"t_first={min(getattr(h, 'arrive_time_ms', 1e9) for h in hits):.2f} ms")
#     # 由结点的参数，进行面的渲染
#
#     # 渲染成 Open3D LineSet（红色）
#     lineset = raypath_to_lineset(path, color=(1, 0, 0))
#     vis_meshes.append(lineset)
#
#     vis_meshes.append(make_probe_grid_lineset(probe_grid))
#     pcd_hits = make_probe_hits_pointcloud(probe_grid, color_by="incident_peak_pressure_kpa")
#     if len(pcd_hits.points) > 0:
#         vis_meshes.append(pcd_hits)
#     # 添加入场景的坐标系（长 10 米）
#     axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=(0.0,0.0,0.0))# 红色 X 轴，蓝色 Z 轴，绿色 Y 轴
#     vis_meshes.append(axes)
#
#
#     o3d.visualization.draw_geometries(vis_meshes)
#     export_probe_hits_csv(probe_grid, "probe_hits_points.csv")
#     export_probe_cell_stats_csv(probe_grid, "probe_hits_cells.csv")
#     print("\n已导出: probe_hits_points.csv / probe_hits_cells.csv")
#     pass

"""
dirs       = uniform_hemisphere(N_RAYS)
ray_paths  = trace_rays_batch(scene, ORIGIN, dirs,
                              material_map=material_map,
                              probe_geom_ids=probe_geom_ids)
for p in ray_paths:
    p.compute_physics(charge_w=10.0, sound_c=343.0, kb_model=KB)

probe_grid = ProbeGrid(...)
aggregate_probe_grid(ray_paths, probe_grid)
probe_grid.save_csv("probe_stats.csv")
"""

