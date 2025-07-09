import numpy as np
import open3d as o3d

# ----------------------------------------
# 场景参数
# ----------------------------------------
HOUSE_CENTERS   = [(5, 15), (15, -15), (-15, 15), (-15, -15)]
HOUSE_SIZE      = 10.0
HOUSE_HEIGHT    = 20.0
GROUND_SIZE     = 100.0

DETECTOR_X      = 30.0
DETECTOR_HEIGHT = HOUSE_HEIGHT
DETECTOR_DEPTH  = 16.0

MAX_BOUNCES     = 3
ORIGIN_R        = 0.5
EPS             = 1e-6

# 炸点位置（略微抬高到球心）
ORIGIN = np.array([0.0, ORIGIN_R, 0.0], dtype=np.float32)

# 一次性采样的射线数量
NUM_RAYS = 1000

# ----------------------------------------
# 构建场景：返回 scene、探测面ID、legacy_meshes、geometry_data
# ----------------------------------------
def build_scene():
    scene = o3d.t.geometry.RaycastingScene()
    legacy_meshes = []
    geometry_data = {}  # geometry_id -> (vertices, triangles)

    # 地面
    mesh_ground = o3d.geometry.TriangleMesh.create_box(
        width=GROUND_SIZE, height=0.1, depth=GROUND_SIZE)
    mesh_ground.translate((-GROUND_SIZE/2, 0, -GROUND_SIZE/2))
    legacy_meshes.append(mesh_ground)
    vs = np.asarray(mesh_ground.vertices)
    tris = np.asarray(mesh_ground.triangles)
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_ground)
    gid = scene.add_triangles(tmesh)
    geometry_data[gid] = (vs, tris)

    # 房屋
    for (cx, cy) in HOUSE_CENTERS:
        mesh_house = o3d.geometry.TriangleMesh.create_box(
            width=HOUSE_SIZE, height=HOUSE_HEIGHT, depth=HOUSE_SIZE)
        mesh_house.translate((cx - HOUSE_SIZE/2, 0, cy - HOUSE_SIZE/2))
        legacy_meshes.append(mesh_house)
        vs = np.asarray(mesh_house.vertices)
        tris = np.asarray(mesh_house.triangles)
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_house)
        gid = scene.add_triangles(tmesh)
        geometry_data[gid] = (vs, tris)
        mesh_house.paint_uniform_color([0.2, 0.0, 0.0])

    # 探测面
    vs = np.array([
        [DETECTOR_X, 0.0, -DETECTOR_DEPTH/2],
        [DETECTOR_X, 0.0,  DETECTOR_DEPTH/2],
        [DETECTOR_X, DETECTOR_HEIGHT,  DETECTOR_DEPTH/2],
        [DETECTOR_X, DETECTOR_HEIGHT, -DETECTOR_DEPTH/2]
    ], dtype=np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh_det = o3d.geometry.TriangleMesh()
    mesh_det.vertices = o3d.utility.Vector3dVector(vs)
    mesh_det.triangles = o3d.utility.Vector3iVector(tris)
    legacy_meshes.append(mesh_det)
    vp = o3d.core.Tensor(vs, dtype=o3d.core.Dtype.Float32)
    ti = o3d.core.Tensor(tris, dtype=o3d.core.Dtype.UInt32)
    detector_id = scene.add_triangles(vp, ti)
    geometry_data[detector_id] = (vs, tris)
    mesh_det.paint_uniform_color([1.0, 0.0, 0.0])

    # 炸弹球（仅可视化）
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(ORIGIN_R)
    mesh_sphere.translate(ORIGIN)
    mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    legacy_meshes.append(mesh_sphere)

    return scene, detector_id, legacy_meshes, geometry_data

# ----------------------------------------
# 反射、法线计算
# ----------------------------------------
def reflect(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    # 反射定律：$r = d - 2(d\cdot n)n$
    return d - 2 * np.dot(d, n) * n

def get_normal(vs: np.ndarray, tris: np.ndarray, pid: int) -> np.ndarray:
    i0, i1, i2 = tris[pid]
    v0, v1, v2 = vs[i0], vs[i1], vs[i2]
    n = np.cross(v1 - v0, v2 - v0)
    return n / np.linalg.norm(n)

# ----------------------------------------
# 半球均匀采样朝 +X 的方向
# ----------------------------------------
def sample_hemisphere(num: int):
    u = np.random.rand(num).astype(np.float32)      # cos(α) in [0,1]
    theta = (2 * np.pi * np.random.rand(num)).astype(np.float32)
    r = np.sqrt(1 - u**2)
    dirs = np.stack([u, r * np.cos(theta), r * np.sin(theta)], axis=1)
    return dirs  # shape (num,3)

# ----------------------------------------
# 主逻辑：批量追踪，多次反射
# ----------------------------------------
def trace_all_rays(scene, detector_id, geom_data):
    # 初始化
    dirs = sample_hemisphere(NUM_RAYS)
    origins = np.tile(ORIGIN, (NUM_RAYS, 1))
    indices = np.arange(NUM_RAYS)
    paths = {i: [ORIGIN.copy()] for i in indices}
    success = {}

    for bounce in range(MAX_BOUNCES + 1):
        if len(indices) == 0:
            break

        # 批量 cast_rays
        rays_np = np.concatenate([origins, dirs], axis=1)
        rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        t_hit = ans['t_hit'].numpy()
        gids  = ans['geometry_ids'].numpy().astype(int)
        pids  = ans['primitive_ids'].numpy().astype(int)

        hits = origins + dirs * t_hit[:, None]

        # 记录所有打点
        for j, idx in enumerate(indices):
            if not np.isinf(t_hit[j]):
                paths[idx].append(hits[j])

        # 筛选出命中探测面的射线
        hit_det = (gids == detector_id) & (~np.isinf(t_hit))
        for j, idx in enumerate(indices):
            if hit_det[j]:
                success[idx] = paths[idx]

        # 准备下一轮：剔除打空 & 已命中探测面的
        mask = (~hit_det) & (~np.isinf(t_hit))
        new_origins = []
        new_dirs    = []
        new_inds    = []

        for j, keep in enumerate(mask):
            if not keep:
                continue
            gid = gids[j]; pid = pids[j]
            vs, tris = geom_data[gid]
            n = get_normal(vs, tris, pid)
            rd = reflect(dirs[j], n)
            rd /= np.linalg.norm(rd)
            new_origins.append(hits[j] + n * EPS)
            new_dirs.append(rd)
            new_inds.append(indices[j])

        origins = np.array(new_origins, dtype=np.float32)
        dirs    = np.array(new_dirs, dtype=np.float32)
        indices = np.array(new_inds, dtype=int)

    return success

# ----------------------------------------
# 将所有命中路径可视化
# ----------------------------------------
def visualize_paths(meshes, paths_dict):
    points = []
    lines  = []
    colors = []
    idx_cnt = 0
    for idx, path in paths_dict.items():
        # 给每条射线一个随机颜色
        color = np.random.rand(3).tolist()
        for k in range(len(path) - 1):
            p0, p1 = path[k], path[k+1]
            points.append(p0); points.append(p1)
            lines.append([idx_cnt, idx_cnt+1])
            colors.append(color)
            idx_cnt += 2

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines  = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

    o3d.visualization.draw_geometries(meshes + [line_set])

# ----------------------------------------
# 运行示例
# ----------------------------------------
if __name__ == "__main__":
    scene, detector_id, meshes, geom_data = build_scene()
    hits = trace_all_rays(scene, detector_id, geom_data)
    print(f"在最多{MAX_BOUNCES}次反射内，共有{len(hits)}条射线击中探测面")
    visualize_paths(meshes, hits)
