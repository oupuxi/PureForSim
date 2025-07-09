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

# ----------------------------------------
# 场景构建，返回 RaycastingScene 和 探测面几何 ID
# ----------------------------------------
def build_scene():
    scene = o3d.t.geometry.RaycastingScene()

    # 地面
    mesh_ground = o3d.geometry.TriangleMesh.create_box(
        width=GROUND_SIZE, height=0.1, depth=GROUND_SIZE)
    mesh_ground.translate((-GROUND_SIZE/2, 0, -GROUND_SIZE/2))
    tmesh_ground = o3d.t.geometry.TriangleMesh.from_legacy(mesh_ground)
    scene.add_triangles(tmesh_ground)

    # 房屋
    for (cx, cy) in HOUSE_CENTERS:
        mesh_house = o3d.geometry.TriangleMesh.create_box(
            width=HOUSE_SIZE, height=HOUSE_HEIGHT, depth=HOUSE_SIZE)
        mesh_house.translate((cx - HOUSE_SIZE/2, 0, cy - HOUSE_SIZE/2))
        tmesh_house = o3d.t.geometry.TriangleMesh.from_legacy(mesh_house)
        scene.add_triangles(tmesh_house)

    # 探测面：用两三角形的面片
    vs = np.array([
        [DETECTOR_X, 0.0, -DETECTOR_DEPTH/2],
        [DETECTOR_X, 0.0,  DETECTOR_DEPTH/2],
        [DETECTOR_X, DETECTOR_HEIGHT,  DETECTOR_DEPTH/2],
        [DETECTOR_X, DETECTOR_HEIGHT, -DETECTOR_DEPTH/2]
    ], dtype=np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    vp = o3d.core.Tensor(vs)
    ti = o3d.core.Tensor(tris)
    detector_id = scene.add_triangles(vp, ti)

    return scene, int(detector_id)

# ----------------------------------------
# 生成探测面上的初始方向射线
# ----------------------------------------
def sample_detector_rays():
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rays = []
    for y in np.linspace(0, DETECTOR_HEIGHT, 9):
        for z in np.linspace(-DETECTOR_DEPTH/2, DETECTOR_DEPTH/2, 9):
            target = np.array([DETECTOR_X, y, z], dtype=np.float32)
            d = target - origin
            d /= np.linalg.norm(d)
            rays.append((origin.copy(), d.copy()))
    return rays

# ----------------------------------------
# 对一组射线进行一次追踪，返回击中信息
# ----------------------------------------
def trace_once(scene, rays):
    arr = np.zeros((len(rays), 6), dtype=np.float32)
    for i, (o, d) in enumerate(rays):
        arr[i, :3] = o
        arr[i, 3:] = d
    o3d_rays = o3d.core.Tensor(arr)
    ans = scene.cast_rays(o3d_rays)

    ts = ans['t_hit'].cpu().numpy()
    gids = ans['geometry_ids'].cpu().numpy()
    normals = ans['primitive_normals'].cpu().numpy()

    results = []
    for i, (o, d) in enumerate(rays):
        gid = int(gids[i])
        if gid != o3d.t.geometry.RaycastingScene.INVALID_ID:
            t_hit = ts[i]
            p_hit = o + d * t_hit
            results.append({
                'hit': True,
                'id': gid,
                'p': p_hit,
                'n': normals[i]
            })
        else:
            results.append({'hit': False})
    return results

# ----------------------------------------
# 枚举多次反射路径
# ----------------------------------------
def find_paths_with_reflection():
    scene, detector_id = build_scene()
    paths = []
    rays = sample_detector_rays()

    for origin, direction in rays:
        cur_o, cur_d = origin, direction
        path = [origin.copy()]
        for bounce in range(MAX_BOUNCES + 1):
            res = trace_once(scene, [(cur_o, cur_d)])[0]
            if not res['hit']:
                break
            if res['id'] == detector_id:
                path.append(res['p'].copy())
                paths.append(path)
                break
            p_hit = res['p']
            path.append(p_hit.copy())
            n = res['n']
            cur_d = cur_d - 2 * np.dot(cur_d, n) * n
            cur_d /= np.linalg.norm(cur_d)
            cur_o = p_hit + cur_d * EPS
    return paths

# ----------------------------------------
# 可视化
# ----------------------------------------
def visualize(paths):
    vis = []

    mesh_g = o3d.geometry.TriangleMesh.create_box(
        width=GROUND_SIZE, height=0.1, depth=GROUND_SIZE)
    mesh_g.translate((-GROUND_SIZE/2, 0, -GROUND_SIZE/2))
    mesh_g.paint_uniform_color([0.5, 0.8, 0.5])
    vis.append(mesh_g)

    for (cx, cy) in HOUSE_CENTERS:
        mh = o3d.geometry.TriangleMesh.create_box(
            width=HOUSE_SIZE, height=HOUSE_HEIGHT, depth=HOUSE_SIZE)
        mh.translate((cx - HOUSE_SIZE/2, 0, cy - HOUSE_SIZE/2))
        mh.paint_uniform_color([0.7, 0.7, 0.7])
        vis.append(mh)

    vs = np.array([
        [DETECTOR_X, 0.0, -DETECTOR_DEPTH/2],
        [DETECTOR_X, 0.0,  DETECTOR_DEPTH/2],
        [DETECTOR_X, DETECTOR_HEIGHT,  DETECTOR_DEPTH/2],
        [DETECTOR_X, DETECTOR_HEIGHT, -DETECTOR_DEPTH/2]
    ])
    tris = [[0,1,2], [0,2,3]]
    pm = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vs),
        triangles=o3d.utility.Vector3iVector(tris)
    )
    pm.paint_uniform_color([0.5, 0.5, 1.0])
    vis.append(pm)

    sp = o3d.geometry.TriangleMesh.create_sphere(ORIGIN_R)
    sp.translate((0, ORIGIN_R, 0))
    sp.paint_uniform_color([1.0, 0.0, 0.0])
    vis.append(sp)

    for path in paths:
        bounces = len(path) - 2
        color = {0:[1,0,0], 1:[0,0,1], 2:[0,1,0], 3:[1,1,0]}.get(bounces, [0,0,0])
        for i in range(len(path) - 1):
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([path[i], path[i+1]]),
                lines=o3d.utility.Vector2iVector([[0,1]])
            )
            ls.colors = o3d.utility.Vector3dVector([color])
            vis.append(ls)

    o3d.visualization.draw_geometries(vis)

if __name__ == "__main__":
    print("计算射线路径…")
    paths = find_paths_with_reflection()
    print(f"共找到 {len(paths)} 条路径，开始可视化")
    visualize(paths)
