# ray_tracing_simulation.py

import numpy as np
import open3d as o3d
from itertools import permutations

# ----------------------------------------
# 场景参数
# ----------------------------------------
HOUSE_CENTERS   = [(5, 15), (15, -15), (-15, 15), (-15, -15)]
HOUSE_SIZE      = 10.0   # 底面边长
HOUSE_HEIGHT    = 20.0   # 建筑高度（沿 y 轴）
GROUND_SIZE     = 100.0  # 地面尺寸

# 探测面：一块竖直面，法线沿 +x 方向，位于 x=DETECTOR_X
DETECTOR_X      = 30.0         # 探测面所在的 x 坐标
DETECTOR_HEIGHT = HOUSE_HEIGHT # 探测面竖直高度（沿 y 轴）
DETECTOR_DEPTH  = 16.0         # 探测面在 z 方向的跨度（左右各 DEPTH/2）
DETECTOR_THICK  = 0.1          # 平面厚度，仅用于可视化

MAX_BOUNCES     = 3            # 最多反射次数
ORIGIN_R        = 0.5          # 爆炸源小球半径

EPS = 1e-6

# ----------------------------------------
# 平面镜像工具
# ----------------------------------------
def mirror_point(pt, plane):
    """
    将点 pt 绕平面 mirror_plane 镜像。
    plane: (normal, d)，平面方程 n·x + d = 0
    """
    n, d = plane
    dist = (np.dot(n, pt) + d)
    return pt - 2 * dist * n

# ----------------------------------------
# 相交测试：房屋墙面
# ----------------------------------------
def intersect_box_surfaces(ray_o, ray_d):
    """
    计算射线 ray_o + t*ray_d 与所有房屋墙面的最近交点。
    返回：t_min, plane_id, hit_pt
    plane_id 对应 0..len(planes)-1
    """
    t_min = np.inf
    hit_plane = None
    hit_pt = None

    # 准备所有墙面的平面描述：(n, d, range1, range2, range_height)
    planes = []
    for (cx, cy) in HOUSE_CENTERS:
        hx = HOUSE_SIZE / 2
        hy = HOUSE_SIZE / 2
        # x = cx ± hx 面
        for sign in (+1, -1):
            n = np.array([sign, 0.0, 0.0])
            p0 = np.array([cx + sign*hx, 0.0, 0.0])
            d = -np.dot(n, p0)
            # 范围：z∈[-hy,hy] 对应 cy±hy, y∈[0, HOUSE_HEIGHT]
            planes.append((n, d, (cy-hy, cy+hy), (0, HOUSE_HEIGHT)))
        # z = cy ± hy 面
        for sign in (+1, -1):
            # 注意这里 z 对应第三个分量
            n = np.array([0.0, 0.0, sign])
            p0 = np.array([0.0, 0.0, cy + sign*hy])
            d = -np.dot(n, p0)
            # 范围：x∈[cx-hx,cx+hx], y∈[0, HOUSE_HEIGHT]
            planes.append((n, d, (cx-hx, cx+hx), (0, HOUSE_HEIGHT)))

    # 遍历所有平面找最近交点
    for i, (n, d, rg1, rgh) in enumerate(planes):
        denom = np.dot(n, ray_d)
        if abs(denom) < EPS:
            continue
        t = -(np.dot(n, ray_o) + d) / denom
        if t <= EPS or t >= t_min:
            continue
        p = ray_o + t * ray_d
        # 检查高度和水平范围
        if not (rgh[0]-EPS <= p[1] <= rgh[1]+EPS):
            continue
        # rg1 对应 x 或 z 维范围
        # 如果法线在 x 方向，则 rg1 是 z 范围；法线在 z 方向，则 rg1 是 x 范围
        if n[0] != 0:
            if not (rg1[0]-EPS <= p[2] <= rg1[1]+EPS):
                continue
        else:
            if not (rg1[0]-EPS <= p[0] <= rg1[1]+EPS):
                continue
        t_min = t
        hit_plane = i
        hit_pt = p

    return t_min, hit_plane, hit_pt

# ----------------------------------------
# 相交测试：竖直探测面
# ----------------------------------------
def intersect_detector(ray_o, ray_d):
    """
    判断射线是否击中竖直探测面 x=DETECTOR_X。
    返回 t_det, hit_pt 或 (None, None)
    """
    if abs(ray_d[0]) < EPS:
        return None, None
    t = (DETECTOR_X - ray_o[0]) / ray_d[0]
    if t <= EPS:
        return None, None
    p = ray_o + t * ray_d
    # 检查高度 y∈[0, DETECTOR_HEIGHT] 和深度 z∈[-DEPTH/2, DEPTH/2]
    if 0-EPS <= p[1] <= DETECTOR_HEIGHT+EPS and \
            -DETECTOR_DEPTH/2-EPS <= p[2] <= DETECTOR_DEPTH/2+EPS:
        return t, p
    return None, None

# ----------------------------------------
# 枚举 0～3 次反射路径
# ----------------------------------------
def find_paths():
    origin = np.array([0.0, 0.0, 0.0])
    # 房屋所有面索引
    all_surfaces = list(range(len(HOUSE_CENTERS)*2*2))
    # 预构造房屋镜像平面列表 (n, d)
    mirror_planes = []
    for (cx, cy) in HOUSE_CENTERS:
        hx = HOUSE_SIZE/2
        hy = HOUSE_SIZE/2
        # x= 面
        for sign in (+1, -1):
            n = np.array([sign,0,0])
            p0 = np.array([cx + sign*hx, 0, 0])
            mirror_planes.append((n, -np.dot(n,p0)))
        # z= 面
        for sign in (+1, -1):
            n = np.array([0,0,sign])
            p0 = np.array([0,0,cy + sign*hy])
            mirror_planes.append((n, -np.dot(n,p0)))

    paths = []

    # 0 次反射（直射）
    # 在探测面 x=DETECTOR_X 上均匀取一个网格点来试探
    for y in np.linspace(0, DETECTOR_HEIGHT, 9):
        for z in np.linspace(-DETECTOR_DEPTH/2, DETECTOR_DEPTH/2, 9):
            q = np.array([DETECTOR_X, y, z])
            d = q - origin
            d /= np.linalg.norm(d)
            t_det, p_det = intersect_detector(origin, d)
            t_wall, _, _ = intersect_box_surfaces(origin, d)
            # 地面阻挡：q 的 y>=0 且直线段两端 y>=0
            if t_det and t_det < t_wall and origin[1]>=0 and p_det[1]>=0:
                paths.append([origin, p_det])

    # 1~MAX_BOUNCES 次反射
    for b in range(1, MAX_BOUNCES+1):
        for combo in permutations(range(len(mirror_planes)), b):
            # 枚举探测面上的网格点
            for y in np.linspace(0, DETECTOR_HEIGHT, 7):
                for z in np.linspace(-DETECTOR_DEPTH/2, DETECTOR_DEPTH/2, 7):
                    q0 = np.array([DETECTOR_X, y, z])
                    # 先对 q0 依次做镜像
                    q_img = q0.copy()
                    for idx in reversed(combo):
                        q_img = mirror_point(q_img, mirror_planes[idx])
                    # 从原点发射
                    o = origin.copy()
                    d = q_img - o
                    d /= np.linalg.norm(d)
                    segment = [origin]
                    valid = True
                    # 依次检测每次反射
                    for idx in combo:
                        t_wall, pid, hit_pt = intersect_box_surfaces(o, d)
                        if pid != idx or hit_pt is None:
                            valid = False
                            break
                        # 地面阻挡检测
                        if hit_pt[1] < 0:
                            valid = False
                            break
                        segment.append(hit_pt)
                        # 计算镜面反射方向
                        n, _ = mirror_planes[pid]
                        if np.dot(d, n) > 0:  # 确保法向指向入射侧外部
                            n = -n
                        d = d - 2 * np.dot(d, n) * n
                        d /= np.linalg.norm(d)
                        o = hit_pt + d * EPS
                    if not valid:
                        continue
                    # 最后检查是否击中探测面
                    t_det, p_det = intersect_detector(o, d)
                    t_wall, _, _ = intersect_box_surfaces(o, d)
                    if t_det and t_det < t_wall and p_det[1]>=0:
                        segment.append(p_det)
                        paths.append(segment)

    return paths

# ----------------------------------------
# 可视化
# ----------------------------------------
def visualize(paths):
    vis_objs = []

    # 地面
    ground = o3d.geometry.TriangleMesh.create_box(GROUND_SIZE, 0.1, GROUND_SIZE)
    ground.translate((-GROUND_SIZE/2, 0.0, -GROUND_SIZE/2))
    ground.paint_uniform_color([0.5, 0.8, 0.5])
    vis_objs.append(ground)

    # 房屋
    for (cx, cy) in HOUSE_CENTERS:
        house = o3d.geometry.TriangleMesh.create_box(HOUSE_SIZE, HOUSE_HEIGHT, HOUSE_SIZE)
        house.translate((cx-HOUSE_SIZE/2, 0.0, cy-HOUSE_SIZE/2))
        house.paint_uniform_color([0.7, 0.7, 0.7])
        vis_objs.append(house)

    # 探测面（竖直）
    plane = o3d.geometry.TriangleMesh.create_box(
        DETECTOR_THICK,
        DETECTOR_HEIGHT,
        DETECTOR_DEPTH
    )
    plane.translate((DETECTOR_X - DETECTOR_THICK/2, 0.0, -DETECTOR_DEPTH/2))
    plane.paint_uniform_color([0.5, 0.5, 1.0])
    vis_objs.append(plane)

    # 爆炸源
    sphere = o3d.geometry.TriangleMesh.create_sphere(ORIGIN_R)
    sphere.translate((0.0, ORIGIN_R, 0.0))
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    vis_objs.append(sphere)

    # 射线路径
    points = []
    lines  = []
    colors = []
    idx = 0
    cmap = {0:[1,0,0], 1:[0,0,1], 2:[0,1,0], 3:[1,1,0]}
    for path in paths:
        bounces = len(path) - 2
        col = cmap.get(bounces, [0,0,0])
        for i in range(len(path)-1):
            points.append(path[i])
            points.append(path[i+1])
            lines.append([idx, idx+1])
            colors.append(col)
            idx += 2

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(points)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    )
    ls.colors = o3d.utility.Vector3dVector(np.array(colors))
    vis_objs.append(ls)

    o3d.visualization.draw_geometries(vis_objs)

# ----------------------------------------
# 主流程
# ----------------------------------------
if __name__ == "__main__":
    print("计算射线路径…")
    paths = find_paths()
    print(f"共找到 {len(paths)} 条路径，开始可视化")
    visualize(paths)
