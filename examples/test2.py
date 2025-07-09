import open3d as o3d
import numpy as np

# 创建一个简单场景（球体）
scene = o3d.t.geometry.RaycastingScene()
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
sphere_t = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
scene.add_triangles(sphere_t)

# 定义批量射线（从不同位置沿z轴方向发射）
num_rays = 1000
origins = np.random.uniform(-2, 2, size=(num_rays, 3))  # 随机起点
directions = np.tile([0, 0, 1], (num_rays, 1))  # 所有射线沿z轴方向

# 组合射线张量 [起点, 方向]，确保为float32类型
rays = np.hstack([origins, directions]).astype(np.float32)

# 批量执行射线追踪
ans = scene.cast_rays(o3d.core.Tensor(rays))

# 处理结果 - 修改此处以避免布尔类型求和
hit_mask = ans['t_hit'].isfinite()
hit_count = hit_mask.to(dtype=o3d.core.Int32).sum().item()  # 转换为Int32再求和

# 获取命中的射线和交点
hit_indices = hit_mask.numpy()
hit_rays = rays[hit_indices]
distances = ans['t_hit'][hit_mask].numpy()
intersections = hit_rays[:, :3] + hit_rays[:, 3:] * distances[:, np.newaxis]

print(f"追踪 {num_rays} 条射线，{hit_count} 条命中")