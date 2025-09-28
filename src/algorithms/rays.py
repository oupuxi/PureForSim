# # src/algorithms/rays.py
# from __future__ import annotations
# import numpy as np
# from typing import Tuple
#
# Vec3 = Tuple[float, float, float]
#
# def uniform_hemisphere(n: int, *, up: Vec3 = (0,1,0), method: str = "fibonacci") -> np.ndarray:
#     """生成 n 条均匀分布于 up 方向半球的单位向量 (n,3)。"""
#     up = np.array(up, dtype=np.float64); up /= np.linalg.norm(up)
#     if method == "fibonacci":
#         i = np.arange(n)
#         phi = (np.sqrt(5.0) - 1.0) / 2.0
#         theta = 2 * np.pi * phi * i
#         z = 1 - i / (n - 1 + 1e-12)   # 防止 n=1 分母零
#         r = np.sqrt(np.maximum(0.0, 1 - z**2))
#         dirs = np.stack([r*np.cos(theta), z, r*np.sin(theta)], axis=1)
#     elif method == "random":
#         u = np.random.rand(n)
#         theta = 2*np.pi*np.random.rand(n)
#         z = u
#         r = np.sqrt(np.maximum(0.0, 1 - z**2))
#         dirs = np.stack([r*np.cos(theta), z, r*np.sin(theta)], axis=1)
#     else:
#         raise ValueError(f"Unknown method: {method}")
#
#     if not np.allclose(up, [0,1,0]):
#         y = np.array([0,1,0], dtype=np.float64)
#         v = np.cross(y, up); c = float(np.dot(y, up))
#         if np.allclose(v, 0):
#             dirs = dirs if c > 0 else -dirs
#         else:
#             vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
#             rot = np.eye(3) + vx + vx @ vx * (1/(1+c))
#             dirs = dirs @ rot.T
#     return dirs.astype(np.float64)
