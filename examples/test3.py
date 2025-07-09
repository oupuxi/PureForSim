"""
Detector pressure animation with Kingery‑Bulmash attenuation
-----------------------------------------------------------

* Metric units throughout (m, kg, ms, kPa)
* TNT equivalent mass W = 100 kg
* Uses KingeryBulmashModel (utils/Kingery.py) to obtain peak reflected
  pressure P_s and positive phase impulse I_d at distance R1 (direct
  segment length from explosion origin to first hit / first reflection).

Workflow
========
1. Build the 3‑D scene (ground, four houses, detector wall, TNT sphere)
   exactly as in previous script.
2. Uniformly sample NUM_RAYS directions inside the half‑sphere that
   faces the detector plane ( +X direction ) and perform batched
   ray‑casting with Open3D t geometry. Each reflected segment is handled
   in further batched calls (MAX_BOUNCES).
3. Every ray that ultimately reaches the detector plane returns a list
   of segments ``[(o0,d0,L0),(o1,d1,L1),…,hit]``.
4. For each successful ray:
   * n = number of reflections (len(segments)‑2)
   * R1 = L0 (origin → first hit)
   * R_cum = cumulative length up to current (final) reflection point
     ( inclusive )
   * Cr = 0.6 → attenuation factor Cr^n
   * Peak reflected pressure after distance & reflection attenuation:
       P_s_att = P_s * Cr**n * (R1 / R_cum)**2
   * Positive phase impulse after attenuation:
       I_d_att = I_d * Cr**n * (R1 / R_cum)
   * Arrival time:
       t_a = (L0 / c) * 1e3 \
           + (sum(L_i for i>=1) / (1.1*c)) * 1e3   # ms
   * Duration:
       t_d = 2 * I_d_att / P_s_att                   # ms
   * Pressure–time history for this detector point:
       P_D(t) = P_s_att * (1‑(t‑t_a)/t_d) * exp(‑β*(t‑t_a)/t_d),
       with β = 0.7, defined only for t_a ≤ t ≤ t_a+t_d.
5. Collect all detector points and their individual (t_a, t_d,
   P_s_att) triples.
6. Animation – iterate 100 frames from global min(t_a) to
   max(t_a+t_d). For each frame:
   * Evaluate pressure for every point; if outside its pulse window,
     pressure = 0.
   * Normalize pressure to [0,1] by dividing by global P_s_att_max.
   * Map scalar → colour (blue –> red) and update point colours.
   * Refresh an Open3D Visualizer in off‑screen or on‑screen mode and
     save ``frames/frame_###.png``.

The script only depends on standard numpy, open3d >= 0.18, and the
kingery‑bulmash PyPI package.
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import open3d as o3d

from src.utils.Kingery import KingeryBulmashModel  # adjust import to your layout

# ---------------------------
# Constants & global params
# ---------------------------
HOUSE_CENTERS = [(5, 15), (15, -15), (-15, 15), (-15, -15)]
HOUSE_SIZE = 10.0
HOUSE_HEIGHT = 20.0
GROUND_SIZE = 100.0

DETECTOR_X = 30.0
DETECTOR_HEIGHT = HOUSE_HEIGHT
DETECTOR_DEPTH = 16.0

MAX_BOUNCES = 3
ORIGIN_R = 0.5
EPS = 1e-6

# explosion origin (raised ORIGIN_R to be at ground level)
ORIGIN = np.array([0.0, ORIGIN_R, 0.0], dtype=np.float32)

# simulation parameters
NUM_RAYS = 1000
C_SOUND = 343.0  # m/s at 20 °C
CR = 0.6  # wall reflection attenuation coefficient
BETA = 0.7
TNT_MASS = 100.0  # kg, metric

FRAME_COUNT = 100
OUTPUT_DIR = Path("frames")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------
# Scene construction helpers
# ---------------------------

def build_scene():
    scene = o3d.t.geometry.RaycastingScene()
    legacy_meshes: List[o3d.geometry.Geometry] = []
    geom_data: Dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # Ground
    mesh_ground = o3d.geometry.TriangleMesh.create_box(
        width=GROUND_SIZE, height=0.1, depth=GROUND_SIZE)
    mesh_ground.translate((-GROUND_SIZE/2, 0, -GROUND_SIZE/2))
    legacy_meshes.append(mesh_ground)
    gid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_ground))
    geom_data[gid] = (np.asarray(mesh_ground.vertices), np.asarray(mesh_ground.triangles))

    # Houses
    for (cx, cy) in HOUSE_CENTERS:
        mesh_house = o3d.geometry.TriangleMesh.create_box(
            width=HOUSE_SIZE, height=HOUSE_HEIGHT, depth=HOUSE_SIZE)
        mesh_house.translate((cx - HOUSE_SIZE/2, 0, cy - HOUSE_SIZE/2))
        legacy_meshes.append(mesh_house)
        gid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_house))
        geom_data[gid] = (np.asarray(mesh_house.vertices), np.asarray(mesh_house.triangles))

    # Detector rectangle
    vs = np.array([
        [DETECTOR_X, 0.0, -DETECTOR_DEPTH/2],
        [DETECTOR_X, 0.0,  DETECTOR_DEPTH/2],
        [DETECTOR_X, DETECTOR_HEIGHT,  DETECTOR_DEPTH/2],
        [DETECTOR_X, DETECTOR_HEIGHT, -DETECTOR_DEPTH/2],
    ], dtype=np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh_det = o3d.geometry.TriangleMesh()
    mesh_det.vertices = o3d.utility.Vector3dVector(vs)
    mesh_det.triangles = o3d.utility.Vector3iVector(tris)
    legacy_meshes.append(mesh_det)
    detector_id = scene.add_triangles(
        o3d.core.Tensor(vs, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(tris, dtype=o3d.core.Dtype.UInt32))
    geom_data[detector_id] = (vs, tris)

    # TNT sphere visual
    sphere = o3d.geometry.TriangleMesh.create_sphere(ORIGIN_R)
    sphere.translate(ORIGIN)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    legacy_meshes.append(sphere)

    return scene, detector_id, legacy_meshes, geom_data

# ---------------------------
# Geometry helpers
# ---------------------------

def reflect(dir_: np.ndarray, n: np.ndarray) -> np.ndarray:
    return dir_ - 2.0 * np.dot(dir_, n) * n

def tri_normal(vs: np.ndarray, tris: np.ndarray, pid: int) -> np.ndarray:
    i0, i1, i2 = tris[pid]
    n = np.cross(vs[i1] - vs[i0], vs[i2] - vs[i0])
    return n / np.linalg.norm(n)

# ---------------------------
# Ray sampling & tracing
# ---------------------------

def sample_half_sphere(num: int):
    u = np.random.rand(num).astype(np.float32)
    theta = (2.0 * np.pi * np.random.rand(num)).astype(np.float32)
    r = np.sqrt(1.0 - u**2)
    # +X facing hemisphere
    return np.stack([u, r * np.cos(theta), r * np.sin(theta)], axis=1)


def trace_rays(scene, detector_id, geom_data):
    dirs = sample_half_sphere(NUM_RAYS)
    origins = np.tile(ORIGIN, (NUM_RAYS, 1))
    active_idx = np.arange(NUM_RAYS)

    paths: Dict[int, List[dict]] = {i: [] for i in active_idx}
    hits: Dict[int, np.ndarray] = {}

    for bounce in range(MAX_BOUNCES + 1):
        if len(active_idx) == 0:
            break
        rays_np = np.hstack([origins, dirs])
        ans = scene.cast_rays(o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32))
        t_hit = ans["t_hit"].numpy()
        gids = ans["geometry_ids"].numpy().astype(int)
        pids = ans["primitive_ids"].numpy().astype(int)

        hit_pts = origins + dirs * t_hit[:, None]

        mask_next = []
        new_origins = []
        new_dirs = []
        new_idx = []

        for j, idx in enumerate(active_idx):
            if np.isinf(t_hit[j]):
                continue  # miss ⇒ discard

            gid = gids[j]
            pid = pids[j]
            L = float(t_hit[j])
            n = tri_normal(*geom_data[gid], pid) if gid != detector_id else np.array([1.0, 0.0, 0.0])
            paths[idx].append({
                "length": L,
                "normal": n,
                "origin": origins[j].copy(),
                "dir": dirs[j].copy(),
            })

            if gid == detector_id:
                hits[idx] = hit_pts[j]
                continue  # reached detector, done

            # prepare reflection
            refl_dir = reflect(dirs[j], n)
            refl_dir /= np.linalg.norm(refl_dir)
            new_origins.append(hit_pts[j] + n * EPS)
            new_dirs.append(refl_dir)
            new_idx.append(idx)

        origins = np.array(new_origins, dtype=np.float32)
        dirs = np.array(new_dirs, dtype=np.float32)
        active_idx = np.array(new_idx, dtype=int)

    return paths, hits

# ---------------------------
# Pressure computations
# ---------------------------

def compute_pressure_curves(paths: Dict[int, List[dict]],
                            hits: Dict[int, np.ndarray]):
    records = []
    p_max_global = 0.0

    for idx, segs in paths.items():
        if idx not in hits:
            continue  # this ray never reached detector
        n_reflect = len(segs) - 1  # n reflections = segments before last detector hit, minus 1 (origin→first bounce)

        # Extract lengths
        L0 = segs[0]["length"]
        L_rest = sum(s["length"] for s in segs[1:])
        R_cum = L0 + L_rest

        # Kingery‑Bulmash at distance R1
        kb_model = KingeryBulmashModel(neq=TNT_MASS, distance=L0, unit_system="metric",safe=False)
        P_s = kb_model.reflected_pressure  # kPa
        I_d = kb_model.reflected_impulse   # kPa·ms

        # Attenuation
        P_s_att = P_s * (CR ** n_reflect) * (L0 / R_cum) ** 2
        I_d_att = I_d * (CR ** n_reflect) * (L0 / R_cum)

        # Arrival & duration in ms
        t_a = (L0 / C_SOUND + L_rest / (1.1 * C_SOUND)) * 1e3
        t_d = 2.0 * I_d_att / P_s_att

        p_max_global = max(p_max_global, P_s_att)

        records.append({
            "pos": hits[idx],
            "P_s": P_s_att,
            "I_d": I_d_att,
            "t_a": t_a,
            "t_d": t_d,
        })
    return records, p_max_global

# ---------------------------
# Animation
# ---------------------------

def colour_map(val: float) -> List[float]:
    """Linear blue→red colormap for scalar [0,1]."""
    return [val, 0.0, 1.0 - val]


def run_animation(meshes, records, p_max):
    if p_max == 0 or len(records) == 0:
        print("No hits to visualise.")
        return

    # Prepare point cloud
    pts = np.array([r["pos"] for r in records])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pts))

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720, visible=True)
    for m in meshes:
        vis.add_geometry(m)
    vis.add_geometry(pcd)

    # Time span
    t0 = min(r["t_a"] for r in records)
    t1 = max(r["t_a"] + r["t_d"] for r in records)

    for frame, t in enumerate(np.linspace(t0, t1, FRAME_COUNT)):
        colours = []
        for r in records:
            if r["t_a"] <= t <= r["t_a"] + r["t_d"]:
                tau = (t - r["t_a"]) / r["t_d"]
                P = r["P_s"] * (1 - tau) * np.exp(-BETA * tau)
            else:
                P = 0.0
            colours.append(colour_map(P / p_max))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colours))
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(OUTPUT_DIR / f"frame_{frame:03d}.png"), do_render=True)
        print(f"Rendered frame {frame+1}/{FRAME_COUNT}")

    vis.destroy_window()

# ---------------------------
# Entry point
# ---------------------------

def main():
    scene, detector_id, meshes, geom_data = build_scene()
    paths, hits = trace_rays(scene, detector_id, geom_data)
    records, p_max = compute_pressure_curves(paths, hits)
    print(f"Detected {len(records)} hit points, max pressure = {p_max:.2f} kPa")
    run_animation(meshes, records, p_max)


if __name__ == "__main__":
    main()
