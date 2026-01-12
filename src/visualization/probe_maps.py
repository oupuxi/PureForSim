"""
把 probe_hits_points.csv / probe_hits_cells.csv 解析为二维格网图层，并提供导出与绘图工具。
- 自动从 CSV 头部注释(# key: value)解析网格元数据：origin/u_vec/v_vec/du/dv/nu/nv 等
- 生成四类静态图：峰压(max)、到达时间(min)、冲量(sum)、命中次数(count)
- 生成任意时刻快照：Friedlander 解析式 P(t) 并按格子合成(sum/max)

它不负责追踪/物理计算（那些在 environment/ 和 physics/ 中）；

它不负责读写 RayPath（IO 可在 utils/ 里），但应提供从 CSV/DataFrame 直接生成图层的函数；

它的产出是：二维数组 + 元数据（可保存为 PNG/CSV/NPZ）以及可选的动画
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Iterable
import io, csv, re, ast, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import numpy as np

# ---------- 数据结构 ----------
Vec3 = Tuple[float, float, float]


@dataclass
class GridMeta:
    origin: Vec3
    u_vec: Vec3
    v_vec: Vec3
    du: float
    dv: float
    nu: int
    nv: int
    # 允许额外上下文元数据（如 plane_name/height/charge_weight 等）
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class Field2D:
    data: np.ndarray  # 形状 (nv, nu)，行=j(沿v)，列=i(沿u)
    meta: GridMeta
    name: str = ""
    units: str = ""


# ---------- 读 CSV + 解析头部元数据 ----------
_KV_RE = re.compile(r"^\s*#\s*([^:=#]+)\s*[:=]\s*(.+?)\s*$")


def _literal(s: str):
    """安全解析标量/元组，如 '(30,0,-24)' -> tuple；'9'->int；'3.5'->float；其他原样返回"""
    try:
        val = ast.literal_eval(s)
        return val
    except Exception:
        return s


def parse_header_meta(path: str) -> Dict[str, object]:
    """
    从 CSV 头部以 # 开头的注释行提取 'key: value' 对。
    返回 dict；不做强类型转换（交由 build_meta 负责）。
    """
    meta: Dict[str, object] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ls = line.strip()
            if not ls.startswith("#"):
                break
            m = _KV_RE.match(ls)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                meta[key] = _literal(val)
    return meta


def build_meta(raw: Dict[str, object]) -> GridMeta:
    """
    接受两种头部格式：
    1) 扁平键值：origin/u_vec/v_vec/du/dv/nu/nv
    2) JSON 一行：meta={ ..., "grid": { origin/u_vec/v_vec/du/dv/nu/nv }, ... }
    """
    # 如果有 'meta' 且是 dict，就从里面取 'grid' 作为主源
    if "meta" in raw and isinstance(raw["meta"], dict):
        meta_block = raw["meta"]
        grid_block = meta_block.get("grid", {})
        # 把 grid 的键提升到顶层（供下面统一读取）
        merged = {**raw, **grid_block}
        # 其余额外信息放到 extra
        extra = {k: v for k, v in meta_block.items() if k != "grid"}
    else:
        merged = dict(raw)
        extra = {}

    # 兼容大小写/别名的取值工具
    def g(key: str, *aliases, default=None):
        keys = (key, *aliases)
        for k in keys:
            for kk in (k, k.upper(), k.lower()):
                if kk in merged:
                    return merged[kk]
        if default is not None:
            return default
        raise KeyError(f"CSV 头部缺少元数据字段: {keys}")

    # 读取必需字段（允许为 list/tuple，转 float/int）
    origin = tuple(float(x) for x in g("origin"))
    u_vec  = tuple(float(x) for x in g("u_vec", "uvec"))
    v_vec  = tuple(float(x) for x in g("v_vec", "vvec"))
    du     = float(g("du"))
    dv     = float(g("dv"))
    nu     = int(g("nu", "cols"))
    nv     = int(g("nv", "rows"))

    # extra 合并其余顶层键，去掉已消费的
    consumed = {"origin","u_vec","uvec","v_vec","vvec","du","dv","nu","nv","cols","rows","meta","grid"}
    for k in list(merged.keys()):
        if k not in consumed:
            extra[k] = merged[k]

    return GridMeta(origin=origin, u_vec=u_vec, v_vec=v_vec,
                    du=du, dv=dv, nu=nu, nv=nv, extra=extra)



def _read_csv_rows_skip_comments(path: str) -> List[Dict[str, str]]:
    """跳过以 # 开头的注释行，返回 csv.DictReader 的行列表（字符串字典）"""
    with open(path, "r", encoding="utf-8") as f:
        data_lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    if not data_lines:
        return []
    reader = csv.DictReader(io.StringIO("".join(data_lines)))
    return list(reader)


# ---------- 工具：把 (i,j)->聚合值 的字典落到二维场 ----------

def _empty_field(meta: GridMeta, fill: float = np.nan) -> np.ndarray:
    arr = np.empty((meta.nv, meta.nu), dtype=float)
    arr[:] = fill
    return arr


def _assign_cell(arr: np.ndarray, i: int, j: int, val: float):
    """按 (i,j) 落到 arr[j, i]；越界则忽略"""
    if 0 <= i < arr.shape[1] and 0 <= j < arr.shape[0]:
        arr[j, i] = val

# ---------- 从 points CSV 生成四类静态图 ----------
def read_points_csv(path: str) -> Tuple[List[Dict[str, object]], GridMeta]:
    """
    读一个点级命中 CSV 文件；
    把文件头部的网格信息转成 GridMeta；
    把数据区的每一行转成 Python dict，字段全部强制为 int/float，方便数值计算；
    返回 (数据记录列表, 网格元数据)。
    """
    raw_meta = parse_header_meta(path)
    meta = build_meta(raw_meta)
    rows = _read_csv_rows_skip_comments(path)

    # 强类型转换
    recs: List[Dict[str, object]] = []
    for r in rows:
        rr = {
            "i": int(r["i"]),
            "j": int(r["j"]),
            "x": float(r.get("x", 0.0)),
            "y": float(r.get("y", 0.0)),
            "z": float(r.get("z", 0.0)),
            "incident_peak_pressure_kpa": float(r.get("incident_peak_pressure_kpa", 0.0)),
            "reflected_peak_pressure_kpa": float(r.get("reflected_peak_pressure_kpa", 0.0)),
            "incident_impulse": float(r.get("incident_impulse", 0.0)),
            "reflected_impulse": float(r.get("reflected_impulse", 0.0)),
            "arrive_time_ms": float(r.get("arrive_time_ms", 0.0)),
            "constant_time_ms": float(r.get("constant_time_ms", 0.0)),
            "friedlander_beta": float(r.get("friedlander_beta", 0.0)),
        }
        # 可选：光线 id / 权重
        if "rid" in r:
            rr["rid"] = int(r["rid"])
        if "ray_weight" in r:
            rr["ray_weight"] = float(r["ray_weight"])
        recs.append(rr)
    return recs, meta

def peak_map_from_points(points_csv: str, which: str = "incident") -> Field2D:
    """
    输入：点级命中 CSV。
    处理：对每个网格单元 (i,j)，取所有命中点的入射/反射峰值压力的最大值。
    输出：一张 Field2D，即峰压分布二维场，可以直接用来画图、导出 PNG/CSV。
    """
    rows, meta = read_points_csv(points_csv)
    key = "incident_peak_pressure_kpa" if which == "incident" else "reflected_peak_pressure_kpa" # 根据 which 参数，确定要取入射还是反射峰压
    best: Dict[Tuple[int,int], float] = {}
    for r in rows:
        ij = (r["i"], r["j"])
        v = r.get(key, 0.0)
        if np.isnan(v):
            continue
        best[ij] = v if ij not in best else max(best[ij], v)
    arr = _empty_field(meta)
    for (i, j), v in best.items():
        _assign_cell(arr, i, j, v)
    name = f"peak_{which}"
    units = "kPa"
    return Field2D(arr, meta, name, units)

def arrival_map_from_points(points_csv: str) -> Field2D:
    rows, meta = read_points_csv(points_csv)
    best: Dict[Tuple[int,int], float] = {}
    for r in rows:
        ij = (r["i"], r["j"])
        v = r.get("arrive_time_ms", np.nan)
        if np.isnan(v):
            continue
        best[ij] = v if ij not in best else min(best[ij], v)  # 每格取最小到达时间
    arr = _empty_field(meta)
    for (i, j), v in best.items():
        _assign_cell(arr, i, j, v)
    return Field2D(arr, meta, "arrival_min", "ms")

def impulse_map_from_points(points_csv: str, which: str = "incident",
                            weight_col: Optional[str] = None) -> Field2D:
    rows, meta = read_points_csv(points_csv)
    key = "incident_impulse" if which == "incident" else "reflected_impulse"
    sums: Dict[Tuple[int,int], float] = {}
    for r in rows:
        ij = (r["i"], r["j"])
        val = r.get(key, 0.0)
        if weight_col and (weight_col in r):
            val *= float(r[weight_col])
        sums[ij] = val if ij not in sums else sums[ij] + val
    arr = _empty_field(meta)
    for (i, j), v in sums.items():
        _assign_cell(arr, i, j, v)
    return Field2D(arr, meta, f"impulse_sum_{which}", "kPa·ms")

def hitcount_map_from_points(points_csv: str) -> Field2D:
    rows, meta = read_points_csv(points_csv)
    cnt: Dict[Tuple[int,int], int] = {}
    for r in rows:
        ij = (r["i"], r["j"])
        cnt[ij] = 1 if ij not in cnt else cnt[ij] + 1
    arr = _empty_field(meta, fill=0.0)
    for (i, j), v in cnt.items():
        _assign_cell(arr, i, j, float(v))
    return Field2D(arr, meta, "hit_count", "count")

# ---------- 任意时刻快照（基于 Friedlander） ----------

def _friedlander_P_at_t(Ps: float, td: float, beta: float, ta: float, t: float) -> float:
    """单次命中在 t_ms 时刻的压力值（kPa）"""
    if td <= 0 or beta <= 0:
        return 0.0
    d = t - ta
    if d < 0 or d > td:
        return 0.0
    x = d / td
    return Ps * (1.0 - x) * math.exp(-beta * x)

def time_range_from_points(points_csv: str) -> Tuple[float, float]:
    rows, _ = read_points_csv(points_csv)
    if not rows:
        return 0.0, 0.0
    t0 = min(r["arrive_time_ms"] for r in rows)
    t1 = max(r["arrive_time_ms"] + r["constant_time_ms"] for r in rows)
    return t0, t1

def snapshot_map(points_csv: str, t_ms: float,
                 which: str = "incident", combine: str = "sum") -> Field2D:
    """
    计算时刻 t_ms 的瞬时压力图。
    combine: 'sum'（默认，线性叠加）或 'max'（包络）
    which:   'incident' / 'reflected'
    """
    rows, meta = read_points_csv(points_csv)
    use_Ps = "incident_peak_pressure_kpa" if which == "incident" else "reflected_peak_pressure_kpa"

    # 每格聚合
    vals: Dict[Tuple[int,int], float] = {}
    for r in rows:
        P = _friedlander_P_at_t(
            Ps  = r.get(use_Ps, 0.0),
            td  = r.get("constant_time_ms", 0.0),
            beta= r.get("friedlander_beta", 0.0),
            ta  = r.get("arrive_time_ms", 0.0),
            t   = t_ms
        )
        if P <= 0:
            continue
        ij = (r["i"], r["j"])
        if combine == "max":
            vals[ij] = P if ij not in vals else max(vals[ij], P)
        else:  # sum
            vals[ij] = P if ij not in vals else vals[ij] + P

    arr = _empty_field(meta)
    for (i, j), v in vals.items():
        _assign_cell(arr, i, j, v)
    return Field2D(arr, meta, f"snapshot_{which}_t{t_ms:.1f}ms", "kPa")


def _building_mask_from_meta(meta) -> Optional[np.ndarray]:
    """
    兼容 export.py 的写法：house_centers/house_size 在 meta['context'] 里
    （见 export.py: meta["context"]=run_meta）
    """
    # meta.extra 里通常就是 meta json 的顶层 dict
    top = getattr(meta, "extra", {}) or {}

    # 关键：先从 context 里找（export.py 把 run_meta 塞进这里）
    ctx = top.get("context", {}) or {}

    def pick(key: str):
        # 先从 context 拿，再从顶层拿（两者都兼容）
        if key in ctx: return ctx[key]
        if key in top: return top[key]
        # 再做大小写容错
        for k, v in ctx.items():
            if str(k).lower() == key.lower():
                return v
        for k, v in top.items():
            if str(k).lower() == key.lower():
                return v
        return None

    centers = pick("house_centers")
    size    = pick("house_size")
    if centers is None or size is None:
        return None

    # CSV 读出来可能是字符串，转成 python 对象
    if isinstance(centers, str):
        centers = ast.literal_eval(centers)
    if isinstance(size, str):
        size = float(ast.literal_eval(size))

    try:
        centers = list(centers)
        size = float(size)
    except Exception:
        return None

    ox, _, oz = meta.origin
    du, dv = meta.du, meta.dv
    nu, nv = meta.nu, meta.nv

    uu = (np.arange(nu) + 0.5) * du
    vv = (np.arange(nv) + 0.5) * dv
    U, V = np.meshgrid(uu, vv)

    mask = np.zeros((nv, nu), dtype=bool)
    half = 0.5 * size

    for (cx, cz) in centers:
        u0 = (cx - half) - ox
        u1 = (cx + half) - ox
        v0 = (cz - half) - oz
        v1 = (cz + half) - oz
        mask |= (U >= u0) & (U <= u1) & (V >= v0) & (V <= v1)

    return mask


# ---------- 可视化和导出 ----------
# --- add these imports near the top of probe_maps.py ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from typing import Optional

# 如果你文件里已经有 GridMeta / Field2D 的定义或 import，就不需要重复
# from .your_structs import GridMeta, Field2D




def plot_field(
    field,
    *,
    title: str | None = None,
    save_path: str | None = None,
    cbar_label: str | None = None,
    show_buildings: bool = True,
    vmax_clip: float = 300.0,                    # 你要 >300 变白
    zero_color_rgba=(1.0, 0.97, 0.80, 1.0),      # 你要 0 是淡黄
    nodata_to_zero: bool = True,                 # 非建筑 NaN -> 0
):
    """
    规则（按你的要求）：
    1) 建筑（house mask）= 白色（不画黑边）
    2) 超压 > vmax_clip = 白色
    3) 非建筑且无数据/无命中（NaN）= 当作 0，上色为淡黄（不是白/灰）
    4) 色带使用 magma_r，且 0 的颜色强制为淡黄
    """
    meta = field.meta
    extent = [0, meta.nu * meta.du, 0, meta.nv * meta.dv]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.set_facecolor("white")

    # --- colormap: magma_r + 0=淡黄 + over=白 ---
    base = plt.get_cmap("magma_r")
    colors = base(np.linspace(0, 1, 256))
    colors[0] = np.array(zero_color_rgba)  # 0 -> 淡黄

    cmap_obj = ListedColormap(colors)
    cmap_obj.set_over("white")             # >vmax_clip -> 白
    # set_bad 仍可设置，但我们会把非建筑 NaN -> 0，所以 bad 基本不会出现
    cmap_obj.set_bad(colors[0])            # 万一还有 NaN，也按淡黄处理

    norm = Normalize(vmin=0.0, vmax=float(vmax_clip), clip=False)

    # --- data prep ---
    data = field.data.astype(float).copy()

    bmask = None
    # --- buildings overlay: solid white RGBA, NO outline ---
    if show_buildings:
        bmask = _building_mask_from_meta(meta)
        if bmask is not None and np.any(bmask):
            rgba = np.zeros((bmask.shape[0], bmask.shape[1], 4), dtype=float)
            rgba[..., 0:3] = 1.0  # RGB=白
            rgba[..., 3] = 0.0  # 默认透明
            rgba[bmask, 3] = 1.0  # 建筑区域不透明

            ax.imshow(
                rgba,
                origin="lower",
                interpolation="nearest",
                extent=extent,
                zorder=100
            )

    bmask = _building_mask_from_meta(meta)
    print("bmask sum:", 0 if bmask is None else int(bmask.sum()))

    # 非建筑 NaN -> 0（让它显示成淡黄）
    if nodata_to_zero:
        if bmask is None:
            data[np.isnan(data)] = 0.0
        else:
            data[(~bmask) & np.isnan(data)] = 0.0
            # 建筑区域即便是 NaN 无所谓（后面会用白色 overlay 盖掉）
            # 如果你想更干净，也可以：data[bmask & np.isnan(data)] = 0.0

    # --- main heatmap ---
    im = ax.imshow(
        data,
        origin="lower",
        interpolation="nearest",
        extent=extent,
        cmap=cmap_obj,
        norm=norm
    )

    # 保持几何比例不变（圆不会变椭圆）
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title)

    # --- buildings overlay: white, NO outline ---
    # --- buildings overlay: solid white RGBA, NO outline ---
    if show_buildings and bmask is not None and np.any(bmask):
        rgba = np.zeros((bmask.shape[0], bmask.shape[1], 4), dtype=float)
        rgba[..., :] = (1.0, 1.0, 1.0, 1.0)  # 纯白不透明
        rgba[~bmask, 3] = 0.0  # 非建筑区域全透明

        ax.imshow(
            rgba,
            origin="lower",
            interpolation="nearest",
            extent=extent,
            zorder=10
        )

        # 不画 contour / 不画黑边（按你的要求）

    # --- colorbar with extend max ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="max")
    cbar.set_label(cbar_label if cbar_label else (field.units or ""))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
def plot_field_basic(
    field,
    *,
    title: str | None = None,
    save_path: str | None = None,
    cbar_label: str | None = None,
    cmap: str = "magma_r",
    vmin: float | None = None,
    vmax: float | None = None,
    show_buildings: bool = True,
):
    meta = field.meta
    extent = [0, meta.nu * meta.du, 0, meta.nv * meta.dv]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.set_facecolor("white")

    # 普通色标：NaN 显示浅灰，不做“NaN->0”，不做 “>vmax 白”
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#F0F0F0")

    data = field.data.astype(float)

    im = ax.imshow(
        data,
        origin="lower",
        interpolation="nearest",
        extent=extent,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title)

    # 建筑仍然可选盖白（不画边框）
    if show_buildings:
        bmask = _building_mask_from_meta(meta)
        if bmask is not None and np.any(bmask):
            rgba = np.zeros((bmask.shape[0], bmask.shape[1], 4), dtype=float)
            rgba[..., 0:3] = 1.0
            rgba[..., 3] = 0.0
            rgba[bmask, 3] = 1.0
            ax.imshow(rgba, origin="lower", interpolation="nearest", extent=extent, zorder=100)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label if cbar_label else (field.units or ""))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()




def export_field_csv(field: Field2D, path: str) -> None:
    """
    把二维场展平成 CSV，并把网格元数据写入文件头注释。
    列：i,j,value
    用于后续用 Excel/Origin/Matlab 等外部工具作图、再次导入到 Python/pandas 里做分析、分享给别人复现实验或者保存为成果快照，便于以后比对

    """
    meta = field.meta
    with open(path, "w", encoding="utf-8", newline="") as f:
        # 头部元数据
        w = f.write
        w(f"# origin: {meta.origin}\n")
        w(f"# u_vec: {meta.u_vec}\n")
        w(f"# v_vec: {meta.v_vec}\n")
        w(f"# du: {meta.du}\n")
        w(f"# dv: {meta.dv}\n")
        w(f"# nu: {meta.nu}\n")
        w(f"# nv: {meta.nv}\n")
        for k, v in meta.extra.items():
            w(f"# {k}: {v}\n")
        # 正文
        writer = csv.writer(f)
        writer.writerow(["i","j","value"])
        H, W = field.data.shape
        for j in range(H):
            for i in range(W):
                v = field.data[j, i]
                if np.isnan(v):
                    continue
                writer.writerow([i, j, float(v)])

if __name__ == "__main__":
    points_csv = r"D:\Code\PureForSim\src\probe_hits_points_y10.csv"

    F_peak = peak_map_from_points(points_csv, which="incident")
    F_arr = arrival_map_from_points(points_csv)
    F_imp = impulse_map_from_points(points_csv, which="incident", weight_col=None)
    F_hit = hitcount_map_from_points(points_csv)

    # 峰值超压：继续用你现在的 plot_field（专用规则）
    plot_field(
        F_peak,
        title="Peak Overpressure (incident)",
        cbar_label="kPa",
        save_path="peak_incident.png",
        vmax_clip=300.0
    )

    # 到达时间 / 冲量 / 命中数：用普通版
    vmax_arr = np.nanpercentile(F_arr.data, 99)
    plot_field_basic(F_arr, title="Earliest Arrival Time",
                     cbar_label="ms", vmin=0, vmax=vmax_arr, save_path="arrival_min.png")

    vmax_imp = np.nanpercentile(F_imp.data, 99)
    plot_field_basic(F_imp, title="Impulse Sum (incident)",
                     cbar_label="(unit)", vmin=0, vmax=vmax_imp, save_path="impulse_sum.png")

    vmax_hit = np.nanpercentile(F_hit.data, 99)
    plot_field_basic(F_hit, title="Hit Count",
                     cbar_label="count", vmin=0, vmax=vmax_hit, save_path="hit_count.png")

    # 快照：取全时段中值
    t0, t1 = time_range_from_points(points_csv)
    t_mid = 0.5 * (t0 + t1)
    F_t = snapshot_map(points_csv, t_ms=t_mid, which="incident", combine="sum")
    plot_field(F_t, title=f"Snapshot @ {t_mid:.1f} ms", save_path="snapshot_mid.png")
