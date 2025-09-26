# src/utils/export.py
from __future__ import annotations
import csv, json, datetime as _dt
from typing import Dict, Tuple, List
from src.data_struct.data_structs import ProbeGrid


def _build_grid_meta(grid: ProbeGrid, *, format_name: str, extra: dict | None = None) -> dict:
    """把 ProbeGrid 的几何与当前运行摘要打包成 JSON 友好的元数据字典。"""
    meta = {
        "format": format_name,              # 'probe_hits_points' 或 'probe_hits_cells'
        "version": "1.0",
        "exported_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "units": { "length": "m", "pressure": "kPa", "time": "ms", "impulse": "kPa·ms" },
        "grid": {
            "origin": list(grid.origin), "u_vec": list(grid.u_vec), "v_vec": list(grid.v_vec),
            "du": float(grid.du), "dv": float(grid.dv), "nu": int(grid.nu), "nv": int(grid.nv),
        },
        "summary": {
            "unique_hit_cells": int(len(grid.cells)),
            "total_hits": int(sum(len(h) for h in grid.cells.values())),
        }
    }
    if extra: meta["context"] = extra
    return meta

def export_probe_hits_csv(grid: ProbeGrid, path: str, *, run_meta: dict | None = None) -> None:
    """逐条导出命中（点级），CSV 头加 JSON 元数据注释。"""
    fields = [
        "i","j","x","y","z",
        "incident_peak_pressure_kpa","reflected_peak_pressure_kpa",
        "incident_impulse","reflected_impulse",
        "arrive_time_ms","constant_time_ms","friedlander_beta",
    ]
    meta = _build_grid_meta(grid, format_name="probe_hits_points", extra=run_meta)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# meta=" + json.dumps(meta, ensure_ascii=False) + "\n")
        f.write("# columns=" + ",".join(fields) + "\n")
        w = csv.writer(f); w.writerow(fields)
        for (i,j), hits in grid.cells.items():
            for n in hits:
                x, y, z = n.coord
                w.writerow([
                    i, j, x, y, z,
                    getattr(n,"incident_peak_pressure_kpa",0.0),
                    getattr(n,"reflected_peak_pressure_kpa",0.0),
                    getattr(n,"incident_impulse",0.0),
                    getattr(n,"reflected_impulse",0.0),
                    getattr(n,"arrive_time_ms",0.0),
                    getattr(n,"constant_time_ms",0.0),
                    getattr(n,"friedlander_beta",0.0),
                ])

def export_probe_cell_stats_csv(grid: ProbeGrid, path: str, *, run_meta: dict | None = None) -> None:
    """每格聚合（命中次数 / 峰压 max / 最早到达 min），CSV 头加元数据与口径说明。"""
    fields = ["i","j","hits","Pmax_inc_kPa","t_first_ms"]
    meta = _build_grid_meta(grid, format_name="probe_hits_cells", extra=run_meta)
    meta["aggregation"] = {
        "Pmax_inc_kPa": "max over incident_peak_pressure_kpa within cell",
        "t_first_ms":   "min over arrive_time_ms within cell",
        "hits":         "count of hits within cell"
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# meta=" + json.dumps(meta, ensure_ascii=False) + "\n")
        f.write("# columns=" + ",".join(fields) + "\n")
        w = csv.writer(f); w.writerow(fields)
        for (i,j), hits in grid.cells.items():
            if not hits: continue
            pmax = max(getattr(h,"incident_peak_pressure_kpa",0.0) for h in hits)
            tmin = min(getattr(h,"arrive_time_ms",float("inf")) for h in hits)
            w.writerow([i, j, len(hits), pmax, tmin])
