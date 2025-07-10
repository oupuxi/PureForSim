import numpy as np
import matplotlib.pyplot as plt
from src.physics.calculate_blast_parameters import *
# ── 1. 基本输入 ───────────────────────────────────────
charge_weight = 50        # kg TNT
path = [(0, 0), (50, 0)]   # 爆心 → 100 m（自由场）
# ── 2. 关键参数 ───────────────────────────────────────
Ps  = calculate_friedlander_peak_overpressure(charge_weight, path)          # kPa
td  = calculate_friedlander_td(charge_weight, 0, 100)                       # ms
beta = calculate_friedlander_beta(charge_weight, path)                      # —
ta  = calculate_friedlander_time_arrive(path, charge_weight)                # ms
print(f"Ps  = {Ps:.1f} kPa")
print(f"td  = {td:.1f} ms")
print(f"β   = {beta:.3f}")
print(f"ta  = {ta:.1f} ms")
# ── 3. 生成 Friedlander 波形 ───────────────────────────
t = np.linspace(ta, ta + td, 400)          # 时间轴 (ms)
P = Ps * (1 - (t - ta) / td) * np.exp(-beta * (t - ta) / td)   # kPa

# ── 4. 绘图 ────────────────────────────────────────────
plt.figure(figsize=(6, 3.5))
plt.plot(t, P)
plt.title("Friedlander Pressure History\n1000 kg TNT, 100 m (free field)")
plt.xlabel("Time  (ms)")
plt.ylabel("Overpressure  (kPa)")
plt.grid(True)
plt.tight_layout()
plt.show()