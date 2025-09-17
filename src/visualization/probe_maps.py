"""
把 probe_hits_points.csv / probe_hits_cells.csv 解析为二维格网图层，并提供导出与绘图工具。
- 自动从 CSV 头部注释(# key: value)解析网格元数据：origin/u_vec/v_vec/du/dv/nu/nv 等
- 生成四类静态图：峰压(max)、到达时间(min)、冲量(sum)、命中次数(count)
- 生成任意时刻快照：Friedlander 解析式 P(t) 并按格子合成(sum/max)

它不负责追踪/物理计算（那些在 environment/ 和 physics/ 中）；

它不负责读写 RayPath（IO 可在 utils/ 里），但应提供从 CSV/DataFrame 直接生成图层的函数；

它的产出是：二维数组 + 元数据（可保存为 PNG/CSV/NPZ）以及可选的动画
"""


