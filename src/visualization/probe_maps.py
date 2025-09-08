"""
把“点级命中数据”转换为“二维格网图层”，并提供导出/可视化的统一入口
它不负责追踪/物理计算（那些在 environment/ 和 physics/ 中）；

它不负责读写 RayPath（IO 可在 utils/ 里），但应提供从 CSV/DataFrame 直接生成图层的函数；

它的产出是：二维数组 + 元数据（可保存为 PNG/CSV/NPZ）以及可选的动画
"""


