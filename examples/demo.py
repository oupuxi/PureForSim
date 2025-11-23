import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据准备（与原始图完全对齐） ---
confusion_matrix = np.array([
    [13, 2, 0],    # 事实数值提取
    [11, 3, 1],    # 表格/列表理解
    [7, 2, 1],     # 图表/曲线解读
    [3, 1, 1],     # 图文交叉推理
    [4, 1, 0]      # 流程方法描述
])

problem_types = ['事实数值提取', '表格/列表理解', '图表/曲线解读', '图文交叉推理', '流程方法描述']
prediction_categories = ['完全正确', '部分正确', '错误/幻觉']

# --- 2. 全局设置（LaTeX学术风格+中文兼容） ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

# --- 3. 绘制混淆矩阵热力图 ---
fig, ax = plt.subplots(figsize=(9, 7))  # 适配论文栏宽
sns.heatmap(
    confusion_matrix,
    annot=True,            # 显示数值
    fmt='d',               # 整数格式
    cmap='viridis',        # 与原始图配色接近的渐变（区分度高）
    xticklabels=prediction_categories,
    yticklabels=problem_types,
    ax=ax,
    cbar=True,             # 显示颜色条
    linewidths=0.8,        # 格子边框
    annot_kws={"size": 12, "weight": "bold"},  # 标注样式
    vmin=0, vmax=13        # 颜色范围与最大值对齐
)

# --- 4. 图表细节优化（LaTeX规范） ---
ax.set_xlabel('预测类别', fontsize=14, fontweight='bold')
ax.set_ylabel('问题类型', fontsize=14, fontweight='bold')
ax.set_title('Alloy-QA 测试集错误类型混淆矩阵', fontsize=16, fontweight='bold', pad=20)

# 调整刻度与标签对齐
ax.tick_params(axis='x', labelsize=11, rotation=0, pad=10)
ax.tick_params(axis='y', labelsize=11, rotation=0, pad=10)

# 确保布局紧凑，兼容LaTeX插入
plt.tight_layout()

# --- 5. 保存高清图片（适配印刷） ---
output_filename = 'fig_confusion_matrix_final.png'
plt.savefig(output_filename, dpi=600, bbox_inches='tight', facecolor='white')

print(f"完全匹配原始图的混淆矩阵已保存为：{output_filename}")