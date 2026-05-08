import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置全局字体参数
rcParams.update({
    'font.sans-serif': ['SimHei'],      # 使用黑体显示中文
    'axes.unicode_minus': False,        # 解决负号显示问题
    'font.size': 14,                    # 全局字体大小
    'axes.titlesize': 18,               # 标题字体大小
    'axes.labelsize': 16,               # 坐标轴标签字体大小
    'xtick.labelsize': 12,              # X轴刻度字体大小
    'ytick.labelsize': 14               # Y轴刻度字体大小
})

# 读取CSV文件
data = pd.read_csv('category_stats_spaq_中文.csv')

# 按count列降序排序
data_sorted = data.sort_values('count', ascending=False)

# 创建图表（增加画布高度）
plt.figure(figsize=(12, 10))  # 高度从8增加到10

# 创建柱状图
bars = plt.barh(data_sorted['Category'], data_sorted['count'],
                color=plt.cm.viridis_r(np.linspace(0.2, 0.8, len(data_sorted))))

# 添加数据标签（增大标签字体）
for bar in bars:
    width = bar.get_width()
    plt.text(width + 5,
             bar.get_y() + bar.get_height()/2,
             f'{int(width)}',
             va='center',
             ha='left',
             fontsize=12)  # 增加数据标签字号

# 美化图表
plt.title('SPAQ噪声类别统计', pad=20)
plt.xlabel('出现次数', labelpad=12)
plt.ylabel('分类', labelpad=12)
plt.xlim(0, data_sorted['count'].max() * 1.15)  # 增加右侧留白空间
plt.grid(axis='x', linestyle='--', alpha=0.6)

# 调整Y轴刻度间距（防止标签重叠）
plt.tight_layout(pad=2.0)  # 增加布局留白
plt.show()