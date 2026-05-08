import pandas as pd

# 读取CSV文件
df = pd.read_csv('prediction——spaq.csv')

# 统计每个类别的图片数量
category_counts = df['pred_class'].value_counts().reset_index()
category_counts.columns = ['Category', 'Image Count']  # 重命名列

# 打印结果
print(category_counts)

# 保存结果为新CSV
category_counts.to_csv('category_stats.csv', index=False)