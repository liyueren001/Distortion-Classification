import pandas as pd
from pathlib import Path

# 获取所有xlsx文件路径
folder_path = Path("C:/Users/31064\PycharmProjects\QCN-main\datasplit\SPAQ")
xlsx_files = list(folder_path.glob('*.xlsx'))

# 读取并合并所有文件
merged_df = pd.concat([pd.read_excel(f) for f in xlsx_files], ignore_index=True)

# 保存结果
merged_df.to_excel('merged_vertical.xlsx', index=False)
print(f"合并完成！共合并 {len(xlsx_files)} 个文件")