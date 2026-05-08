import os
import shutil
import pandas as pd
from tqdm import tqdm


def organize_images_by_category(csv_path, image_dir, output_root):
    """
    根据CSV中的分类结果整理图片到类别文件夹

    参数:
        csv_path: 包含预测结果的CSV路径（需有'filename'和'pred_class'列）
        image_dir: 原始图像文件夹路径
        output_root: 分类结果根目录
    """
    # 读取预测结果
    df = pd.read_csv(csv_path)

    # 确保输出目录存在
    os.makedirs(output_root, exist_ok=True)

    # 统计处理情况
    copied_files = 0
    missing_files = 0

    # 遍历每一行预测结果
    for _, row in tqdm(df.iterrows(), total=len(df), desc="分类整理中"):
        src_path = os.path.join(image_dir, row['filename'])
        category_dir = os.path.join(output_root, row['pred_class'])

        # 创建类别文件夹（如果不存在）
        os.makedirs(category_dir, exist_ok=True)

        # 复制文件
        if os.path.exists(src_path):
            shutil.copy2(src_path, category_dir)
            copied_files += 1
        else:
            missing_files += 1
            print(f"文件缺失: {src_path}")

    # 输出统计报告
    print(f"\n整理完成！")
    print(f"成功分类 {copied_files} 张图片")
    print(f"缺失文件 {missing_files} 个")
    print(f"分类目录结构已创建在: {output_root}")


# 使用示例
if __name__ == '__main__':
    organize_images_by_category(
        csv_path='prediction——kon.csv',  # 预测结果CSV
        image_dir='low_mos_dataset',  # 原图目录
        output_root='classified_results_Koniq10k/'  # 输出根目录
    )