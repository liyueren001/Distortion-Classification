import pandas as pd
import os
import shutil


def filter_low_mos_images(metadata_path, image_dir, output_dir, threshold=50):
    """
    筛选MOS评分低于阈值的图像

    参数:
        metadata_path: 包含MOS评分的CSV文件路径
        image_dir: 原始图像文件夹路径
        output_dir: 输出文件夹路径
        threshold: MOS阈值 (默认0.5)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"错误：未找到元数据文件 {metadata_path}")
        return

    # 筛选图像
    low_mos_df = df[df['MOS'] < threshold]
    print(f"找到 {len(low_mos_df)} 张MOS < {threshold}的图像")

    # 复制图像
    missing_files = 0
    for img_name in low_mos_df['image_name']:
        src_path = os.path.join(image_dir, img_name)
        dst_path = os.path.join(output_dir, img_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            missing_files += 1

    if missing_files > 0:
        print(f"警告：有 {missing_files} 张图像未找到")

    print(f"处理完成！图像已保存到 {output_dir}")


# 使用示例
if __name__ == '__main__':
    filter_low_mos_images(
        metadata_path="output_SPAQ.csv",
        image_dir="C:/Users/31064\Pictures\imagesresize\SPAQ\Images_resized_384",
        output_dir='low_mos_dataset_SPAQ/',
        threshold=50
    )