import pandas as pd

# 定义英文到中文的映射字典（根据TID_CLASSES补充）
category_mapping = {
    # 来自TID_CLASSES的标准翻译
    'additive_gaussian_noise': '加性高斯噪声',
    'additive_noise_in_color_components': '彩色分量加性噪声',
    'spatially_correlated_noise': '空间相关噪声',
    'masked_noise': '掩模噪声',
    'high_frequency_noise': '高频噪声',
    'impulse_noise': '脉冲噪声',
    'quantization_noise': '量化噪声',
    'gaussian_blur': '高斯模糊',
    'image_denoising': '图像去噪',
    'jpeg_compression': 'JPEG压缩',  # 注意原始数据中的拼写错误
    'jpeg2000_compression': 'JPEG2000压缩',
    'jpeg_transmission_errors': 'JPEG传输错误',
    'jpeg2000_transmission_errors': 'JPEG2000传输错误',
    'non_eccentricity_pattern_noise': '非偏心模式噪声',
    'local_block-wise_distortions': '局部块状失真',
    'mean shift': '均值偏移',
    'contrast_change': '对比度变化',
    'change of color saturation': '色彩饱和度变化',
    'multiplicative_gaussian_noise': '乘性高斯噪声',
    'comfort_noise': '舒适噪声',
    'lossy_compression_of_color_images': '噪声图像有损压缩',
    'color_quantization_with_dithering': '抖动颜色量化',
    'chromatic_aberrations': '色差',
    'sparse_sampling_and_reconstruction': '稀疏采样重建'
}

# 处理 KonIQ 数据集
df_kon = pd.read_csv("category_stats_kon.csv")
df_kon["Category"] = df_kon["Category"].map(category_mapping)
df_kon.to_csv("category_stats_kon_中文.csv", index=False, encoding='utf_8_sig')

# 处理 SPAQ 数据集
df_spaq = pd.read_csv("category_stats_spaq.csv")
df_spaq["Category"] = df_spaq["Category"].map(category_mapping)
df_spaq.to_csv("category_stats_spaq_中文.csv", index=False, encoding='utf_8_sig')

print("翻译完成，文件已保存为: category_stats_kon_中文.csv 和 category_stats_spaq_中文.csv")