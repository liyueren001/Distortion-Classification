import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


# 1. 定义TID2013数据集类
class TID2013Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: TID2013文件夹路径（包含reference_images和distorted_images）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # TID2013的24种失真类型（按官方顺序）
        self.distortion_types = [
            'additive_gaussian_noise', 'additive_noise_in_color_components',
            'spatially_correlated_noise', 'masked_noise', 'high_frequency_noise',
            'impulse_noise', 'quantization_noise', 'gaussian_blur', 'image_denoising',
            'jpeg_compression', 'jpeg2000_compression', 'jpeg_transmission_errors',
            'jpeg2000_transmission_errors', 'non_eccentricity_pattern_noise',
            'local_block-wise_distortions', 'mean shift',
            'contrast_change', 'change of color saturation' , 'multiplicative_gaussian_noise', 'comfort_noise',
            'lossy_compression_of_color_images', 'color_quantization_with_dithering',
            'chromatic_aberrations', 'sparse_sampling_and_reconstruction'
        ]

        # 自动扫描所有失真图像
        dist_dir = os.path.join(root_dir, 'distorted_images')
        for fname in os.listdir(dist_dir):
            if fname.lower().endswith('.bmp'):
                parts = fname.split('_')
                dist_type_id = int(parts[1]) - 1  # 转换为0-24
                self.samples.append({
                    'image_path': os.path.join(dist_dir, fname),
                    'label': dist_type_id,
                    'distortion_name': self.distortion_types[dist_type_id],
                    'level': int(parts[2].split('.')[0])  # 失真强度1-5
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]['image_path']).convert('RGB')
        label = self.samples[idx]['label']

        if self.transform:
            img = self.transform(img)

        return img, label


# 2. 定义数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(448, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 3. 创建数据集和DataLoader
def get_data_loaders(tid_root, batch_size=8, val_ratio=0.2):
    # 创建完整数据集
    full_dataset = TID2013Dataset(root_dir=tid_root, transform=train_transform)

    # 分割训练集和验证集
    train_size = int((1 - val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 验证集使用测试增强
    val_set.dataset.transform = test_transform

    # 创建DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

