import os
from PIL import Image  # 图像加载和处理
import pandas as pd    # 结果保存为CSV
import torch           # PyTorch基础库
from torch.utils.data import Dataset, DataLoader  # 数据加载
from torchvision import transforms  # 图像预处理
from tqdm import tqdm  # 进度条显示
from model import test_transform, train_transform
from train import DistortionClassifier
from model import TID2013Dataset

TID_CLASSES = [
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


class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_files = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
        ]
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        return self.transform(img), os.path.basename(self.image_files[idx])

def predict_folder(model, folder_path, output_csv='prediction——spaq.csv'):
    dataset = PredictionDataset(folder_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    results = []
    with torch.no_grad():
        for images, filenames in tqdm(loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            for fn, pred, prob in zip(filenames, preds, probs):
                results.append({
                    'filename': fn,
                    'pred_class': TID_CLASSES[pred.item()],
                    'confidence': prob[pred].item(),
                    'top3_classes': [
                        (TID_CLASSES[i], p.item())
                        for i, p in zip(torch.topk(prob, 3).indices, torch.topk(prob, 3).values)
                    ]
                })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"预测结果已保存到 {output_csv}")


if __name__ == '__main__':
    device = torch.device('cuda')
    model = DistortionClassifier().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()  # 切换到评估模式
    predict_folder(model, "low_mos_dataset_SPAQ")