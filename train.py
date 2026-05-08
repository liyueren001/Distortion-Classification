import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import get_data_loaders


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. 定义模型
class DistortionClassifier(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.backbone.set_grad_checkpointing(True)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# 2. 训练函数
def train_model():
    # 初始化
    device = torch.device('cuda')
    model = DistortionClassifier().to(device)

    # 获取DataLoader
    train_loader, val_loader = get_data_loaders("C:/Users/31064\Pictures/tid2013")

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_acc = 0.0
    for epoch in range(30):
        model.train()
        train_loss = 0.0

        # 训练阶段（使用tqdm显示进度条）
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        # 打印统计信息
        val_acc = 100. * correct / total
        print(f'Epoch {epoch + 1}: '
              f'Train Loss: {train_loss / len(train_loader):.4f} | '
              f'Val Loss: {val_loss / len(val_loader):.4f} | '
              f'Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model1.pth')
            print(f'Model saved with acc {val_acc:.2f}%')

if __name__ == '__main__':
    train_model()
