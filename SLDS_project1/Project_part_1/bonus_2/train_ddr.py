import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. 配置与参数
DDR_ROOT = '/path/to/DDR'  # 修改为DDR数据集根目录
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. DDR自定义数据集
class DDRDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # label_file: 每行格式为 "imgname.jpg,3"
        with open(label_file, 'r') as f:
            lines = f.readlines()
        self.samples = []
        for line in lines:
            img, label = line.strip().split(',')
            self.samples.append((img, int(label)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 3. 图像增强方法（与MNIST实验保持一致，可扩展）
def get_transforms(aug_type='none'):
    if aug_type == 'none':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
    elif aug_type == 'hflip':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif aug_type == 'rotate':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
        ])
    elif aug_type == 'colorjitter':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])
    elif aug_type == 'all':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError("Unknown aug_type")

# 4. 简单CNN模型（与MNIST实验一致，也可用ResNet等）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. 训练与评估函数
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)

# 6. 主流程：不同增强方法实验
def main():
    # 假设DDR数据集已分为train/val，且有对应label文件
    aug_types = ['none', 'hflip', 'rotate', 'colorjitter', 'all']
    results = {}
    for aug in aug_types:
        print(f"\n=== Training with augmentation: {aug} ===")
        train_transform = get_transforms(aug)
        val_transform = get_transforms('none')
        train_set = DDRDataset(
            img_dir=os.path.join(DDR_ROOT, 'train_images'),
            label_file=os.path.join(DDR_ROOT, 'train_labels.txt'),
            transform=train_transform
        )
        val_set = DDRDataset(
            img_dir=os.path.join(DDR_ROOT, 'val_images'),
            label_file=os.path.join(DDR_ROOT, 'val_labels.txt'),
            transform=val_transform
        )
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
        model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")
        preds, labels = evaluate(model, val_loader)
        acc = (preds == labels).mean()
        print(f"Validation Accuracy: {acc:.4f}")
        print(classification_report(labels, preds, digits=4))
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap='Blues')
        plt.title(f'Confusion Matrix ({aug})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join('./', f'cm_{aug}.png'))
        plt.close()
        results[aug] = acc
    # 绘制不同增强方法的准确率对比
    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.ylabel('Accuracy')
    plt.title('Augmentation Comparison on DDR')
    plt.savefig(os.path.join('./', 'augmentation_comparison.png'))
    plt.show()

if __name__ == '__main__':
    main()

"""
任务完成度说明：
1. 实现了DDR眼底彩照数据集的自定义加载与标签读取。
2. 复用了MNIST实验中的CNN结构与训练流程，支持多种图像增强方法。
3. 支持多种增强方法的实验，自动训练、评估、保存混淆矩阵和准确率对比图。 
4. 详细输出每种增强下的分类报告和混淆矩阵，便于分析不同增强方法和数据集的效果差异。
5. 代码结构清晰，便于扩展和对比MNIST实验结果。
6. 可根据实验结果分析DDR与MNIST在模型表现和增强敏感性上的差异，并尝试解释原因。
"""
