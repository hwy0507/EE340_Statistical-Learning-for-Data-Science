import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
import umap

from data_augmentation import DataAugmentation

# ============================
# 参数设置
# ============================
DATA_ROOT = '/home/nkd/ouyangzl/Project/data/DDR'
SAVE_DIR  = '/home/nkd/ouyangzl/Project/Bonus_2/images'
os.makedirs(SAVE_DIR, exist_ok=True)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512
num_epochs = 10

# ============================
# 定义 CNN 模型
# ============================
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# ============================
# 初始化增强类与方法，增加原始数据方法
# ============================
da = DataAugmentation(image_size=28)
methods = {
    'original':        transforms.ToTensor(),  # 不使用增强，直接转 Tensor
    'random_rotation':  da.random_rotation,
    'gaussian_blur':    da.gaussian_blur,
    'affine_transform': da.affine_transform,
    'translation_only': da.translation_only,
    'all_transforms':   da.all_transforms,
}

# 保存各方法训练过程指标
all_metrics = {m: {'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[]} for m in methods}

# ============================
# 遍历方法：训练模型并记录指标
# ============================
for name, aug_fn in methods.items():
    print(f"\nTraining with augmentation: {name}")
    # 组合 transform
    if name == 'all_transforms':
        train_transform = transforms.Compose([aug_fn])
    elif name == 'original':
        train_transform = transforms.Compose([aug_fn])  # 仅 ToTensor
    else:
        train_transform = transforms.Compose([aug_fn, transforms.ToTensor()])
    test_transform = transforms.ToTensor()

    # 加载数据集
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(Subset(train_ds, list(range(20000))), batch_size, shuffle=True)
    test_ds = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False)

    # 初始化模型、优化器和损失函数
    model = CNN().to(device)
    opt = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()

    # 训练 & 测试
    for epoch in range(1, num_epochs+1):
        # 训练阶段
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, lbls)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += imgs.size(0)
        train_loss = total_loss / total
        train_acc = correct / total
        all_metrics[name]['train_loss'].append(train_loss)
        all_metrics[name]['train_acc'].append(train_acc)

        # 测试阶段
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = crit(out, lbls)
                total_loss += loss.item() * imgs.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += imgs.size(0)
        test_loss = total_loss / total
        test_acc = correct / total
        all_metrics[name]['test_loss'].append(test_loss)
        all_metrics[name]['test_acc'].append(test_acc)

        print(f"{name} Epoch {epoch}: TL={train_loss:.4f}, TA={train_acc:.4f}, VL={test_loss:.4f}, VA={test_acc:.4f}")

# ============================
# 绘制各方法的综合 Loss 和 Accuracy 曲线
# ============================
epochs = list(range(1, num_epochs+1))

# 绘制 Loss 综合图
plt.figure(figsize=(8,6))
for name in methods:
    plt.plot(epochs, all_metrics[name]['train_loss'], label=f"{name} Train", linestyle='-')
    plt.plot(epochs, all_metrics[name]['test_loss'],  label=f"{name} Test" , linestyle='--')
plt.title('Combined Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'combined_loss.png'))
plt.close()

# 绘制 Accuracy 综合图
plt.figure(figsize=(8,6))
for name in methods:
    plt.plot(epochs, all_metrics[name]['train_acc'], label=f"{name} Train", linestyle='-')
    plt.plot(epochs, all_metrics[name]['test_acc'],  label=f"{name} Test" , linestyle='--')
plt.title('Combined Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'combined_accuracy.png'))
plt.close()

print("All plots saved successfully.")
