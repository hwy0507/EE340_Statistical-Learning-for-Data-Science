import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from data_augmentation import DataAugmentation

# 设置路径和参数
data_root = '/home/nkd/ouyangzl/Project/data/MNIST'
save_dir  = '/home/nkd/ouyangzl/Project/feature'
os.makedirs(save_dir, exist_ok=True)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512
num_epochs = 10

# 定义 CNN 模型
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

# 初始化数据增强
da = DataAugmentation(image_size=28)
augmentation_methods = {
    'random_rotation':  da.random_rotation,
    'gaussian_blur':    da.gaussian_blur,
    'affine_transform': da.affine_transform,
    'translation_only': da.translation_only,
    'all_transforms':   da.all_transforms
}

for method_name, aug_fn in augmentation_methods.items():
    print(f"Training with augmentation method: {method_name}")

    # 针对 all_transforms 已经包含 ToTensor，其他方法再附加 ToTensor
    if method_name == 'all_transforms':
        train_transform = transforms.Compose([
            aug_fn
        ])
    else:
        train_transform = transforms.Compose([
            aug_fn,
            transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载数据集
    full_train = datasets.MNIST(root=data_root, train=True,
                                download=True, transform=train_transform)
    train_subset = Subset(full_train, list(range(20000)))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root=data_root, train=False,
                                 download=True, transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型、优化器、损失函数
    model     = CNN().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 训练与测试
    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []

    for epoch in range(1, num_epochs + 1):
        # 训练
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # 测试
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += images.size(0)

        test_loss = test_loss / total
        test_acc  = correct / total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # 绘图保存
    epochs = list(range(1, num_epochs + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, test_losses,  label='Test Loss')
    ax1.set_title(f"Loss - {method_name}");   ax1.legend()
    ax2.plot(epochs, train_accs,  label='Train Acc')
    ax2.plot(epochs, test_accs,   label='Test Acc')
    ax2.set_title(f"Accuracy - {method_name}"); ax2.legend()

    plt.suptitle(f"Metrics with {method_name}")
    plt.savefig(os.path.join(save_dir, f"{method_name}_metrics.png"))
    plt.close()
    print(f"✅ Saved metrics plot for {method_name}\n")
