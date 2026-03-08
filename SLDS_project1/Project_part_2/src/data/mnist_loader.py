import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

DATA_DIR    = "/home/nkd/ouyangzl/Project/data/MNIST"

def load_mnist(batch_size=512):
    transform = transforms.Compose([
        transforms.ToTensor()  # 将图像转换为张量
    ])
    # 加载训练集与测试集
    train_dataset = MNIST(DATA_DIR, transform=transform, train=True, download=True)
    test_dataset = MNIST(DATA_DIR, transform=transform, train=False, download=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 提取第一个batch用于某些需要numpy格式的数据（例如降维可视化）
    images, labels = next(iter(train_loader)) # 从 DataLoader 中取出一个 batch
    images = images.view(images.size(0), -1).numpy()  # 将图像展平为二维数组 (batch_size, 28*28)
    labels = labels
    
    return train_loader, test_loader, images, labels