import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os

# —— 配置 —— #
DATA_DIR = "/home/nkd/ouyangzl/Project/data/MNIST"
IMAGE_DIR = "/home/nkd/ouyangzl/Project/images"
BATCH_SIZE = 512
SEED = 42

# 创建图片保存目录
os.makedirs(IMAGE_DIR, exist_ok=True)

# 设置随机种子以保证结果可复现
np.random.seed(SEED)
torch.manual_seed(SEED)

# —— 数据加载 —— #
# 定义数据预处理（将图像转换为张量）
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载 MNIST 数据集
train_dataset = MNIST(DATA_DIR, transform=transform, train=True, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 提取一批数据用于降维和可视化
images, labels = next(iter(train_loader))  # 从 DataLoader 中取出一个 batch
images = images.view(images.size(0), -1).numpy()  # 将图像展平为二维数组 (batch_size, 28*28)
labels = labels.numpy()  # 提取标签

# —— 数据降维与可视化 —— #
def visualize_with_tsne(data, labels, title="t-SNE Visualization", save_path=None):
    """
    使用 t-SNE 对数据进行降维并可视化。
    Args:
        - data: 输入数据，形状为 (num_samples, num_features)
        - labels: 数据对应的标签，形状为 (num_samples,)
        - title: 图像标题
        - save_path: 图片保存路径
    """
    tsne = TSNE(n_components=2, random_state=SEED)
    reduced_data = tsne.fit_transform(data)  # 降维到 2D
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label="Digit Label")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_with_umap(data, labels, title="UMAP Visualization", save_path=None):
    """
    使用 UMAP 对数据进行降维并可视化。
    Args:
        - data: 输入数据，形状为 (num_samples, num_features)
        - labels: 数据对应的标签，形状为 (num_samples,)
        - title: 图像标题
        - save_path: 图片保存路径
    """
    reducer = umap.UMAP(random_state=SEED)
    reduced_data = reducer.fit_transform(data)  # 降维到 2D
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label="Digit Label")
    plt.title(title)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 可视化原始数据的分布
visualize_with_tsne(images, labels, title="t-SNE Visualization of Original MNIST Data", save_path=os.path.join(IMAGE_DIR, "tsne_original.png"))
visualize_with_umap(images, labels, title="UMAP Visualization of Original MNIST Data", save_path=os.path.join(IMAGE_DIR, "umap_original.png"))

# —— 数据增强与样本分布可视化 —— #
# 定义数据增强方法
augment_transform = transforms.Compose([
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # 随机高斯模糊
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomCrop(size=(28, 28), padding=4),  # 随机裁剪
    transforms.ToTensor()
])

# 加载增强后的数据集
augmented_dataset = MNIST(DATA_DIR, transform=augment_transform, train=True, download=True)
augmented_loader = DataLoader(augmented_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 提取增强后的数据
aug_images, aug_labels = next(iter(augmented_loader))
aug_images = aug_images.view(aug_images.size(0), -1).numpy()
aug_labels = aug_labels.numpy()

# 可视化增强后的数据分布
visualize_with_tsne(aug_images, aug_labels, title="t-SNE Visualization of Augmented MNIST Data", save_path=os.path.join(IMAGE_DIR, "tsne_augmented.png"))
visualize_with_umap(aug_images, aug_labels, title="UMAP Visualization of Augmented MNIST Data", save_path=os.path.join(IMAGE_DIR, "umap_augmented.png"))

# —— 数据增强方法的有效性分析 —— #
def compare_distributions(original_data, augmented_data, labels, title="Comparison of Original and Augmented Data", save_path=None):
    """
    比较原始数据和增强数据的分布。
    Args:
        - original_data: 原始数据，形状为 (num_samples, num_features)
        - augmented_data: 增强数据，形状为 (num_samples, num_features)
        - labels: 数据对应的标签，形状为 (num_samples,)
        - title: 图像标题
        - save_path: 图片保存路径
    """
    tsne = TSNE(n_components=2, random_state=SEED)
    combined_data = np.vstack([original_data, augmented_data])  # 合并原始数据和增强数据
    combined_labels = np.hstack([labels, labels + 10])  # 增强数据的标签加 10 以区分
    reduced_data = tsne.fit_transform(combined_data)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=combined_labels, cmap='tab20', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(20), label="Digit Label (0-9: Original, 10-19: Augmented)")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    if save_path:
        plt.savefig(save_path)
    plt.show()

# —— 基于 UMAP 的原始 vs 增强 数据分布对比 —— #
def compare_distribution_umap(original_data, augmented_data, labels, title="UMAP Comparison of Original and Augmented Data",save_path=None):
    """
    使用 UMAP 比较原始数据和增强数据的分布。
    Args:
        - original_data: 原始数据，形状为 (num_samples, num_features)
        - augmented_data: 增强数据，形状为 (num_samples, num_features)
        - labels: 数据对应的标签，形状为 (num_samples,)
        - title: 图像标题
        - save_path: 图片保存路径
    """
    # 合并原始和增强数据
    combined_data = np.vstack([original_data, augmented_data])
    # 为增强数据的标签加上一个偏移，以便在图中区分
    combined_labels = np.hstack([labels, labels + 10])
    
    # 使用 UMAP 进行降维
    reducer = umap.UMAP(n_components=2, random_state=SEED)
    reduced_data = reducer.fit_transform(combined_data)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=combined_labels,
        cmap='tab20',
        s=10, alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(20), label="Digit Label (0-9: Original, 10-19: Augmented)")
    plt.title(title)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")  
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
# 比较原始数据和增强数据的分布
compare_distributions(images, aug_images, labels, title="t-SNE Comparison of Original and Augmented Data", save_path=os.path.join(IMAGE_DIR, "comparison.png"))
# 调用 compare_distribution_umap
compare_distribution_umap(
    images, aug_images, labels,
    title="UMAP Comparison of Original and Augmented MNIST Data",
    save_path=os.path.join(IMAGE_DIR, "umap_comparison.png")
)