import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# —— 配置 —— #
DATA_DIR = "/home/nkd/ouyangzl/Project/data/MNIST"
OUT_DIR = "/home/nkd/ouyangzl/Project/images"
BIN_THRESH = 0.5
BATCH_SIZE = 64
NUM_WORKERS = 4

# —— 数据加载 —— #
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像转换为张量
])

train_dataset = MNIST(DATA_DIR, transform=transform, train=True, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# —— 统计学特征提取 —— #
def extract_features(image_array, bin_thresh=BIN_THRESH, hist_bins=16):
    """
    提取单张图像的统计学特征
    """
    features = {}
    features['mean_pixel_value'] = image_array.mean()
    features['std_pixel_value'] = image_array.std()
    binary_image = (image_array > bin_thresh).astype(np.uint8)
    features['binary_pixel_ratio'] = binary_image.mean()
    hist, _ = np.histogram(image_array, bins=hist_bins, range=(0, 1))
    features.update({f'hist_{i}': hist[i] / hist.sum() for i in range(hist_bins)})
    gy = np.abs(np.gradient(image_array, axis=0))
    gx = np.abs(np.gradient(image_array, axis=1))
    edge = np.sqrt(gx**2 + gy**2)
    features['edge_density'] = (edge > edge.mean()).mean()
    return features

# 批量处理数据集并存储特征
def process_loader(loader):
    records = []
    for images, labels in tqdm(loader, desc="Processing Features"):
        arrs = images.squeeze(1).numpy()
        for arr, lbl in zip(arrs, labels.numpy()):
            f = extract_features(arr)
            f['label'] = int(lbl)
            records.append(f)
    return pd.DataFrame(records)

# 提取训练集特征
train_features_df = process_loader(train_loader)

# —— 可视化特征 —— #
def visualize_features(df, feature_name, title, xlabel, ylabel, save_path):
    """
    针对每个标签（0-9）绘制特征的柱状图
    """
    plt.figure(figsize=(10, 6))
    for label in range(10):
        label_data = df[df['label'] == label][feature_name]
        plt.hist(label_data, bins=30, alpha=0.5, label=f"Label {label}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 创建输出目录
os.makedirs(os.path.join(OUT_DIR, "feature_visualizations"), exist_ok=True)

# 可视化平均像素值
visualize_features(
    train_features_df,
    feature_name="mean_pixel_value",
    title="Distribution of Mean Pixel Value by Label",
    xlabel="Mean Pixel Value",
    ylabel="Frequency",
    save_path=os.path.join(OUT_DIR, "feature_visualizations", "mean_pixel_value_distribution.png")
)

# 可视化像素值标准差
visualize_features(
    train_features_df,
    feature_name="std_pixel_value",
    title="Distribution of Pixel Value Standard Deviation by Label",
    xlabel="Standard Deviation",
    ylabel="Frequency",
    save_path=os.path.join(OUT_DIR, "feature_visualizations", "std_pixel_value_distribution.png")
)

# 可视化二值化比例
visualize_features(
    train_features_df,
    feature_name="binary_pixel_ratio",
    title="Distribution of Binary Pixel Ratio by Label",
    xlabel="Binary Pixel Ratio",
    ylabel="Frequency",
    save_path=os.path.join(OUT_DIR, "feature_visualizations", "binary_pixel_ratio_distribution.png")
)

# 可视化边缘密度
visualize_features(
    train_features_df,
    feature_name="edge_density",
    title="Distribution of Edge Density by Label",
    xlabel="Edge Density",
    ylabel="Frequency",
    save_path=os.path.join(OUT_DIR, "feature_visualizations", "edge_density_distribution.png")
)
