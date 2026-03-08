from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# —— 配置 —— #
DATA_DIR    = "/home/nkd/ouyangzl/Project/data/MNIST"
OUT_DIR     = "/home/nkd/ouyangzl/Project/feature"
BIN_THRESH  = 0.5
BATCH_SIZE  = 64
NUM_WORKERS = 4

# —— 加载数据 —— #
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像转换为张量
])

train_dataset = MNIST(DATA_DIR, transform=transform, train=True, download=True)
test_dataset = MNIST(DATA_DIR, transform=transform, train=False, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# —— 统计学特征提取 —— #
def extract_features(image_array, bin_thresh=BIN_THRESH, hist_bins=16):
    """
    提取单张图像的统计学特征

    Args:
        image_array (np.ndarray): 单张图像的二维数组，形状为 (H, W)，像素范围为 0~1

    Returns:
        dict: 包含以下键值的特征字典：
            - mean_pixel_value: 所有像素的平均值
            - std_pixel_value: 所有像素的标准差
            - binary_pixel_ratio: 二值化后（阈值为0.5）像素为1的占比
    """
    # 初始化一个空字典，用于存储提取的图像特征
    features = {}
    # 计算图像所有像素的平均值，反映整体亮度水平
    features['mean_pixel_value'] = image_array.mean()  
    # 计算图像所有像素的标准差，衡量亮度的变化程度
    features['std_pixel_value'] = image_array.std()
    # 将图像进行二值化处理，像素值大于阈值 bin_thresh 的设为 1，其余设为 0
    binary_image = (image_array > bin_thresh).astype(np.uint8)
    # 计算二值化后像素值为 1 的比例，反映图像中前景（如数字笔画）所占的面积比例
    features['binary_pixel_ratio'] = binary_image.mean()
    # 计算图像像素值的直方图，将像素值范围 [0, 1] 分成 hist_bins 个区间
    hist, _ = np.histogram(image_array, bins=hist_bins, range=(0, 1))
    # 将直方图归一化，并将每个区间的比例作为特征添加到字典中
    features.update({f'hist_{i}': hist[i]/hist.sum() for i in range(hist_bins)})
    # 计算图像在垂直方向上的梯度（即亮度变化率）
    gy = np.abs(np.gradient(image_array, axis=0))
    # 计算图像在水平方向上的梯度
    gx = np.abs(np.gradient(image_array, axis=1))
    # 计算图像的梯度幅值，结合水平和垂直方向的变化，得到边缘强度图
    edge = np.sqrt(gx**2 + gy**2)
    # 计算边缘强度大于平均值的像素比例，反映图像中边缘（如笔画）密集程度
    features['edge_density'] = (edge > edge.mean()).mean()
    return features

# 批量处理数据集并存储特征
def process_loader(loader, out_csv):
    # 用于存储每一张图像的特征字典
    records = []
    
    # 遍历 DataLoader 中的所有 batch，tqdm 用于显示进度条
    for images, labels in tqdm(loader, desc=f"Processing {out_csv}"):
        # 去掉 channel 维度（因为 MNIST 是单通道灰度图，形状 [B, 1, H, W] → [B, H, W]）
        arrs = images.squeeze(1).numpy()
        
        # 遍历当前 batch 中的每一张图像及其对应标签
        for arr, lbl in zip(arrs, labels.numpy()):
            f = extract_features(arr)   # 提取统计学特征（平均值、标准差、二值化占比等）
            f['label'] = int(lbl)       # 添加标签信息
            records.append(f)           # 存入记录列表中
            
    # 将所有图像的特征记录转换为 Pandas DataFrame（表格形式）
    df = pd.DataFrame(records)
    
    # 保存到 CSV 文件（无索引列）
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} samples to {out_csv}")
    
os.makedirs(OUT_DIR, exist_ok=True)
process_loader(train_loader, os.path.join(OUT_DIR, "train_features.csv"))
process_loader(test_loader, os.path.join(OUT_DIR, "test_features.csv"))