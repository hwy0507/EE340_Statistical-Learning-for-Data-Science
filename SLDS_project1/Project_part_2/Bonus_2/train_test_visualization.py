import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.manifold import TSNE
import umap
import torch
from PIL import Image

from data_augmentation import DataAugmentation

# 处理图像为 NumPy 格式（支持 PIL.Image 和 Tensor）
def to_numpy_flat(img):
    if isinstance(img, torch.Tensor):
        arr = img.squeeze().numpy()
    else:
        arr = np.array(img)
    return arr.flatten()  # 展平为向量

# 参数配置
# DDR 数据集目录结构：
# /home/nkd/ouyangzl/Project/data/DDR/DDR-dataset/DR_grading/train/*.jpg
# /home/nkd/ouyangzl/Project/data/DDR/DDR-dataset/DR_grading/test/*.jpg
# 同时，标注文件位于：
# /home/nkd/ouyangzl/Project/data/DDR/DDR-dataset/DR_grading/train.txt
# /home/nkd/ouyangzl/Project/data/DDR/DDR-dataset/DR_grading/test.txt
DATA_ROOT = '/home/nkd/ouyangzl/Project/data/DDR/DDR-dataset/DR_grading'
SAVE_DIR  = '/home/nkd/ouyangzl/Project/Bonus_2/images'
os.makedirs(SAVE_DIR, exist_ok=True)

# 标注文件路径
train_label_file = os.path.join(DATA_ROOT, 'train.txt')
test_label_file  = os.path.join(DATA_ROOT, 'test.txt')

def load_labels(label_file, image_dir):
    """
    读取 label 文件，每行格式 assumed: <filename> <label>
    返回列表： [(PIL.Image, label), ...]
    """
    data = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        filename, label = parts[0], parts[1]
        img_path = os.path.join(image_dir, filename)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                data.append((img, int(label)))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return data

# 加载训练集和测试集数据
train_dir = os.path.join(DATA_ROOT, 'train')
test_dir  = os.path.join(DATA_ROOT, 'test')

train_data = load_labels(train_label_file, train_dir)
test_data  = load_labels(test_label_file, test_dir)

# 采样：取 5000 张训练图片和 2000 张测试图片进行可视化
train_subset = train_data[:3000]
test_subset  = test_data[:1000]

# 初始化数据增强类及方法（这里设置目标图像尺寸为64×64，可根据需要调整）
da = DataAugmentation(image_size=64)
methods = {
    'random_rotation':  da.random_rotation,
    'gaussian_blur':    da.gaussian_blur,
    'affine_transform': da.affine_transform,
    'translation_only': da.translation_only,
    'all_transforms':   da.all_transforms,
}

# 主循环：每种增强方法画一张图，包含 4 个子图：Train-TSNE, Train-UMAP, Test-TSNE, Test-UMAP
for name, aug_fn in methods.items():
    print(f"Processing augmentation: {name}")
    # 提取训练集特征：对训练数据应用增强（aug_fn）后转换为向量
    train_feats, train_labels = [], []
    for img, lbl in train_subset:
        aug_img = aug_fn(img)
        vec = to_numpy_flat(aug_img)
        train_feats.append(vec)
        train_labels.append(lbl)
    train_feats = np.stack(train_feats)
    train_labels = np.array(train_labels)

    # 提取测试集特征（不应用增强，直接用原图）
    test_feats, test_labels = [], []
    for img, lbl in test_subset:
        vec = to_numpy_flat(img)
        test_feats.append(vec)
        test_labels.append(lbl)
    test_feats = np.stack(test_feats)
    test_labels = np.array(test_labels)

    # 计算降维结果：TSNE 和 UMAP
    tsne = TSNE(n_components=2, random_state=0)
    um = umap.UMAP(n_components=2, random_state=0)
    train_tsne = tsne.fit_transform(train_feats)
    train_umap = um.fit_transform(train_feats)
    test_tsne  = tsne.fit_transform(test_feats)
    test_umap  = um.fit_transform(test_feats)

    # 绘制图形：共 2x2 个子图
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    # Train TSNE
    axes[0].scatter(train_tsne[:,0], train_tsne[:,1], c=train_labels, s=5, cmap='tab10')
    axes[0].set_title('Train TSNE')
    axes[0].axis('off')
    # Train UMAP
    axes[1].scatter(train_umap[:,0], train_umap[:,1], c=train_labels, s=5, cmap='tab10')
    axes[1].set_title('Train UMAP')
    axes[1].axis('off')
    # Test TSNE
    axes[2].scatter(test_tsne[:,0], test_tsne[:,1], c=test_labels, s=5, cmap='tab10')
    axes[2].set_title('Test TSNE')
    axes[2].axis('off')
    # Test UMAP
    axes[3].scatter(test_umap[:,0], test_umap[:,1], c=test_labels, s=5, cmap='tab10')
    axes[3].set_title('Test UMAP')
    axes[3].axis('off')

    plt.suptitle(f"Distribution: {name}")
    out_path = os.path.join(SAVE_DIR, f"DDR_{name}_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

print("✅ DDR 数据集可视化完成！")