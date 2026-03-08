import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.manifold import TSNE
import umap.umap_ as umap

# 数据和保存路径
DATA_ROOT = '/home/nkd/ouyangzl/Project/data/MNIST'
SAVE_DIR  = '/home/nkd/ouyangzl/Project/feature'
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载 MNIST 数据集（不使用任何数据增强）
mnist_train = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transforms.ToTensor())
subset_mnist = Subset(mnist_train, list(range(20000)))

features = []
labels = []

print("Processing original images (no augmentation)...")
for img, label in subset_mnist:
    img_array = img.numpy().squeeze()  # 去除 channel 维度 (1, 28, 28) -> (28, 28)
    features.append(img_array.flatten())
    labels.append(label)

features = np.array(features)
labels = np.array(labels)

# UMAP降维
umap_embed = umap.UMAP(n_components=2, random_state=0).fit_transform(features)
plt.figure(figsize=(8, 6))
plt.scatter(umap_embed[:, 0], umap_embed[:, 1], c=labels, cmap='tab10', s=5)
plt.title('UMAP - original (no augmentation)')
umap_save_path = os.path.join(SAVE_DIR, 'umap_original.png')
plt.savefig(umap_save_path)
plt.close()

# t-SNE降维
tsne_embed = TSNE(n_components=2, random_state=0).fit_transform(features)
plt.figure(figsize=(8, 6))
plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=labels, cmap='tab10', s=5)
plt.title('t-SNE - original (no augmentation)')
tsne_save_path = os.path.join(SAVE_DIR, 'tsne_original.png')
plt.savefig(tsne_save_path)
plt.close()

print(f"Saved: {umap_save_path} and {tsne_save_path}")
