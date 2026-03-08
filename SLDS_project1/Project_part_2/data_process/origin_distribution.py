import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.manifold import TSNE
import umap
import torch

# 处理图像为 NumPy 格式（支持 PIL 和 Tensor）
def to_numpy_flat(img):
    if isinstance(img, torch.Tensor):
        arr = img.squeeze().numpy()
    else:
        arr = np.array(img)
    return arr.flatten()  # 展平为向量

# 参数配置
DATA_ROOT = '/home/nkd/ouyangzl/Project/data/MNIST'
SAVE_DIR  = '/home/nkd/ouyangzl/Project/feature/distributions'
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载原始 MNIST（不作 transform）
mnist_train = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=None)
mnist_test = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=None)

# 采样方便可视化，2000 个训练，2000 个测试
train_subset = Subset(mnist_train, list(range(2000)))
test_subset  = Subset(mnist_test, list(range(2000)))

# 提取训练特征
train_feats, train_labels = [], []
for img, lbl in train_subset:
    vec = to_numpy_flat(img)
    train_feats.append(vec)
    train_labels.append(lbl)
train_feats = np.stack(train_feats)
train_labels = np.array(train_labels)

# 提取测试特征
test_feats, test_labels = [], []
for img, lbl in test_subset:
    vec = to_numpy_flat(img)
    test_feats.append(vec)
    test_labels.append(lbl)
test_feats = np.stack(test_feats)
test_labels = np.array(test_labels)

# 计算降维
tsne = TSNE(n_components=2, random_state=0)
um = umap.UMAP(n_components=2, random_state=0)
train_tsne = tsne.fit_transform(train_feats)
train_umap = um.fit_transform(train_feats)
test_tsne  = tsne.fit_transform(test_feats)
test_umap  = um.fit_transform(test_feats)

# 绘图
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

plt.suptitle("Distribution: Original (No Augmentation)")
out_path = os.path.join(SAVE_DIR, "original_distribution.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")
