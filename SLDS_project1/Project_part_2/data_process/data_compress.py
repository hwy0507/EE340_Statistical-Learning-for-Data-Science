import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Subset
from data_augmentation import DataAugmentation
import umap.umap_ as umap
from sklearn.manifold import TSNE

# 数据和保存路径
data_root = '/home/nkd/ouyangzl/Project/data/MNIST'
save_dir = '/home/nkd/ouyangzl/Project/feature'
os.makedirs(save_dir, exist_ok=True)

# 加载 MNIST 数据集（不应用任何 transform，让 DataAugmentation 自己处理 PIL Image）
mnist_train = datasets.MNIST(root=data_root, train=True, download=True, transform=None)
subset_mnist = Subset(mnist_train, list(range(20000)))

# 初始化数据增强类
da = DataAugmentation(image_size=28)

# 将 DataAugmentation 中的各个方法（属性）整合到一个字典中，便于循环使用
augmentation_methods = {
    'random_rotation': da.random_rotation,
    'gaussian_blur': da.gaussian_blur,
    'affine_transform': da.affine_transform,
    'translation_only': da.translation_only,
    'all_transforms': da.all_transforms
}

# 针对每种数据增强方法进行循环
for method_name, transform_fn in augmentation_methods.items():
    print(f"Processing augmentation method: {method_name}")
    features = []
    labels = []
    
    for img, label in subset_mnist:
        # 对原始图像应用当前增强方法（img 为 PIL.Image 对象）
        augmented_img = transform_fn(img)
        # 将增强后的图像转换为 numpy 数组并展平（例如 28×28 的灰度图变为 784 维向量）
        img_array = np.array(augmented_img)
        features.append(img_array.flatten())
        labels.append(label)
    
    features = np.array(features)  # (20000, feature_dim)
    labels = np.array(labels)
    
    # UMAP降维
    umap_embed = umap.UMAP(n_components=2, random_state=0).fit_transform(features)
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_embed[:, 0], umap_embed[:, 1], c=labels, cmap='tab10', s=5)
    plt.title(f'UMAP - {method_name}')
    umap_save_path = os.path.join(save_dir, f'umap_{method_name}.png')
    plt.savefig(umap_save_path)
    plt.close()
    
    # t-SNE降维
    tsne_embed = TSNE(n_components=2, random_state=0).fit_transform(features)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=labels, cmap='tab10', s=5)
    plt.title(f"t-SNE - {method_name}")
    tsne_save_path = os.path.join(save_dir, f'tsne_{method_name}.png')
    plt.savefig(tsne_save_path)
    plt.close()
    
    print(f"Saved: {umap_save_path} and {tsne_save_path}")