import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Subset
from data_augmentation import DataAugmentation
import torch

# ✅ 处理图像为 NumPy 格式（支持 PIL 和 Tensor）
def to_numpy_for_plot(img):
    if isinstance(img, torch.Tensor):
        return img.squeeze().numpy()  # 去掉通道维度 (1, H, W) → (H, W)
    else:
        return np.array(img)

# 数据路径和保存路径
data_root = '/home/nkd/ouyangzl/Project/data/MNIST'
save_dir = '/home/nkd/ouyangzl/Project/feature'
os.makedirs(save_dir, exist_ok=True)

# 加载 MNIST 数据集（不应用任何 transform，让 DataAugmentation 自己处理 PIL Image）
mnist_train = datasets.MNIST(root=data_root, train=True, download=True, transform=None)
subset_mnist = Subset(mnist_train, list(range(20000)))

# 初始化数据增强类
da = DataAugmentation(image_size=28)

# 将 DataAugmentation 中的各个方法整合到一个字典中
augmentation_methods = {
    'random_rotation': da.random_rotation,
    'gaussian_blur': da.gaussian_blur,
    'affine_transform': da.affine_transform,
    'translation_only': da.translation_only,
    'all_transforms': da.all_transforms
}

# 对每种数据增强方法进行可视化
for method_name, transform_fn in augmentation_methods.items():
    print(f"Visualizing augmentation method: {method_name}")
    augmented_images = {}

    # 收集每个类别（0-9）的一张图像
    for img, label in subset_mnist:
        if label not in augmented_images:
            augmented_img = transform_fn(img)
            augmented_images[label] = augmented_img
        if len(augmented_images) == 10:
            break

    # 绘图
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        ax = axes[i]
        if i in augmented_images:
            img_arr = to_numpy_for_plot(augmented_images[i])
            ax.imshow(img_arr, cmap='gray')
            ax.set_title(f"Label {i}")
        else:
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
        ax.axis('off')
    
    plt.suptitle(f"{method_name} Augmentation")

    # 保存图像
    save_path = os.path.join(save_dir, f"{method_name}_augmentation.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved visualization: {save_path}")
