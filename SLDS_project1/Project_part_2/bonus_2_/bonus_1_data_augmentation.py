import os
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random

# ----------------------------------------------------------------------
# 1. 数据集定义
# ----------------------------------------------------------------------
class DDRDataset(Dataset):
    def __init__(self, root_dir, list_file, transforms=None, samples_per_class=800):
        """
        Args:
            root_dir (str): 图像所在目录，如 ".../DR_grading/train"
            list_file (str): 列表文件路径，如 ".../DR_grading/train.txt"
            transforms (callable, optional): 图像预处理／增强操作
            samples_per_class (int): 每类样本的数量
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.samples_per_class = samples_per_class
        
        # 读取列表文件，解析 (filename, label)
        self.all_samples = []
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname, label = line.split()
                self.all_samples.append((fname, int(label)))

        # 按照类别分组
        grouped_samples = [[] for _ in range(6)]  # 假设有6个类别
        for fname, label in self.all_samples:
            grouped_samples[label].append((fname, label))

        # 对每个类别随机抽样
        self.samples = []
        for group in grouped_samples:
            self.samples.extend(random.sample(group, min(len(group), samples_per_class)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, fname)
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, label

# ----------------------------------------------------------------------
# 2. 数据增强定义
# ----------------------------------------------------------------------
class DataAugmentation:
    def __init__(self, image_size=64):
        self.image_size = image_size

        # 1. 随机旋转 ±10 度
        self.random_rotation = T.RandomRotation(degrees=10)

        # 2. 高斯模糊（默认 kernel_size = 3）
        self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

        # 3. 仿射变换（平移、缩放、旋转、剪切）
        self.affine_transform = T.RandomAffine(
            degrees=10,      # 仿射中的旋转范围
            translate=(0.1, 0.1),  # 平移范围：图像宽高的10%
            scale=(0.9, 1.1),      # 缩放比例
            shear=10              # 剪切角度
        )

        # 4. 平移单独实现（仿射中已有，但可拆分独立使用）
        self.translation_only = T.RandomAffine(
            degrees=0,            # 无旋转
            translate=(0.2, 0.2), # 平移范围更大
            scale=None,
            shear=None
        )

        # 5. 所有增强组合在一起
        self.all_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=10),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            T.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=None,
                shear=None
            ),
            T.ToTensor()
        ])

    def apply_rotation(self, img: Image.Image):
        return self.random_rotation(img)

    def apply_blur(self, img: Image.Image):
        return self.gaussian_blur(img)

    def apply_affine(self, img: Image.Image):
        return self.affine_transform(img)

    def apply_translation(self, img: Image.Image):
        return self.translation_only(img)

    def apply_all(self, img: Image.Image):
        return self.all_transforms(img)

# ----------------------------------------------------------------------
# 3. 辅助函数
# ----------------------------------------------------------------------
def to_numpy_flat(img):
    if isinstance(img, torch.Tensor):
        arr = img.squeeze().numpy()
    else:
        arr = np.array(img)
    return arr.flatten()  # 展平为向量

def no_augmentation(img):
    return img

# ----------------------------------------------------------------------
# 4. 参数设置
# ----------------------------------------------------------------------
DATA_ROOT = '/home/nkd/ouyangzl/Project/data/DDR/DDR-dataset/DR_grading'
SAVE_DIR  = '/home/nkd/ouyangzl/Project/bonus_2/images'
os.makedirs(SAVE_DIR, exist_ok=True)
SAMPLES_PER_CLASS = 800
NUM_CLASSES = 6
TOTAL_SAMPLES = SAMPLES_PER_CLASS * NUM_CLASSES

# ----------------------------------------------------------------------
# 5. 数据集加载
# ----------------------------------------------------------------------
# 5.1 定义 transforms
base_transforms = T.Compose([
    T.Resize((64, 64)),  # 统一尺寸
    T.ToTensor()          # 转 Tensor
])

# 5.2 加载数据集
train_ds = DDRDataset(os.path.join(DATA_ROOT, "train"),
                      os.path.join(DATA_ROOT, "train.txt"),
                      transforms=base_transforms,
                      samples_per_class=SAMPLES_PER_CLASS)
valid_ds = DDRDataset(os.path.join(DATA_ROOT, "valid"),
                      os.path.join(DATA_ROOT, "valid.txt"),
                      transforms=base_transforms,
                      samples_per_class=SAMPLES_PER_CLASS)
test_ds  = DDRDataset(os.path.join(DATA_ROOT, "test"),
                      os.path.join(DATA_ROOT, "test.txt"),
                      transforms=base_transforms,
                      samples_per_class=SAMPLES_PER_CLASS)

# ----------------------------------------------------------------------
# 6. 数据增强方法
# ----------------------------------------------------------------------
da = DataAugmentation(image_size=64)
methods = {
    '0_no_augmentation':   no_augmentation,
    '1_random_rotation':   da.apply_rotation,
    '2_gaussian_blur':     da.apply_blur,
    '3_affine_transform':  da.apply_affine,
    '4_translation_only':  da.apply_translation,
    '5_all_transforms':    da.apply_all,
}

# ----------------------------------------------------------------------
# 7. 可视化
# ----------------------------------------------------------------------
datasets_ = {"train": train_ds, "valid": valid_ds, "test": test_ds}

for dataset_name, dataset in datasets_.items():
    for aug_name, aug_fn in methods.items():
        print(f"Processing: {dataset_name} + {aug_name}")
        
        # 提取特征
        features, labels = [], []
        for i in range(len(dataset)):
            img, label = dataset[i]
            
            # 应用数据增强
            img = T.ToPILImage()(img)  # 转换为 PIL Image

            if aug_name != '0_no_augmentation':
                aug_img = aug_fn(img)  # 应用数据增强
            else:
                aug_img = img  # 不应用数据增强

            vec = to_numpy_flat(aug_img)  # 转换为 numpy 数组并展平
            features.append(vec)
            labels.append(label)

        features = np.stack(features)
        labels = np.array(labels)

        # 降维
        tsne = TSNE(n_components=2, random_state=0)
        tsne_result = tsne.fit_transform(features)

        # 绘图
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', s=5)
        plt.title(f"{dataset_name} + {aug_name} (TSNE)")
        plt.xlabel("TSNE Dimension 1")
        plt.ylabel("TSNE Dimension 2")
        plt.colorbar(label="Class")
        plt.clim(0, NUM_CLASSES - 1)  # 确保颜色条范围正确

        # 保存
        filename = f"{dataset_name}_{aug_name}_tsne.png"
        filepath = os.path.join(SAVE_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved: {filepath}")

print("✅ 所有可视化完成！")