import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import random

class DataAugmentation:
    def __init__(self, image_size=28):
        self.image_size = image_size

        # 1. 随机旋转 ±10 度
        self.random_rotation = transforms.RandomRotation(degrees=10)

        # 2. 高斯模糊（默认 kernel_size = 3）
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

        # 3. 仿射变换（平移、缩放、旋转、剪切）
        self.affine_transform = transforms.RandomAffine(
            degrees=10,      # 仿射中的旋转范围
            translate=(0.1, 0.1),  # 平移范围：图像宽高的10%
            scale=(0.9, 1.1),      # 缩放比例
            shear=10              # 剪切角度
        )

        # 4. 平移单独实现（仿射中已有，但可拆分独立使用）
        self.translation_only = transforms.RandomAffine(
            degrees=0,            # 无旋转
            translate=(0.2, 0.2), # 平移范围更大
            scale=None,
            shear=None
        )

        # 5. 所有增强组合在一起
        self.all_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            self.random_rotation,
            self.gaussian_blur,
            self.affine_transform,
            self.translation_only,
            transforms.ToTensor()
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
