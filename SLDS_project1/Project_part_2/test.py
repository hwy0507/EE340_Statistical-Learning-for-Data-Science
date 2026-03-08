# 测试 get_balanced_subset 函数
from torchvision.datasets import MNIST
from torchvision import transforms

DATA_DIR = "/home/nkd/ouyangzl/Project/data"
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(DATA_DIR, transform=transform, train=True, download=True)

# 测试样本数量大于某些类别的样本数量
subset = get_balanced_subset(train_dataset, num_samples_per_class=6000)
print(f"Subset size: {len(subset)}")