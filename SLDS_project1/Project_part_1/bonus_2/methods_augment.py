import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

# ----------------------------
# 参数设置 & 路径
# ----------------------------
ddr_data_path = "/home/ouyangzl/Project/Project/data/DDR/DDR-dataset/DR_grading~"
image_save_path = "/home/ouyangzl/Project/Project/bonus_2/images"
os.makedirs(image_save_path, exist_ok=True)

# ----------------------------
# 1. 定义DDR数据集
# ----------------------------
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

# ----------------------------
# 2. 数据预处理
# ----------------------------
scaler = StandardScaler()

# ----------------------------
# 3. 定义传统机器学习模型
# ----------------------------
models_ml = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "SVM": SVC(kernel='rbf', random_state=0),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

# ----------------------------
# 4. 定义深度学习模型
# ----------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ----------------------------
# 5. 定义训练和评估函数
# ----------------------------
def train_evaluate_ml(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    return acc, prec, rec, f1

def train_evaluate_dl(model, train_loader, test_loader, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    # 评估
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, prec, rec, f1

# ----------------------------
# 6. 定义不同数据集
# ----------------------------
SAMPLES_PER_CLASS = 800
NUM_CLASSES = 6
TOTAL_SAMPLES = SAMPLES_PER_CLASS * NUM_CLASSES

# ----------------------------
# 7. 加载数据集
# ----------------------------
# 定义不同的数据增强方式
data_transforms = {
    'no_augmentation': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]),
    'horizontal_flip': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]),
    'random_rotation': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
    ]),
    'color_jitter': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ]),
    'affine_transform': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor()
    ]),
    'all_transforms': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor()
    ])
}

# 创建数据集
datasets = {}
dataloaders = {}
for aug_type, transform in data_transforms.items():
    datasets[aug_type] = DDRDataset(os.path.join(ddr_data_path, "train"),
                                      os.path.join(ddr_data_path, "train.txt"),
                                      transforms=transform,
                                      samples_per_class=SAMPLES_PER_CLASS)
    dataloaders[aug_type] = DataLoader(datasets[aug_type], batch_size=32, shuffle=True)

test_dataset = DDRDataset(os.path.join(ddr_data_path, "test"),
                            os.path.join(ddr_data_path, "test.txt"),
                            transforms=data_transforms['no_augmentation'], # 测试集不增强
                            samples_per_class=SAMPLES_PER_CLASS)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 8. 准备数据加载器
# ----------------------------
# 已经包含在上面的循环中

# ----------------------------
# 9. 训练和评估模型
# ----------------------------
all_results = {metric: {aug_type: [] for aug_type in data_transforms} for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for aug_type, train_loader in dataloaders.items():
    print(f"Training and evaluating CNN with {aug_type}...")
    cnn_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    acc, prec, rec, f1 = train_evaluate_dl(cnn_model, train_loader, test_loader, device=device)
    all_results["Accuracy"][aug_type].append(acc)
    all_results["Precision"][aug_type].append(prec)
    all_results["Recall"][aug_type].append(rec)
    all_results["F1 Score"][aug_type].append(f1)
    print(f"{aug_type} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# ----------------------------
# 10. 绘制柱状图
# ----------------------------
color_map = {
    "no_augmentation": "#1f77b4",
    "horizontal_flip": "#ff7f0e",
    "random_rotation": "#2ca02c",
    "color_jitter": "#d62728",
    "affine_transform": "#9467bd",
    "all_transforms": "#8c564b"
}

aug_types = list(data_transforms.keys())

for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(aug_types))
    y = [all_results[metric][aug_type][0] for aug_type in aug_types]
    plt.bar(x, y, color=[color_map.get(aug_type, "#333333") for aug_type in aug_types])
    plt.xticks(x, aug_types, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison on DDR with Different Augmentations")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    save_path = os.path.join(image_save_path, f"ddr_aug_{metric.lower().replace(' ', '_')}_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

print("✅ 所有实验已完成！")