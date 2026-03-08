"""
本代码实现：
1. 加载 DDR 眼底彩照图像数据集，并将数据筛选为训练集（10000个样本，其中拆分为8000训练 & 2000验证）和测试集（2000个样本）。
2. 对图像进行预处理：调整为固定尺寸（64×64）、归一化；对于传统机器学习，将图像平铺为向量（3×64×64=12288维）。
3. 采用与 MNIST 实验中相同的传统机器学习模型（决策树、随机森林、SVM、逻辑回归、KNN）及深度学习模型（简单 MLP、简单 CNN），
   进行训练与评估，计算 Accuracy、Precision、Recall、F1 Score。
4. 观察 DDR 数据集上的性能指标与 MNIST 是否存在明显差异。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms

# -------------------------------
# 数据路径和图片保存路径
# -------------------------------
ddr_data_path = "/home/nkd/ouyangzl/Project/data/DDR"   # DDR 数据集文件夹，要求图像按类别存放在不同子目录中
image_save_path = "/home/nkd/ouyangzl/Project/Bonus_2/images"
os.makedirs(image_save_path, exist_ok=True)

# -------------------------------
# 1. 加载 DDR 数据集
# -------------------------------
print("加载 DDR 数据集中...")
# 定义预处理：调整图像尺寸为 64×64，转换为 tensor，归一化到 [0,1]
ddr_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
ddr_dataset = ImageFolder(root=ddr_data_path, transform=ddr_transform)

# 遍历 DDR 数据集，将所有图像和标签加载为 numpy 数组
images = []
labels = []
for img, label in ddr_dataset:
    images.append(img.numpy())  # img shape: (C, H, W) ，这里 C=3
    labels.append(label)
images = np.array(images)  # shape: (N, 3, 64, 64)
labels = np.array(labels)

# 为保证样本数，随机抽取 12000 个样本（假设数据集中至少有12000张图像）
# 分层抽样：训练集 10000，测试集 2000
X_full, X_test, y_full, y_test = train_test_split(
    images, labels, train_size=10000, test_size=2000, stratify=labels, random_state=42
)
# 将 10000 个样本划分为 8000 训练和 2000 验证
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=2000, stratify=y_full, random_state=42
)

# 为传统 ML 模型准备扁平化数据，图像通道数为 3，尺寸 64×64 -> 3*64*64 = 12288
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# -------------------------------
# 2. 训练传统机器学习模型
# -------------------------------
def train_ml_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    return acc, prec, rec, f1

# 定义传统 ML 模型字典
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "SVM": SVC(kernel='rbf', probability=False, random_state=0),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

ml_metrics = { "Accuracy": {}, "Precision": {}, "Recall": {}, "F1 Score": {} }
print("\n训练传统机器学习模型...")
for name, model in models.items():
    acc, prec, rec, f1 = train_ml_model(model, X_train_flat, y_train, X_test_flat, y_test)
    ml_metrics["Accuracy"][name] = acc
    ml_metrics["Precision"][name] = prec
    ml_metrics["Recall"][name] = rec
    ml_metrics["F1 Score"][name] = f1
    print(f"{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

# -------------------------------
# 3. 定义深度学习模型：SimpleMLP 和 SimpleCNN
# -------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=12288, hidden_dim=128, num_classes= len(ddr_dataset.classes), num_layers=2):
        super(SimpleMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        # x: [B, 3, 64, 64] -> flatten to [B, 12288]
        x = x.view(x.size(0), -1)
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes= len(ddr_dataset.classes), channels=32, depth=2):
        super(SimpleCNN, self).__init__()
        conv_layers = []
        in_channels = 3  # DDR 图像为彩色，通道数=3
        for i in range(depth):
            conv_layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            in_channels = channels
        # 使用自适应池化使输出固定为 7x7
        conv_layers.append(nn.AdaptiveAvgPool2d((7, 7)))
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# -------------------------------
# 4. 训练深度学习模型
# -------------------------------
def train_dl_model(model, X_train, y_train, X_test, y_test, epochs=10, lr=1e-3, batch_size=128, is_cnn=False, device='cpu'):
    model = model.to(device)
    # 若为 CNN, 数据形状保持; 若为 MLP, 数据继续以 [B, 3, 64, 64] 接收，然后在 forward flatten
    if is_cnn == False:
        # 对于 MLP，直接使用原始图像数据，后续在 forward 中 flatten
        pass
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
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
            outputs = model(xb)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, prec, rec, f1

# -------------------------------
# 5. 训练并记录深度学习模型性能（MLP 与 CNN）
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dl_metrics = { "Accuracy": {}, "Precision": {}, "Recall": {}, "F1 Score": {} }

print("\n训练深度学习模型...")

# 训练 SimpleMLP，使用原始 DDR 彩照（shape: [N,3,64,64])
mlp_model = SimpleMLP()
mlp_acc, mlp_prec, mlp_rec, mlp_f1 = train_dl_model(mlp_model, X_train, y_train, X_test, y_test,
                                                     epochs=10, lr=1e-3, batch_size=128, is_cnn=False, device=device)
dl_metrics["Accuracy"]["MLP"] = mlp_acc
dl_metrics["Precision"]["MLP"] = mlp_prec
dl_metrics["Recall"]["MLP"] = mlp_rec
dl_metrics["F1 Score"]["MLP"] = mlp_f1
print(f"SimpleMLP: Acc={mlp_acc:.4f}, Prec={mlp_prec:.4f}, Rec={mlp_rec:.4f}, F1={mlp_f1:.4f}")

# 训练 SimpleCNN，is_cnn=True
cnn_model = SimpleCNN()
cnn_acc, cnn_prec, cnn_rec, cnn_f1 = train_dl_model(cnn_model, X_train, y_train, X_test, y_test,
                                                     epochs=10, lr=1e-3, batch_size=128, is_cnn=True, device=device)
dl_metrics["Accuracy"]["CNN"] = cnn_acc
dl_metrics["Precision"]["CNN"] = cnn_prec
dl_metrics["Recall"]["CNN"] = cnn_rec
dl_metrics["F1 Score"]["CNN"] = cnn_f1
print(f"SimpleCNN: Acc={cnn_acc:.4f}, Prec={cnn_prec:.4f}, Rec={cnn_rec:.4f}, F1={cnn_f1:.4f}")

# -------------------------------
# 6. 绘制传统 ML 与 深度学习模型的性能指标对比柱状图
# -------------------------------
# 合并模型名称及各指标
all_model_names = list(models.keys()) + ["MLP", "CNN"]
all_accuracy = [ ml_metrics["Accuracy"][m] for m in models.keys() ] + [mlp_acc, cnn_acc]
all_precision = [ ml_metrics["Precision"][m] for m in models.keys() ] + [mlp_prec, cnn_prec]
all_recall = [ ml_metrics["Recall"][m] for m in models.keys() ] + [mlp_rec, cnn_rec]
all_f1 = [ ml_metrics["F1 Score"][m] for m in models.keys() ] + [mlp_f1, cnn_f1]

color_map = {
    "Decision Tree": "#1f77b4",
    "Random Forest": "#ff7f0e",
    "SVM": "#2ca02c",
    "Logistic Regression": "#d62728",
    "KNN": "#9467bd",
    "MLP": "#8c564b",
    "CNN": "#e377c2"
}

def plot_bar(metric_values, metric_name, save_filename, model_names, color_map):
    plt.figure(figsize=(12, 6))
    bars = []
    for i, name in enumerate(model_names):
        bar = plt.bar(name, metric_values[i], color=color_map.get(name, 'gray'))
        bars.append(bar)
    plt.title(f"{metric_name} Comparison on DDR")
    plt.ylabel(metric_name)
    plt.ylim(0, 1.1)
    for bar in bars:
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height + 0.01, f"{height:.2f}",
                     ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(image_save_path, save_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"图保存至: {save_path}")

plot_bar(all_accuracy, "Accuracy", "ddr_comparison_accuracy.png", all_model_names, color_map)
plot_bar(all_precision, "Precision", "ddr_comparison_precision.png", all_model_names, color_map)
plot_bar(all_recall, "Recall", "ddr_comparison_recall.png", all_model_names, color_map)
plot_bar(all_f1, "F1 Score", "ddr_comparison_f1.png", all_model_names, color_map)

print("✅ DDR 实验已完成！")