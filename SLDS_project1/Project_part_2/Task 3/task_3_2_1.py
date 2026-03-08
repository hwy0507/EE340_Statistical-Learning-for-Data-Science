import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------
# 参数设置 & 路径
# ----------------------------
mnist_data_path = "/home/nkd/ouyangzl/Project/data/MNIST"
image_save_path = "/home/nkd/ouyangzl/Project/Task 3/images"
os.makedirs(image_save_path, exist_ok=True)

# ----------------------------
# 1. 加载 MNIST 数据集 (Fetch OpenML)
# ----------------------------
print("加载 MNIST 数据中...")
mnist = fetch_openml('mnist_784', data_home=mnist_data_path, version=1, as_frame=False)
X_raw, y_raw = mnist["data"], mnist["target"].astype(int)

# 为了加快运行速度，仅使用10000个样本（保证类别平衡）
X_sample, _, y_sample, _ = train_test_split(X_raw, y_raw, train_size=10000, stratify=y_raw, random_state=42)

# ----------------------------
# 2. 数据预处理：传统 ML与深度学习分别处理
# ----------------------------
# 传统 ML：使用StandardScaler，对扁平向量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_scaled, y_sample, test_size=0.2, random_state=42)

# 深度学习：归一化至[0,1]，恢复成图像形状：[N, 1, 28, 28]
X_dl = X_sample.astype(np.float32) / 255.0
X_dl = X_dl.reshape(-1, 28, 28)  # (N, 28, 28)
X_dl = np.expand_dims(X_dl, axis=1)  # (N, 1, 28, 28)
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_sample, test_size=0.2, random_state=42, stratify=y_sample)

# ----------------------------
# 3. 定义传统机器学习模型并评估
# ----------------------------
models_ml = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "SVM": SVC(kernel='rbf', random_state=0),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

ml_metrics = { "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [] }
ml_model_names = []

print("训练传统机器学习模型并计算指标...")
for name, model in models_ml.items():
    model.fit(X_train_ml, y_train_ml)
    y_pred = model.predict(X_test_ml)
    ml_model_names.append(name)
    ml_metrics["Accuracy"].append(accuracy_score(y_test_ml, y_pred))
    ml_metrics["Precision"].append(precision_score(y_test_ml, y_pred, average="macro", zero_division=0))
    ml_metrics["Recall"].append(recall_score(y_test_ml, y_pred, average="macro", zero_division=0))
    ml_metrics["F1 Score"].append(f1_score(y_test_ml, y_pred, average="macro", zero_division=0))
    print(f"{name} 完成")

# ----------------------------
# 4. 定义深度学习模型：CNN 和 MLP
# ----------------------------
# CNN 模型定义
class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers=2):
        super(SimpleCNN, self).__init__()
        # 根据num_conv_layers构建简单的卷积网络，以下示例只考虑增加卷积层后的Linear输入尺寸。
        # 固定采用卷积层: conv->relu, 最后池化，假设每层kernel_size=3, stride=1, 无padding
        layers = []
        in_channels = 1
        out_channels = 32
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            # 增加out_channels 每层翻倍
            out_channels *= 2
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)
        # 为简单起见，假设输入28x28经过conv后尺寸按公式计算粗略下来为 28 - 2*num_conv_layers
        conv_output_size = 28 - 2 * num_conv_layers  # 无padding, kernel=3
        conv_output_size = conv_output_size // 2  # after pooling
        conv_channels = in_channels  # final out_channels after conv layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_channels * conv_output_size * conv_output_size, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# MLP 模型定义
class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=256, num_hidden_layers=2):
        super(SimpleMLP, self).__init__()
        layers = []
        input_size = 28 * 28
        # 第一层
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        # 隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        # 输出层
        layers.append(nn.Linear(hidden_size, 10))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

# ----------------------------
# 5. 定义训练函数（适用于CNN和MLP）
# ----------------------------
def train_deep(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    device = next(model.parameters()).device
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)
        train_loss = total_loss / total
        train_acc = correct / total
        
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)
        test_loss = total_loss / total
        test_acc = correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
    return train_losses, test_losses, train_accs, test_accs

# ----------------------------
# 6. 准备深度学习的数据加载器
# ----------------------------
batch_size_dl = 128
X_train_tensor = torch.tensor(X_train_dl)
y_train_tensor = torch.tensor(y_train_dl, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test_dl)
y_test_tensor  = torch.tensor(y_test_dl, dtype=torch.long)

train_dataset_dl = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset_dl  = TensorDataset(X_test_tensor, y_test_tensor)

train_loader_dl = DataLoader(train_dataset_dl, batch_size=batch_size_dl, shuffle=True)
test_loader_dl  = DataLoader(test_dataset_dl, batch_size=batch_size_dl, shuffle=False)

# ----------------------------
# 7. 训练基础 CNN 和 MLP 模型
# ----------------------------
num_epochs_dl = 10
device_dl = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练CNN（基础配置）
cnn_model = SimpleCNN(num_conv_layers=2).to(device_dl)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
print("\n训练基础 CNN 模型...")
_, _, cnn_train_accs, cnn_test_accs = train_deep(cnn_model, train_loader_dl, test_loader_dl, criterion, cnn_optimizer, num_epochs=num_epochs_dl)

cnn_model.eval()
cnn_preds = []
with torch.no_grad():
    for X_batch, _ in test_loader_dl:
        outputs = cnn_model(X_batch.to(device_dl))
        preds = outputs.argmax(dim=1).cpu().numpy()
        cnn_preds.extend(preds)
cnn_acc = accuracy_score(y_test_dl, cnn_preds)
print(f"基础 CNN 测试 Accuracy: {cnn_acc:.4f}")

# 训练MLP（基础配置）
mlp_model = SimpleMLP(hidden_size=256, num_hidden_layers=2).to(device_dl)
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)
print("\n训练基础 MLP 模型...")
_, _, mlp_train_accs, mlp_test_accs = train_deep(mlp_model, train_loader_dl, test_loader_dl, criterion, mlp_optimizer, num_epochs=num_epochs_dl)

mlp_model.eval()
mlp_preds = []
with torch.no_grad():
    for X_batch, _ in test_loader_dl:
        outputs = mlp_model(X_batch.to(device_dl))
        preds = outputs.argmax(dim=1).cpu().numpy()
        mlp_preds.extend(preds)
mlp_acc = accuracy_score(y_test_dl, mlp_preds)
print(f"基础 MLP 测试 Accuracy: {mlp_acc:.4f}")

# ----------------------------
# 8. 超参数实验：调整超参数对 CNN 与 MLP 的影响
# ----------------------------

# (A) 实验1：CNN不同batch size对性能的影响
batch_sizes = [64, 128, 256, 512]
cnn_batch_acc = []
for bs in batch_sizes:
    print(f"\n训练CNN, Batch Size={bs}")
    train_loader_temp = DataLoader(train_dataset_dl, batch_size=bs, shuffle=True)
    test_loader_temp  = DataLoader(test_dataset_dl, batch_size=bs, shuffle=False)
    model_temp = SimpleCNN(num_conv_layers=2).to(device_dl)
    optimizer_temp = optim.Adam(model_temp.parameters(), lr=1e-3)
    _ , _ , _ , test_accs = train_deep(model_temp, train_loader_temp, test_loader_temp, criterion, optimizer_temp, num_epochs=5)
    cnn_batch_acc.append(test_accs[-1])
    
# (B) 实验2：MLP不同隐藏层神经元数量对性能的影响
hidden_sizes = [128, 256, 512]
mlp_hidden_acc = []
for hs in hidden_sizes:
    print(f"\n训练MLP, Hidden Size={hs}")
    model_temp = SimpleMLP(hidden_size=hs, num_hidden_layers=2).to(device_dl)
    optimizer_temp = optim.Adam(model_temp.parameters(), lr=1e-3)
    _ , _ , _ , test_accs = train_deep(model_temp, train_loader_dl, test_loader_dl, criterion, optimizer_temp, num_epochs=5)
    mlp_hidden_acc.append(test_accs[-1])

# (C) 实验3：CNN不同网络深度对性能的影响
# 这里不同的卷积层数量决定网络深度，例如2层、3层、4层
conv_layers_list = [2, 3, 4]
cnn_depth_acc = []
for num_layers in conv_layers_list:
    print(f"\n训练CNN, Conv Layers={num_layers}")
    model_temp = SimpleCNN(num_conv_layers=num_layers).to(device_dl)
    optimizer_temp = optim.Adam(model_temp.parameters(), lr=1e-3)
    _ , _ , _ , test_accs = train_deep(model_temp, train_loader_dl, test_loader_dl, criterion, optimizer_temp, num_epochs=5)
    cnn_depth_acc.append(test_accs[-1])

# ----------------------------
# 9. 绘图：超参数与性能对比
# ----------------------------
def plot_hyperparam(x_vals, y_vals, x_label, y_label, title, save_filename):
    plt.figure(figsize=(8,6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', markersize=8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0,1.1)
    plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(image_save_path, save_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"超参数图已保存: {save_path}")

# 绘制CNN Batch Size vs Accuracy
plot_hyperparam(batch_sizes, cnn_batch_acc, "Batch Size", "Test Accuracy", "CNN Batch Size vs Accuracy", "cnn_batch_vs_accuracy.png")

# 绘制MLP Hidden Size vs Accuracy
plot_hyperparam(hidden_sizes, mlp_hidden_acc, "Hidden Layer Size", "Test Accuracy", "MLP Hidden Size vs Accuracy", "mlp_hidden_vs_accuracy.png")

# 绘制CNN Network Depth (Conv Layers) vs Accuracy
plot_hyperparam(conv_layers_list, cnn_depth_acc, "Number of Conv Layers", "Test Accuracy", "CNN Network Depth vs Accuracy", "cnn_depth_vs_accuracy.png")

# ----------------------------
# 10. 综合对比：传统 ML、CNN 和 MLP 的性能指标
# ----------------------------
all_model_names = ml_model_names + ["CNN", "MLP"]
all_accuracy = ml_metrics["Accuracy"] + [cnn_acc, mlp_acc]
all_precision = ml_metrics["Precision"] + [None, None]  # 仅提供Accuracy对比
all_recall = ml_metrics["Recall"] + [None, None]
all_f1 = ml_metrics["F1 Score"] + [None, None]

color_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

# 定义一个新的plot_bar函数，每个方法有固定颜色
def plot_bar(metric_values, metric_name, save_filename, model_names, color_map):
    plt.figure(figsize=(12, 6))
    bars = []
    for i, name in enumerate(model_names):
        bar = plt.bar(name, metric_values[i], color=color_map.get(name, 'gray'))
        bars.append(bar)
    plt.title(f"{metric_name} Comparison on MNIST")
    plt.ylabel(metric_name)
    plt.ylim(0, 1.1)
    for bar in bars:
        # bar是一个container对象，通常只有一个bar
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height + 0.02, f"{height:.2f}",
                     ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(image_save_path, save_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"综合对比图已保存: {save_path}")

# 定义每个模型对应的颜色（确保包含所有模型）
color_map = {
    "Decision Tree": "#1f77b4",
    "Random Forest": "#ff7f0e",
    "SVM": "#2ca02c",
    "Logistic Regression": "#d62728",
    "KNN": "#9467bd",
    "CNN": "#e377c2",
    "MLP": "#8c564b"
}

# 绘制并保存各项指标的柱状图，每个柱子用对应的颜色
plot_bar(all_accuracy, "Accuracy", "comparison_accuracy.png", all_model_names, color_map)
plot_bar(all_precision, "Precision", "comparison_precision.png", all_model_names, color_map)
plot_bar(all_recall, "Recall", "comparison_recall.png", all_model_names, color_map)
plot_bar(all_f1, "F1 Score", "comparison_f1.png", all_model_names, color_map)

# ----------------------------
# 11. 讨论与分析 (注释说明)
# ----------------------------
"""
讨论：
1. 对于CNN，较小的batch size有时能带来更稳定的更新，但可能训练时间更长；而较大的batch size计算效率高，但可能导致泛化能力降低。
2. MLP中隐藏层神经元数量越多，模型表达能力越强，但过多的神经元可能导致过拟合，反而影响测试准确率。
3. CNN的网络深度（卷积层数量）增加可以提取更多的特征层次，但过深的网络在数据量有限的情况下可能导致过拟合或梯度消失问题。
4. 综合对比时，深度学习方法（特别是CNN）通常可以比传统机器学习模型在图像任务上取得更好的表现，但对超参数和网络结构更为敏感。
"""
print("所有实验已完成！")