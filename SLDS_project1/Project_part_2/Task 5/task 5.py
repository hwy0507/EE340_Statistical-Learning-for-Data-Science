"""
本代码实现：
1. 加载 MNIST 数据集，并将数据分为训练集(10000个样本，其中进一步拆分为8000训练 & 2000验证) 和测试集(2000个样本)。
2. 定义简单的 CNN 和 MLP 模型。
3. 在训练过程中在损失函数中添加 L1 或 L2 正则化项，正则化参数 reg_lambda 递增。
4. 分别记录在验证集和测试集上的性能指标（Accuracy, Precision, Recall, F1 Score），绘制正则化强度与模型性能的折线图，比较不同正则化强度下模型性能的变化。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------
# 参数设置 & 路径
# -------------------------------
mnist_data_path = "/home/nkd/ouyangzl/Project/data/MNIST"
image_save_path = "/home/nkd/ouyangzl/Project/Task 3/images"
os.makedirs(image_save_path, exist_ok=True)

# -------------------------------
# 1. 加载 MNIST 数据集
# -------------------------------
print("加载 MNIST 数据中...")
mnist = fetch_openml('mnist_784', data_home=mnist_data_path, version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# 为加快运行，仅取总共12000个样本（训练+验证:10000，测试:2000）
X_sample, X_test, y_sample, y_test = train_test_split(X, y, train_size=10000, test_size=2000,
                                                       stratify=y, random_state=42)

# -------------------------------
# 2. 数据预处理
# 对于深度学习，归一化并转换为图像格式 [N, 1, 28, 28]
X_sample = X_sample.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_sample = X_sample.reshape(-1,28,28)  # shape (10000, 28, 28)
X_sample = np.expand_dims(X_sample, axis=1)  # shape (10000, 1, 28, 28)
X_test = X_test.reshape(-1,28,28)
X_test = np.expand_dims(X_test, axis=1)  # (2000, 1, 28, 28)

# 将10000样本中的一部分作为验证集：划分 8000 训练, 2000 验证
X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=2000, random_state=42, stratify=y_sample)

# -------------------------------
# 3. 定义深度学习模型：SimpleMLP 和 SimpleCNN
# -------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, num_layers=2):
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
        # x: [B, 1, 28, 28] -> flatten to [B, 784]
        x = x.view(x.size(0), -1)
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, channels=32, depth=2):
        super(SimpleCNN, self).__init__()
        conv_layers = []
        in_channels = 1
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
# 4. 定义带正则化的深度学习训练函数
# -------------------------------
def train_dl_model_reg(model, X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs=10, lr=1e-3, batch_size=128, is_cnn=False,
                       device='cpu', reg_type=None, reg_lambda=0.0):
    """
    reg_type: None, 'l1', or 'l2'
    reg_lambda: 正则化强度
    """
    model = model.to(device)
    # 若是CNN，则确保数据形状为 [B, 1, 28, 28]
    if is_cnn:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_val = X_val.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
    # 构建数据加载器
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练过程
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            # 加入正则化项
            if reg_type == 'l1':
                l1_loss = 0.
                for param in model.parameters():
                    l1_loss += torch.sum(torch.abs(param))
                loss = loss + reg_lambda * l1_loss
            elif reg_type == 'l2':
                l2_loss = 0.
                for param in model.parameters():
                    l2_loss += torch.sum(param ** 2)
                loss = loss + reg_lambda * l2_loss
            loss.backward()
            optimizer.step()
    # 定义评估函数
    def evaluate(loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in loader:
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

    val_metrics = evaluate(val_loader)
    test_metrics = evaluate(test_loader)
    return val_metrics, test_metrics

# -------------------------------
# 5. 定义正则化参数列表（不同正则化强度）
# -------------------------------
reg_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]

# -------------------------------
# 6. 训练并记录不同正则化强度下的模型性能
# 分别针对 L1 和 L2 正则化，对 CNN 与 MLP 进行实验
# 使用训练集：8000，验证集：2000，测试集：2000
# -------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义存储结果的字典，其结构：
# results[model_type][reg_type] = { reg_value: (val_metrics, test_metrics) }
results = {
    "MLP": {"l1": {}, "l2": {}},
    "CNN": {"l1": {}, "l2": {}}
}

print("\n开始正则化实验...")

# 针对 MLP 模型
for reg in reg_values:
    for rtype in ['l1', 'l2']:
        print(f"训练 MLP, 正则化类型: {rtype}, reg_lambda = {reg}")
        model = SimpleMLP()
        # 训练并返回 (val_metrics, test_metrics)
        val_metrics, test_metrics = train_dl_model_reg(model,
            X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=10, lr=1e-3, batch_size=128, is_cnn=False, device=device,
            reg_type=rtype, reg_lambda=reg)
        results["MLP"][rtype][reg] = (val_metrics, test_metrics)
        
# 针对 CNN 模型
for reg in reg_values:
    for rtype in ['l1', 'l2']:
        print(f"训练 CNN, 正则化类型: {rtype}, reg_lambda = {reg}")
        model = SimpleCNN()
        val_metrics, test_metrics = train_dl_model_reg(model,
            X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=10, lr=1e-3, batch_size=128, is_cnn=True, device=device,
            reg_type=rtype, reg_lambda=reg)
        results["CNN"][rtype][reg] = (val_metrics, test_metrics)

# -------------------------------
# 7. 绘制性能随正则化参数变化的折线图
# 横轴：正则化参数；纵轴：指标得分；分曲线展示验证集与测试集表现；分别对 L1 和 L2，MLP 和 CNN 绘图
# -------------------------------
def plot_reg_curve(model_type, reg_type, metric_idx, metric_name, reg_values, results, save_path):
    # metric_idx: 0-accuracy,1-precision,2-recall,3-f1
    val_scores = []
    test_scores = []
    for reg in reg_values:
        val, test = results[model_type][reg_type][reg]
        val_scores.append(val[metric_idx])
        test_scores.append(test[metric_idx])
    plt.figure(figsize=(8,6))
    plt.plot(reg_values, val_scores, marker='o', linestyle='-', label='Validation', color='blue')
    plt.plot(reg_values, test_scores, marker='o', linestyle='--', label='Test', color='red')
    plt.xscale('log')  # 正则化参数以对数刻度显示
    plt.xlabel('Regularization Strength (lambda)')
    plt.ylabel(metric_name)
    plt.title(f"{model_type} {reg_type.upper()} Regularization: {metric_name} vs Reg Strength")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"图保存至: {save_path}")

# 对于每个模型和每种正则化（L1和L2），绘制 Accuracy, Precision, Recall, F1 的折线图
for model_type in ["MLP", "CNN"]:
    for rtype in ["l1", "l2"]:
        for idx, metric in enumerate(["Accuracy", "Precision", "Recall", "F1 Score"]):
            fname = f"{model_type}_{rtype}_{metric.lower().replace(' ', '_')}_reg_curve.png"
            save_path = os.path.join(image_save_path, fname)
            plot_reg_curve(model_type, rtype, idx, metric, reg_values, results, save_path)

# -------------------------------
# 8. 分析注释
# -------------------------------
"""
分析说明：
1. 在深度学习模型（MLP和CNN）中引入L1/L2正则化，可控制模型复杂度，防止权重过大，改善模型泛化能力。
2. 随着正则化强度的增加（reg_lambda从0到1e-2），模型可能会因过于约束而欠拟合，从而导致验证集和测试集性能下降；
   而适度的正则化可能会提升性能。
3. 不同正则化类型对模型参数的约束不同：L1正则化引入稀疏性，有助于特征选择；L2正则化则倾向于平滑权重分布。
4. 实验结果图中可以观察到，在某个合适的正则化参数下，模型在验证集和测试集上指标达到较优。
"""
print("✅ 所有正则化实验已完成！")