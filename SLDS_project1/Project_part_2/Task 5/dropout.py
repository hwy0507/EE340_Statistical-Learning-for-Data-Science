"""
本代码实现：
1. 加载 MNIST 数据集，并将数据分为训练集(10000个样本，其中拆分为8000训练 & 2000验证) 和测试集(2000个样本)。
2. 定义带 Dropout 的 SimpleMLP 和 SimpleCNN 模型，Dropout率作为参数传入，可在隐藏层使用。
3. 分别针对不同 Dropout 率（例如：0.0, 0.1, 0.3, 0.5, 0.7）训练模型，记录在验证集和测试集上的性能指标（Accuracy, Precision, Recall, F1 Score）。
4. 绘制不同指标对于 Dropout 率的折线图，以观察 Dropout 率对模型性能的影响。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------
# 参数设置 & 路径
# -------------------------------
mnist_data_path = "/home/nkd/ouyangzl/Project/data/MNIST"
image_save_path = "/home/nkd/ouyangzl/Project/Task 5/images"
os.makedirs(image_save_path, exist_ok=True)

# -------------------------------
# 1. 加载 MNIST 数据集
# -------------------------------
print("加载 MNIST 数据中...")
mnist = fetch_openml('mnist_784', data_home=mnist_data_path, version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# 取总共12000个样本：训练+验证 10000个，测试 2000个
X_sample, X_test, y_sample, y_test = train_test_split(X, y, train_size=10000, test_size=2000,
                                                       stratify=y, random_state=42)

# -------------------------------
# 2. 数据预处理
# 对于深度学习，归一化并转换为图像格式 [N, 1, 28, 28]
X_sample = X_sample.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_sample = X_sample.reshape(-1, 28, 28)
X_sample = np.expand_dims(X_sample, axis=1)  # (10000, 1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)
X_test = np.expand_dims(X_test, axis=1)        # (2000, 1, 28, 28)

# 将10000个样本拆分为 8000训练 和 2000验证
X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=2000,
                                                   random_state=42, stratify=y_sample)

# -------------------------------
# 3. 定义带 Dropout 的深度学习模型：SimpleMLP_Dropout 和 SimpleCNN_Dropout
# -------------------------------
class SimpleMLP_Dropout(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, num_layers=2, dropout_rate=0.5):
        super(SimpleMLP_Dropout, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        # x: [B, 1, 28, 28] -> flatten to [B, 784]
        x = x.view(x.size(0), -1)
        return self.net(x)

class SimpleCNN_Dropout(nn.Module):
    def __init__(self, num_classes=10, channels=32, depth=2, dropout_rate=0.5):
        super(SimpleCNN_Dropout, self).__init__()
        conv_layers = []
        in_channels = 1
        for i in range(depth):
            conv_layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            # 为防止过拟合，可在每个卷积块后加入 Dropout
            conv_layers.append(nn.Dropout2d(dropout_rate))
            in_channels = channels
        conv_layers.append(nn.AdaptiveAvgPool2d((7, 7)))
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# -------------------------------
# 4. 定义带 Dropout 的深度学习训练函数
# -------------------------------
def train_dl_model_dropout(model, X_train, y_train, X_val, y_val, X_test, y_test,
                           epochs=10, lr=1e-3, batch_size=128, is_cnn=False, device='cpu'):
    model = model.to(device)
    if is_cnn:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_val = X_val.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
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
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    # 评估函数
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
# 5. 定义 Dropout 率列表（实验不同 dropout 率）
# -------------------------------
dropout_values = [0.0, 0.1, 0.3, 0.5, 0.7]

# -------------------------------
# 6. 训练并记录不同 dropout 率下的模型性能
# 我们分别针对 MLP 和 CNN 模型进行实验，记录验证集及测试集指标
# -------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 结果字典，结构：
# results[model_type][dropout_value] = (val_metrics, test_metrics)
dropout_results = {"MLP": {}, "CNN": {}}

print("\n开始 Dropout 实验...")

# 针对 MLP 模型（使用 SimpleMLP_Dropout）
for dp in dropout_values:
    print(f"训练 MLP, Dropout rate = {dp}")
    model = SimpleMLP_Dropout(dropout_rate=dp)
    val_metrics, test_metrics = train_dl_model_dropout(model,
                                                       X_train, y_train,
                                                       X_val, y_val,
                                                       X_test, y_test,
                                                       epochs=10, lr=1e-3,
                                                       batch_size=128, is_cnn=False, device=device)
    dropout_results["MLP"][dp] = (val_metrics, test_metrics)

# 针对 CNN 模型（使用 SimpleCNN_Dropout）
for dp in dropout_values:
    print(f"训练 CNN, Dropout rate = {dp}")
    model = SimpleCNN_Dropout(dropout_rate=dp)
    val_metrics, test_metrics = train_dl_model_dropout(model,
                                                       X_train, y_train,
                                                       X_val, y_val,
                                                       X_test, y_test,
                                                       epochs=10, lr=1e-3,
                                                       batch_size=128, is_cnn=True, device=device)
    dropout_results["CNN"][dp] = (val_metrics, test_metrics)

# -------------------------------
# 7. 绘制性能随 Dropout 率变化的折线图
# 横轴：Dropout 率；纵轴：指标得分；分曲线展示验证集与测试集表现；
# 对于每个模型分别绘制 Accuracy, Precision, Recall, F1 Score 折线图
# -------------------------------
def plot_dropout_curve(model_type, metric_idx, metric_name, dropout_values, results, save_path):
    # metric_idx: 0-accuracy, 1-precision, 2-recall, 3-f1
    val_scores = []
    test_scores = []
    for dp in dropout_values:
        val, test = results[model_type][dp]
        val_scores.append(val[metric_idx])
        test_scores.append(test[metric_idx])
    plt.figure(figsize=(8,6))
    plt.plot(dropout_values, val_scores, marker='o', linestyle='-', label='Validation', color='blue')
    plt.plot(dropout_values, test_scores, marker='o', linestyle='--', label='Test', color='red')
    plt.xlabel('Dropout Rate')
    plt.ylabel(metric_name)
    plt.title(f"{model_type} Dropout: {metric_name} vs Dropout Rate")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"图保存至: {save_path}")

# 对于 MLP 和 CNN，分别绘制各指标折线图
for model_type in ["MLP", "CNN"]:
    for idx, metric in enumerate(["Accuracy", "Precision", "Recall", "F1 Score"]):
        fname = f"{model_type.lower()}_dropout_{metric.lower().replace(' ', '_')}_curve.png"
        save_path = os.path.join(image_save_path, fname)
        plot_dropout_curve(model_type, idx, metric, dropout_values, dropout_results, save_path)

# -------------------------------
# 8. 分析注释
# -------------------------------
"""
分析说明：
1. Dropout 技术通过在训练时随机将部分神经元的输出置为 0，减少模型过拟合现象，提高泛化能力。
2. 随着 Dropout 率的提高，可能会导致欠拟合，模型性能下降；而过低的 Dropout 率又可能不足以缓解过拟合问题。
3. 对于 MLP 和 CNN 模型，通过实验可以观察到在一定范围内（例如 0.1~0.5）的 Dropout 率使模型在验证集和测试集上的指标更优。
4. 不同模型对 Dropout 的敏感程度可能不同，实验结果图能直观展示各项指标（Accuracy, Precision, Recall, F1 Score）随 Dropout 率的变化趋势。
"""
print("✅ 所有 Dropout 实验已完成！")