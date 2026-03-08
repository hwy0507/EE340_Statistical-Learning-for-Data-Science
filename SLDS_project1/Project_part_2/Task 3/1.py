import os
import matplotlib.pyplot as plt
import numpy as np
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
from torch.utils.data import TensorDataset, DataLoader

# 数据路径和图片保存路径
mnist_data_path = "/home/nkd/ouyangzl/Project/data/MNIST"
image_save_path = "/home/nkd/ouyangzl/Project/Task 3/images"
os.makedirs(image_save_path, exist_ok=True)

# 1. 加载 MNIST 数据集
print("加载 MNIST 数据中...")
mnist = fetch_openml('mnist_784', data_home=mnist_data_path, version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# 为了加快运行速度，可以仅使用部分样本
X, _, y, _ = train_test_split(X, y, train_size=60000, stratify=y, random_state=42)

# 2. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. 定义模型
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "SVM": SVC(kernel='rbf', probability=False, random_state=0),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

# 训练传统机器学习模型
def train_ml_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    return acc, prec, rec, f1

# 训练深度学习模型
def train_dl_model(model, X_train, y_train, X_test, y_test, epochs=10, lr=1e-3, batch_size=128, is_cnn=False, device='cpu'):
    model = model.to(device)
    if is_cnn:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
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
            logits = model(xb)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, prec, rec, f1

# 多模型多指标绘图
def plot_metrics_bar_line(metrics_dict, model_names, image_save_path):
    color_map = {
        "Decision Tree": "#1f77b4",
        "Random Forest": "#ff7f0e",
        "SVM": "#2ca02c",
        "Logistic Regression": "#d62728",
        "KNN": "#9467bd",
        "MLP": "#8c564b",
        "CNN": "#e377c2"
    }
    # 柱状图
    for metric_name in metrics_dict:
        values = [metrics_dict[metric_name][name] for name in model_names]
        colors = [color_map.get(name, "#333333") for name in model_names]
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, values, color=colors)
        plt.title(f"{metric_name} Comparison on MNIST (Sampled)")
        plt.ylabel(metric_name)
        plt.ylim(0, 1.1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
        save_path = os.path.join(image_save_path, f"{metric_name.lower().replace(' ', '_')}_mnist.png")
        # 新增：自动创建目录
        os.makedirs(image_save_path, exist_ok=True)
        save_path = os.path.join(image_save_path, f"{metric_name.lower().replace(' ', '_')}_mnist.png")
        plt.savefig(save_path)
        plt.close()
    # 折线图
    plt.figure(figsize=(10, 6))
    for name in model_names:
        vals = [metrics_dict[m][name] for m in metrics_dict]
        plt.plot(list(metrics_dict.keys()), vals, marker='o', color=color_map.get(name, "#333333"), label=name)
    plt.title("ML Methods Metrics Comparison on MNIST (Sampled)")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(image_save_path, exist_ok=True)
    save_path = os.path.join(image_save_path, "ml_methods_metrics_line.png")
    plt.savefig(save_path)
    plt.close()

# 6. 定义深度学习模型
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, channels=32, depth=2):
        super().__init__()
        conv_layers = []
        in_channels = 1
        for i in range(depth):
            conv_layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            in_channels = channels
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

# 主流程
data_sizes = [200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000, 18000, 20000, 30000, 40000, 60000]
all_results = {metric: {name: [] for name in ["Decision Tree", "Random Forest", "SVM", "Logistic Regression", "KNN", "MLP", "CNN"]} for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]}

for data_size in data_sizes:
    print(f"\n=== Training with data size: {data_size} ===")
    # 采样数据
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=data_size, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_sub, test_size=0.2, random_state=42)

    # 训练并记录传统ML模型
    metrics_dict = {k: {} for k in ["Accuracy", "Precision", "Recall", "F1 Score"]}
    model_names = list(models.keys())
    for name, model in models.items():
        acc, prec, rec, f1 = train_ml_model(model, X_train, y_train, X_test, y_test)
        metrics_dict["Accuracy"][name] = acc
        metrics_dict["Precision"][name] = prec
        metrics_dict["Recall"][name] = rec
        metrics_dict["F1 Score"][name] = f1

    # 训练并记录MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_acc, mlp_prec, mlp_rec, mlp_f1 = train_dl_model(
        SimpleMLP(), X_train, y_train, X_test, y_test, epochs=10, lr=1e-3, batch_size=128, is_cnn=False, device=device
    )
    metrics_dict["Accuracy"]["MLP"] = mlp_acc
    metrics_dict["Precision"]["MLP"] = mlp_prec
    metrics_dict["Recall"]["MLP"] = mlp_rec
    metrics_dict["F1 Score"]["MLP"] = mlp_f1
    model_names.append("MLP")

    # 训练并记录CNN
    cnn_acc, cnn_prec, cnn_rec, cnn_f1 = train_dl_model(
        SimpleCNN(), X_train, y_train, X_test, y_test, epochs=10, lr=1e-3, batch_size=128, is_cnn=True, device=device
    )
    metrics_dict["Accuracy"]["CNN"] = cnn_acc
    metrics_dict["Precision"]["CNN"] = cnn_prec
    metrics_dict["Recall"]["CNN"] = cnn_rec
    metrics_dict["F1 Score"]["CNN"] = cnn_f1
    model_names.append("CNN")

    # 保存每个数据量下的结果
    for metric in all_results:
        for name in all_results[metric]:
            all_results[metric][name].append(metrics_dict[metric].get(name, np.nan))

    # 绘图（每个数据量单独保存）
    plot_metrics_bar_line(metrics_dict, model_names, os.path.join(image_save_path, f"metrics_{data_size}"))

# 绘制不同数据量下各方法各指标的折线图
color_map = {
    "Decision Tree": "#1f77b4",
    "Random Forest": "#ff7f0e",
    "SVM": "#2ca02c",
    "Logistic Regression": "#d62728",
    "KNN": "#9467bd",
    "MLP": "#8c564b",
    "CNN": "#e377c2"
}
for metric in all_results:
    plt.figure(figsize=(10, 6))
    for name in all_results[metric]:
        plt.plot(data_sizes, all_results[metric][name], marker='o', label=name, color=color_map.get(name, "#333333"))
    plt.title(f"{metric} vs Data Size")
    plt.xlabel("Training Data Size")
    plt.ylabel(metric)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    save_path = os.path.join(image_save_path, f"{metric.lower().replace(' ', '_')}_vs_datasize.png")
    plt.savefig(save_path)
    plt.close()
    print(f"折线图已保存：{save_path}")

print("✅ 不同数据量下的所有实验与可视化已完成！")
