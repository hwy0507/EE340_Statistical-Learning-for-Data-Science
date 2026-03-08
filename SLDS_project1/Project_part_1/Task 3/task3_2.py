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
X, _, y, _ = train_test_split(X, y, train_size=200, stratify=y, random_state=42)

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
metrics_dict = {k: {} for k in ["Accuracy", "Precision", "Recall", "F1 Score"]}
model_names = list(models.keys())

# 训练并记录传统ML模型
print("训练模型并计算指标...")
for name, model in models.items():
    acc, prec, rec, f1 = train_ml_model(model, X_train, y_train, X_test, y_test)
    metrics_dict["Accuracy"][name] = acc
    metrics_dict["Precision"][name] = prec
    metrics_dict["Recall"][name] = rec
    metrics_dict["F1 Score"][name] = f1
    print(f"{name} 完成")

# 训练并记录MLP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练 SimpleMLP ...")
mlp_acc, mlp_prec, mlp_rec, mlp_f1 = train_dl_model(
    SimpleMLP(), X_train, y_train, X_test, y_test, epochs=10, lr=1e-3, batch_size=128, is_cnn=False, device=device
)
metrics_dict["Accuracy"]["MLP"] = mlp_acc
metrics_dict["Precision"]["MLP"] = mlp_prec
metrics_dict["Recall"]["MLP"] = mlp_rec
metrics_dict["F1 Score"]["MLP"] = mlp_f1
model_names.append("MLP")

# 训练并记录CNN
print("训练 SimpleCNN ...")
cnn_acc, cnn_prec, cnn_rec, cnn_f1 = train_dl_model(
    SimpleCNN(), X_train, y_train, X_test, y_test, epochs=10, lr=1e-3, batch_size=128, is_cnn=True, device=device
)
metrics_dict["Accuracy"]["CNN"] = cnn_acc
metrics_dict["Precision"]["CNN"] = cnn_prec
metrics_dict["Recall"]["CNN"] = cnn_rec
metrics_dict["F1 Score"]["CNN"] = cnn_f1
model_names.append("CNN")

# 绘图
plot_metrics_bar_line(metrics_dict, model_names, image_save_path)

# 9. 探索不同网络结构和深度对CNN性能的影响
cnn_depths = [1, 2, 3, 4]
cnn_channels = [16, 32, 64]
cnn_results = {}
for depth in cnn_depths:
    for ch in cnn_channels:
        print(f"训练 CNN: depth={depth}, channels={ch}")
        acc, prec, rec, f1 = train_dl_model(
            SimpleCNN(channels=ch, depth=depth), X_train, y_train, X_test, y_test,
            epochs=10, lr=1e-3, batch_size=128, is_cnn=True, device=device
        )
        cnn_results[(depth, ch)] = (acc, prec, rec, f1)

# 绘制不同深度和通道数下的CNN准确率热力图
acc_matrix = np.zeros((len(cnn_depths), len(cnn_channels)))
for i, d in enumerate(cnn_depths):
    for j, c in enumerate(cnn_channels):
        acc_matrix[i, j] = cnn_results[(d, c)][0]
plt.figure(figsize=(8, 6))
plt.imshow(acc_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Accuracy')
plt.xticks(range(len(cnn_channels)), cnn_channels)
plt.yticks(range(len(cnn_depths)), cnn_depths)
plt.xlabel('Channels')
plt.ylabel('Depth')
plt.title('CNN Accuracy vs Depth and Channels')
plt.savefig(os.path.join(image_save_path, "cnn_depth_channel_accuracy.png"))
plt.close()

# 10. 超参数实验：batch size、MLP隐藏层神经元数、CNN深度
batch_sizes = [32, 64, 128, 256]
mlp_hidden_dims = [64, 128, 256, 512]
cnn_depths = [1, 2, 3, 4]
cnn_channels = [16, 32, 64]

# 10.1 不同 batch size 对 MLP/CNN 的影响
mlp_batch_results = []
cnn_batch_results = []
for bs in batch_sizes:
    print(f"MLP batch size={bs}")
    acc, prec, rec, f1 = train_dl_model(
        SimpleMLP(hidden_dim=128), X_train, y_train, X_test, y_test,
        epochs=10, lr=1e-3, batch_size=bs, is_cnn=False, device=device
    )
    mlp_batch_results.append(acc)
    print(f"CNN batch size={bs}")
    acc, prec, rec, f1 = train_dl_model(
        SimpleCNN(), X_train, y_train, X_test, y_test,
        epochs=10, lr=1e-3, batch_size=bs, is_cnn=True, device=device
    )
    cnn_batch_results.append(acc)
plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, mlp_batch_results, marker='o', label='MLP')
plt.plot(batch_sizes, cnn_batch_results, marker='o', label='CNN')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.title('Effect of Batch Size on MLP and CNN')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(image_save_path, "batch_size_effect.png"))
plt.close()

# 10.2 MLP隐藏层神经元数影响
mlp_hidden_results = []
for h in mlp_hidden_dims:
    print(f"MLP hidden dim={h}")
    acc, prec, rec, f1 = train_dl_model(
        SimpleMLP(hidden_dim=h), X_train, y_train, X_test, y_test,
        epochs=10, lr=1e-3, batch_size=128, is_cnn=False, device=device
    )
    mlp_hidden_results.append(acc)
plt.figure(figsize=(8, 5))
plt.plot(mlp_hidden_dims, mlp_hidden_results, marker='o')
plt.xlabel('MLP Hidden Dim')
plt.ylabel('Accuracy')
plt.title('Effect of Hidden Dim on MLP')
plt.grid(True)
plt.savefig(os.path.join(image_save_path, "mlp_hidden_dim_effect.png"))
plt.close()

# 10.3 CNN深度影响
cnn_depth_results = []
for d in cnn_depths:
    print(f"CNN depth={d}")
    acc, prec, rec, f1 = train_dl_model(
        SimpleCNN(depth=d), X_train, y_train, X_test, y_test,
        epochs=10, lr=1e-3, batch_size=128, is_cnn=True, device=device
    )
    cnn_depth_results.append(acc)
plt.figure(figsize=(8, 5))
plt.plot(cnn_depths, cnn_depth_results, marker='o')
plt.xlabel('CNN Depth')
plt.ylabel('Accuracy')
plt.title('Effect of CNN Depth')
plt.grid(True)
plt.savefig(os.path.join(image_save_path, "cnn_depth_effect.png"))
plt.close()

# 10.4 CNN每层通道数影响
cnn_channel_results = []
for ch in cnn_channels:
    print(f"CNN channels={ch}")
    acc, prec, rec, f1 = train_dl_model(
        SimpleCNN(channels=ch), X_train, y_train, X_test, y_test,
        epochs=10, lr=1e-3, batch_size=128, is_cnn=True, device=device
    )
    cnn_channel_results.append(acc)
plt.figure(figsize=(8, 5))
plt.plot(cnn_channels, cnn_channel_results, marker='o')
plt.xlabel('CNN Channels')
plt.ylabel('Accuracy')
plt.title('Effect of CNN Channels')
plt.grid(True)
plt.savefig(os.path.join(image_save_path, "cnn_channels_effect.png"))
plt.close()

# 分析注释
"""
超参数影响分析：
- batch size 增大，训练更稳定但可能收敛变慢，过大可能影响泛化。
- MLP隐藏层神经元数增加，模型表达能力增强，但过大易过拟合且训练变慢。
- CNN深度增加，能提取更复杂特征，但过深可能梯度消失或过拟合。
- CNN每层通道数增大，特征表达能力增强，但参数量和计算量也增加，需权衡。
实验结果显示，合理调整超参数能显著提升模型性能，但需结合数据规模和计算资源综合考虑。
"""

# 分析注释
"""
性能分析与原因讨论（含深度学习）：
- MLP：对像素展平后进行全连接，能捕捉部分全局特征，但缺乏空间结构建模能力，通常优于简单ML方法但弱于CNN。
- CNN：能自动提取空间局部特征，对图像结构有更强建模能力，通常在MNIST等图像任务上表现最佳。
- 网络深度和通道数增加通常能提升CNN性能，但过深或过宽可能导致过拟合或训练困难。
- 机器学习方法如随机森林、SVM在小样本或特征工程充分时表现良好，但在大规模图像任务上通常不及深度学习。
- 实验结果显示，CNN在MNIST上通常优于MLP和传统ML方法，且结构调整（如深度/通道数）对性能有显著影响。
"""
print("✅ 所有任务已完成！")
