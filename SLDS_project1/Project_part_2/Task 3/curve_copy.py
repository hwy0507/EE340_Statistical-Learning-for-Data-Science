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
from sklearn.manifold import TSNE
import umap

# 配置
mnist_data_path = "/home/nkd/ouyangzl/Project/data/MNIST"
image_save_path = "/home/nkd/ouyangzl/Project/Task 3/images"
# 加载 MNIST 数据集
print("加载 MNIST 数据中...")
mnist = fetch_openml('mnist_784', data_home=mnist_data_path, version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# 训练函数
def train_ml_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    return acc, prec, rec, f1

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

# 定义深度学习模型
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

# 数据量设置
data_sizes = [200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000, 18000, 20000, 30000, 40000, 60000]
all_results = {metric: {name: [] for name in ["Decision Tree", "Random Forest", "SVM", "Logistic Regression", "KNN", "MLP", "CNN"]} for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]}

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 模型定义
ml_models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier()
}

dl_models = {
    "MLP": SimpleMLP(),
    "CNN": SimpleCNN()
}

# 训练和评估
for size in data_sizes:
    print(f"训练数据量: {size}")
    X_train, X_test, y_train, y_test = train_test_split(X[:size], y[:size], test_size=0.2, random_state=42)
    for name, model in ml_models.items():
        acc, prec, rec, f1 = train_ml_model(model, X_train, y_train, X_test, y_test)
        all_results["Accuracy"][name].append(acc)
        all_results["Precision"][name].append(prec)
        all_results["Recall"][name].append(rec)
        all_results["F1 Score"][name].append(f1)
    for name, model in dl_models.items():
        is_cnn = name == "CNN"
        acc, prec, rec, f1 = train_dl_model(model, X_train, y_train, X_test, y_test, is_cnn=is_cnn, device='cuda' if torch.cuda.is_available() else 'cpu')
        all_results["Accuracy"][name].append(acc)
        all_results["Precision"][name].append(prec)
        all_results["Recall"][name].append(rec)
        all_results["F1 Score"][name].append(f1)

# 画图
color_map = {
    "Decision Tree": "#1f77b4",
    "Random Forest": "#ff7f0e",
    "SVM": "#2ca02c",
    "Logistic Regression": "#d62728",
    "KNN": "#9467bd",
    "MLP": "#8c564b",
    "CNN": "#e377c2"
}

for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    plt.figure(figsize=(10, 6))
    for method in all_results[metric]:
        plt.plot(
            data_sizes,
            all_results[metric][method],
            marker='o',
            label=method,
            color=color_map.get(method, "#333333")
        )
    plt.title(f"{metric} vs Data Size (All Methods)")
    plt.xlabel("Training Data Size")
    plt.ylabel(metric)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(image_save_path, f"all_methods_{metric.lower().replace(' ', '_')}_vs_datasize.png")
    plt.savefig(save_path)
    plt.close()
    print(f"折线图已保存：{save_path}")
