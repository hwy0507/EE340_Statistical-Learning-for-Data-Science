import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0)
}

# ----------------------------
# 4. 定义训练和评估函数
# ----------------------------
def train_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    return acc, prec, rec, f1

# ----------------------------
# 5. 定义不同数据量
# ----------------------------
data_sizes = [1000, 5000, 10000, 20000, 40000]

# ----------------------------
# 6. 训练和评估不同数据量下的模型
# ----------------------------
results = {}
for size in data_sizes:
    print(f"\nTraining with data size: {size}")
    X_sample, _, y_sample, _ = train_test_split(X_raw, y_raw, train_size=size, stratify=y_raw, random_state=42)
    X_scaled = scaler.fit_transform(X_sample)
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_scaled, y_sample, test_size=0.2, random_state=42)
    results[size] = {}
    for name, model in models_ml.items():
        print(f"Training {name}...")
        acc, prec, rec, f1 = train_evaluate(model, X_train_ml, y_train_ml, X_test_ml, y_test_ml)
        results[size][name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# ----------------------------
# 7. 绘制折线图
# ----------------------------
metrics = ["accuracy", "precision", "recall", "f1"]
model_names = list(models_ml.keys())

for metric in metrics:
    plt.figure(figsize=(10, 6))
    for name in model_names:
        metric_values = [results[size][name][metric] for size in data_sizes]
        plt.plot(data_sizes, metric_values, marker='o', label=name)
    plt.xlabel("Data Size")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} vs Data Size")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(image_save_path, f"{metric}_vs_data_size.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# ----------------------------
# 8. 讨论与分析
# ----------------------------
"""
讨论：
1. 随着数据量的增加，所有模型的性能通常会提升，但提升幅度可能不同。
2. 不同模型在MNIST数据集上的表现存在差异，可能是因为模型对数据特征的拟合能力不同。
3. 决策树可能更容易受到过拟合的影响，而SVM和逻辑回归可能泛化能力更强。
4. 模型的选择取决于具体应用场景，需要根据实际情况进行权衡。
"""
print("所有实验已完成！")