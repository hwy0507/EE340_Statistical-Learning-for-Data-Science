import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# 配置
DATA_DIR = "/home/nkd/ouyangzl/Project/data"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(DATA_DIR, transform=transform, train=True, download=True)
test_dataset = MNIST(DATA_DIR, transform=transform, train=False, download=True)

# 提取测试集数据
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
test_images, test_labels = next(iter(test_loader))
test_images = test_images.view(test_images.size(0), -1).numpy()
test_labels = test_labels.numpy()

# 定义机器学习模型
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel="linear")
}

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # 全连接层将在 forward 中动态初始化
        self.fc1 = None
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # (batch_size, 32, 28, 28)
        x = self.pool(self.relu(self.conv2(x)))  # (batch_size, 64, 14, 14)
        x = self.pool(x)  # (batch_size, 64, 7, 7)
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 64*7*7)

        # 动态初始化 fc1
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = self.relu(self.fc1(x))  # (batch_size, 128)
        x = self.fc2(x)  # (batch_size, 10)
        return x

# 定义浅层 MLP 模型
class ShallowMLP(nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义深层 MLP 模型
class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 定义浅层 Transformer 模型
class ShallowTransformer(nn.Module):
    def __init__(self):
        super(ShallowTransformer, self).__init__()
        self.embedding = nn.Linear(28 * 28, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256),
            num_layers=2
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.embedding(x).unsqueeze(1)  # 添加序列维度
        x = self.transformer(x).squeeze(1)  # 移除序列维度
        x = self.fc(x)
        return x

def train_cnn(train_loader, test_loader, epochs=5, lr=0.001):
    """
    训练 CNN 模型并记录损失和准确率。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []
    test_recalls = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # 测试模型
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average="macro")
        test_accuracies.append(accuracy)
        test_recalls.append(recall)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")

    return model, train_losses, test_accuracies, test_recalls

def train_deep_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    """
    训练深度学习模型（MLP 或 Transformer），记录损失和准确率。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_scores = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # 测试模型
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        score = recall_score(all_labels, all_preds, average="macro")
        test_scores.append(score)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Score: {score:.4f}")

    return train_losses, test_scores

def train_and_record_epochs(model_class, train_loader, test_loader, epochs_list, lr=0.001):
    """
    针对不同的 epoch 数训练模型，并记录每个 epoch 的性能。
    Args:
        - model_class: 模型类（如 CNN, ShallowMLP, DeepMLP, ShallowTransformer）
        - train_loader: 训练数据加载器
        - test_loader: 测试数据加载器
        - epochs_list: 需要训练的 epoch 数列表
        - lr: 学习率
    Returns:
        - results: 每个 epoch 的性能记录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for epochs in epochs_list:
        print(f"Training {model_class.__name__} for {epochs} epochs...")
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 测试模型
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average="macro")
        results[epochs] = {"accuracy": accuracy, "recall": recall}
        print(f"Epochs: {epochs}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")

    return results

def plot_metrics(train_losses, test_accuracies, test_recalls, save_path=None):
    """
    绘制损失、准确率和召回率曲线。
    """
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, test_accuracies, label="Test Accuracy", marker='o')
    plt.plot(epochs, test_recalls, label="Test Recall", marker='o')
    plt.title("Test Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_epoch_comparison(results_dict, metric, title, ylabel, save_path=None):
    """
    绘制不同模型在不同 epoch 下的性能对比图。
    Args:
        - results_dict: 每个模型的性能记录字典
        - metric: 性能指标名称（如 "accuracy" 或 "recall"）
        - title: 图像标题
        - ylabel: y 轴标签
        - save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 8))
    for model_name, results in results_dict.items():
        epochs = list(results.keys())
        scores = [results[epoch][metric] for epoch in epochs]
        plt.plot(epochs, scores, label=model_name, marker='o')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_balanced_subset(dataset, num_samples_per_class):
    """
    从数据集中均匀选取每个类别的样本。
    Args:
        - dataset: 数据集
        - num_samples_per_class: 每个类别的样本数量
    Returns:
        - subset: 子集索引
    """
    indices = []
    targets = np.array(dataset.targets)
    for label in range(10):
        label_indices = np.where(targets == label)[0]
        if len(label_indices) < num_samples_per_class:
            # 如果样本数量不足，直接使用所有样本
            selected_indices = label_indices
        else:
            # 否则随机采样指定数量的样本
            selected_indices = np.random.choice(label_indices, num_samples_per_class, replace=False)
        indices.extend(selected_indices)
    return Subset(dataset, indices)

def train_and_evaluate(models, train_images, train_labels, test_images, test_labels):
    """
    训练模型并评估性能。
    Args:
        - models: 模型字典
        - train_images: 训练图像
        - train_labels: 训练标签
        - test_images: 测试图像
        - test_labels: 测试标签
    Returns:
        - results: 每个模型的性能指标
    """
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(train_images, train_labels)
        predictions = model.predict(test_images)
        accuracy = accuracy_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions, average="macro")
        results[name] = {"accuracy": accuracy, "recall": recall}
    return results

def plot_results(data_sizes, results, metric):
    """
    绘制数据量与性能指标的关系曲线。
    Args:
        - data_sizes: 数据量列表
        - results: 每个模型的性能结果
        - metric: 性能指标名称
    """
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        plt.plot(data_sizes, [metrics[size][metric] for size in data_sizes], label=model_name)
    plt.xlabel("Number of Training Samples")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} vs. Training Data Size")
    plt.legend()
    plt.grid()
    plt.show()

def plot_results_with_annotations(data_sizes, results, metric, title, ylabel, save_path=None):
    """
    绘制数据量与性能指标的关系曲线，并添加详细说明。
    Args:
        - data_sizes: 数据量列表
        - results: 每个模型的性能结果
        - metric: 性能指标名称
        - title: 图像标题
        - ylabel: y轴标签
        - save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 8))
    for model_name, metrics in results.items():
        metric_values = [metrics[size][metric] for size in data_sizes]
        plt.plot(data_sizes, metric_values, label=model_name, marker='o')
        # 添加注释
        for i, value in enumerate(metric_values):
            plt.text(data_sizes[i], value, f"{value:.2f}", fontsize=8, ha='right')

    plt.xlabel("Number of Training Samples", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_initial_exploration(results, save_path=None):
    """
    绘制初始探索阶段的模型准确率和召回率。
    Args:
        - results: 每个模型的初始性能结果
        - save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))
    metrics = ["accuracy", "recall"]
    for metric in metrics:
        for model_name, model_metrics in results.items():
            plt.bar(model_name + f" ({metric})", model_metrics[metric], label=f"{model_name} ({metric})")
    plt.title("Initial Exploration: Accuracy and Recall")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_comparison(data_sizes, results, save_path=None):
    """
    绘制不同数据量下模型的准确率和召回率对比图。
    Args:
        - data_sizes: 数据量列表
        - results: 每个模型的性能结果
        - save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 8))
    metrics = ["accuracy", "recall"]
    for metric in metrics:
        for model_name, model_results in results.items():
            plt.plot(data_sizes, [model_results[size][metric] for size in data_sizes], label=f"{model_name} ({metric})", marker='o')
    plt.title("Comparison of Accuracy and Recall Across Data Sizes")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_model_comparisons(epochs, shallow_mlp_scores, deep_mlp_scores, transformer_scores, save_path=None):
    """
    绘制浅层 MLP、深层 MLP 和浅层 Transformer 的对比图。
    """
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, shallow_mlp_scores, label="Shallow MLP", marker='o')
    plt.plot(epochs, deep_mlp_scores, label="Deep MLP", marker='o')
    plt.plot(epochs, transformer_scores, label="Shallow Transformer", marker='o')
    plt.title("Model Comparison: Score vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # 初始配置
    initial_samples_per_class = 100  # 每个类别的初始样本数量
    increment_samples_per_class = 100  # 每次增加的样本数量
    max_samples_per_class = 1000  # 每个类别的最大样本数量

    data_sizes = []  # 用于记录每次实验的总样本数量
    results = {name: {} for name in MODELS.keys()}  # 用于存储每个模型的性能结果

    # 初始化测试集 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始探索：从每个类别中选取 initial_samples_per_class 个样本
    print("Initial Exploration...")
    subset = get_balanced_subset(train_dataset, initial_samples_per_class)
    train_loader = DataLoader(subset, batch_size=64, shuffle=True)
    train_images, train_labels = next(iter(train_loader))
    train_images = train_images.view(train_images.size(0), -1).numpy()
    train_labels = train_labels.numpy()

    # 训练和评估机器学习模型
    metrics = train_and_evaluate(MODELS, train_images, train_labels, test_images, test_labels)
    for model_name, model_metrics in metrics.items():
        results[model_name][initial_samples_per_class * 10] = model_metrics
    data_sizes.append(initial_samples_per_class * 10)

    # 绘制初始探索阶段的准确率和召回率
    plot_initial_exploration(
        {model_name: model_metrics for model_name, model_metrics in metrics.items()},
        save_path="/home/nkd/ouyangzl/Project/images/initial_exploration.png"
    )

    # 训练和评估 CNN 模型
    print("Training CNN Model...")
    cnn_model, train_losses, test_accuracies, test_recalls = train_cnn(train_loader, test_loader, epochs=5)
    plot_metrics(train_losses, test_accuracies, test_recalls, save_path="/home/nkd/ouyangzl/Project/images/cnn_metrics.png")

    # 使用全部 MNIST 数据训练模型
    print("Training on Full MNIST Dataset...")
    full_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    full_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 浅层 MLP
    print("Training Shallow MLP...")
    shallow_mlp = ShallowMLP()
    shallow_mlp_losses, shallow_mlp_scores = train_deep_model(shallow_mlp, full_train_loader, full_test_loader, epochs=5)
    plot_metrics(
        shallow_mlp_losses, shallow_mlp_scores, shallow_mlp_scores,
        save_path="/home/nkd/ouyangzl/Project/images/shallow_mlp_metrics.png"
    )

    # 深层 MLP
    print("Training Deep MLP...")
    deep_mlp = DeepMLP()
    deep_mlp_losses, deep_mlp_scores = train_deep_model(deep_mlp, full_train_loader, full_test_loader, epochs=5)
    plot_metrics(
        deep_mlp_losses, deep_mlp_scores, deep_mlp_scores,
        save_path="/home/nkd/ouyangzl/Project/images/deep_mlp_metrics.png"
    )

    # 浅层 Transformer
    print("Training Shallow Transformer...")
    shallow_transformer = ShallowTransformer()
    transformer_losses, transformer_scores = train_deep_model(shallow_transformer, full_train_loader, full_test_loader, epochs=5)
    plot_metrics(
        transformer_losses, transformer_scores, transformer_scores,
        save_path="/home/nkd/ouyangzl/Project/images/transformer_metrics.png"
    )

    # 绘制不同模型的对比图
    print("Plotting Model Comparisons...")
    epochs = range(1, 6)
    plot_model_comparisons(
        epochs,
        shallow_mlp_scores,
        deep_mlp_scores,
        transformer_scores,
        save_path="/home/nkd/ouyangzl/Project/images/model_comparison.png"
    )

    # 逐步增加数据量
    print("Incremental Data Exploration...")
    for num_samples_per_class in tqdm(range(initial_samples_per_class + increment_samples_per_class, max_samples_per_class + 1, increment_samples_per_class)):
        print(f"Using {num_samples_per_class * 10} training samples...")
        data_sizes.append(num_samples_per_class * 10)

        # 获取平衡子集
        subset = get_balanced_subset(train_dataset, num_samples_per_class)
        train_loader = DataLoader(subset, batch_size=64, shuffle=True)
        train_images, train_labels = next(iter(train_loader))
        train_images = train_images.view(train_images.size(0), -1).numpy()
        train_labels = train_labels.numpy()

        # 训练和评估机器学习模型
        metrics = train_and_evaluate(MODELS, train_images, train_labels, test_images, test_labels)
        for model_name, model_metrics in metrics.items():
            results[model_name][num_samples_per_class * 10] = model_metrics

    # 绘制数据量与准确率的关系曲线
    print("Plotting Results...")
    plot_results_with_annotations(
        data_sizes, results, "accuracy",
        title="Accuracy vs. Training Data Size",
        ylabel="Accuracy",
        save_path="/home/nkd/ouyangzl/Project/images/accuracy_vs_data_size.png"
    )
    plot_results_with_annotations(
        data_sizes, results, "recall",
        title="Recall vs. Training Data Size",
        ylabel="Recall",
        save_path="/home/nkd/ouyangzl/Project/images/recall_vs_data_size.png"
    )

    # 绘制不同数据量下模型的准确率和召回率对比图
    plot_comparison(
        data_sizes, results,
        save_path="/home/nkd/ouyangzl/Project/images/comparison_across_data_sizes.png"
    )

    # 定义 epoch 列表
    epochs_list = [5, 10, 25, 50, 100, 150, 200]

    # 使用不同大小的数据集训练深度神经网络
    print("Training Deep Models with Different Epochs and Data Sizes...")
    data_sizes = [1000, 5000, 10000, 60000]  # 数据集大小列表
    results_dict = {}

    for data_size in data_sizes:
        print(f"Using {data_size} training samples...")
        subset = get_balanced_subset(train_dataset, data_size // 10)
        train_loader = DataLoader(subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # 记录每个模型的结果
        results_dict[f"CNN ({data_size} samples)"] = train_and_record_epochs(CNN, train_loader, test_loader, epochs_list)
        results_dict[f"Shallow MLP ({data_size} samples)"] = train_and_record_epochs(ShallowMLP, train_loader, test_loader, epochs_list)
        results_dict[f"Deep MLP ({data_size} samples)"] = train_and_record_epochs(DeepMLP, train_loader, test_loader, epochs_list)
        results_dict[f"Shallow Transformer ({data_size} samples)"] = train_and_record_epochs(ShallowTransformer, train_loader, test_loader, epochs_list)

    # 绘制不同模型在不同 epoch 下的准确率对比图
    print("Plotting Epoch Comparison for Accuracy...")
    plot_epoch_comparison(
        results_dict,
        metric="accuracy",
        title="Accuracy vs. Epochs for Different Models and Data Sizes",
        ylabel="Accuracy",
        save_path="/home/nkd/ouyangzl/Project/images/epoch_comparison_accuracy.png"
    )

    # 绘制不同模型在不同 epoch 下的召回率对比图
    print("Plotting Epoch Comparison for Recall...")
    plot_epoch_comparison(
        results_dict,
        metric="recall",
        title="Recall vs. Epochs for Different Models and Data Sizes",
        ylabel="Recall",
        save_path="/home/nkd/ouyangzl/Project/images/epoch_comparison_recall.png"
    )

if __name__ == "__main__":
    main()

