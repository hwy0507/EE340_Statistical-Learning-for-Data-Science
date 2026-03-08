"""
Task 5: 欠拟合与过拟合现象探究

本文件提供了一个灵活的接口，用于对多种模型（机器学习和深度学习）进行错误样本的记录和降维可视化，
从而探究欠拟合与过拟合现象。主要解决如下问题：
1. 记录各模型在分类任务中错误分类的样本，并通过t-SNE或UMAP降维后可视化展示，
   以分析错误样本在不同模型中是否具有相似性及其特点。（对应Task 5第1点）
2. 在深度学习模型中，通过调整正则化、Dropout和模型复杂度，观察其对过拟合现象的影响。（对应Task 5第2-4点）

注意：为解决plt.scatter颜色数组维度不匹配的报错，本文件对传入的labels进行检查，
如果发现labels数组仅有1个元素但样本数不止1，则自动使用np.repeat扩充labels，与样本数保持一致。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 如果项目中安装了umap-learn，则可选用 UMAP 降维
import umap

# 导入数据加载与指标计算模块（根据实际项目结构调整路径）
from src.data.mnist_loader import load_mnist
from src.utils.metrics import compute_metrics

# 导入机器学习模型训练方法（接口可灵活替换）
from src.models.ml_models import train_decision_tree, train_random_forest, train_svm, train_logistic_regression

# 深度学习相关导入
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn_model import CNN

# 设定图片输出目录
IMAGE_DIR = "/home/nkd/ouyangzl/Project/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# 固定随机种子，保证结果可复现
SEED = 42


##########################################
# 通用数据降维与可视化函数（支持 t-SNE 和 UMAP）
##########################################
def visualize_with_dim_reduction(data, labels, method='tsne', title="Dimensionality Reduction Visualization", save_path=None):
    """
    使用降维技术对数据进行降维并可视化展示。
    
    参数:
      data      : numpy 数组，形状为 (num_samples, num_features)
      labels    : numpy 数组，应与 data 行数一致
      method    : 降维方法；可以传入 'tsne' 或 'umap'
      title     : 图像标题
      save_path : 保存图像的路径，若为 None 则不保存
    
    改进：如果 labels 数组只有一个元素而 data 有多行，会自动扩充 labels 数组。
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=SEED)
    elif method == 'umap':
        reducer = umap.UMAP(random_state=SEED)
    else:
        raise ValueError("Invalid method specified. Choose 'tsne' or 'umap'.")
    
    reduced_data = reducer.fit_transform(data)

    labels = np.squeeze(np.asarray(labels))
    if labels.ndim != 1 or labels.shape[0] != reduced_data.shape[0]:
        if labels.size == 1 and reduced_data.shape[0] > 1:
            labels = np.repeat(labels, reduced_data.shape[0])
        else:
            raise ValueError(f"The 'labels' array must be 1D with length equal to number of samples ({reduced_data.shape[0]}), got shape {labels.shape}")

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, label="Error Flag (0: Correct, 1: Error)")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if save_path:
        plt.savefig(save_path)
    plt.show()


##########################################
# Part 1: 机器学习模型的错误样本分析与可视化
##########################################
def evaluate_ml_model(train_func, X_train, y_train, X_test, y_test):
    """
    训练并评估机器学习模型，同时返回测试集真实标签和预测标签（均为1D数组）。

    参数:
      train_func: 模型训练函数
      X_train, y_train: 训练集数据和标签（numpy数组）
      X_test, y_test:   测试集数据和标签（numpy数组）

    返回:
      y_test, y_pred: 测试集真实标签和预测标签
    """
    model = train_func(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.ravel(y_test), np.ravel(y_pred)

def analyze_ml_models():
    """
    对多个机器学习模型进行训练、评估及错误样本降维可视化。
    记录模型在测试集中的误分类样本，并用降维方法展示正确与错误样本的分布，
    有助于分析错误样本的特点及分类困难的原因。

    该方法对应 Task 5 中记录并分析错误样本的要求。此处分别调用 t-SNE 和 UMAP 两种降维方式，
    以便对比观察错误样本的分布。
    """
    batch_size = 2048
    # 加载MNIST数据（返回numpy格式的图像数据与标签）
    _, _, images, labels = load_mnist(batch_size=batch_size)

    # 简单划分训练集和测试集（此处可替换为官方测试集划分）
    split = int(0.8 * images.shape[0])
    X_train, X_test = images[:split], images[split:]
    y_train, y_test = labels[:split], labels[split:]

    # 定义不同机器学习模型，接口允许灵活替换
    model_dict = {
        "Decision Tree": train_decision_tree,
        "Random Forest": train_random_forest,
        "SVM": train_svm,
        "Logistic Regression": train_logistic_regression
    }

    for model_name, train_func in model_dict.items():
        print(f"Evaluating ML Model: {model_name}")
        true_labels, pred_labels = evaluate_ml_model(train_func, X_train, y_train, X_test, y_test)
        metrics = compute_metrics(true_labels, pred_labels)
        print(f"{model_name} Metrics:")
        for metric, score in metrics.items():
            print(f"   {metric.capitalize()}: {score:.4f}")

        # 生成错误标记数组：1 表示误分类，0 表示正确分类
        error_flags = np.where(true_labels != pred_labels, 1, 0)
        # 调试输出确保形状正确
        print(f"X_test samples: {X_test.shape[0]}, error_flags shape: {error_flags.shape}")

        # 使用 t-SNE 绘制误分类样本降维图
        title_tsne = f"{model_name} - t-SNE Visualization of Misclassified Samples"
        save_path_tsne = os.path.join(IMAGE_DIR, f"{model_name.replace(' ', '_').lower()}_tsne_errors.png")
        visualize_with_dim_reduction(X_test, error_flags, method='tsne', title=title_tsne, save_path=save_path_tsne)

        # 使用 UMAP 绘制误分类样本降维图
        title_umap = f"{model_name} - UMAP Visualization of Misclassified Samples"
        save_path_umap = os.path.join(IMAGE_DIR, f"{model_name.replace(' ', '_').lower()}_umap_errors.png")
        visualize_with_dim_reduction(X_test, error_flags, method='umap', title=title_umap, save_path=save_path_umap)
        
        print("\n")


##########################################
# Part 2: 深度学习CNN模型的正则化与过拟合探究
##########################################
def prepare_tensor_dataset(loader):
    """
    将DataLoader中的所有批次数据拼接成一个TensorDataset，用于整体评估与降维。

    参数:
      loader: torch.utils.data.DataLoader 对象

    返回:
      images: 拼接后的图像Tensor，形状 (N, C, H, W)
      labels: 拼接后的标签Tensor，形状 (N,)
    """
    image_list, label_list = [], []
    for data, target in loader:
        image_list.append(data)
        label_list.append(target)
    images = torch.cat(image_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return images, labels

def evaluate_dl_model(model, data_loader, device):
    """
    在深度学习模型上评估整个数据集，返回真实标签和预测标签（均为1D numpy数组）。

    参数:
      model      : 已训练的深度学习模型
      data_loader: DataLoader对象
      device     : 计算设备 (CPU 或 GPU)

    返回:
      y_true, y_pred: numpy数组格式的真实标签和预测标签
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
    y_true = np.ravel(torch.cat(all_targets).numpy())
    y_pred = np.ravel(torch.cat(all_preds).numpy())
    return y_true, y_pred

def train_cnn_with_regularization(weight_decay=0.0, dropout1_rate=0.25, dropout2_rate=0.5, epochs=10, lr=0.001):
    """
    训练CNN模型，允许通过调整L2正则化（weight_decay）和Dropout参数探究防止过拟合的效果。
    
    参数：
      weight_decay  : L2正则化系数（0.0 表示无正则化）
      dropout1_rate : 第一层Dropout的概率
      dropout2_rate : 第二层Dropout的概率
      epochs        : 训练轮数（已增加为10，可以根据需要进一步增大）
      lr            : 学习率
    
    改进：
      - 增加训练轮次以充分学习；
      - 训练结束后，分别使用 t-SNE 和 UMAP 绘制CNN误分类样本的降维图，检查图像是否正确。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, _, _ = load_mnist(batch_size=64)

    model = CNN().to(device)
    model.dropout1.p = dropout1_rate
    model.dropout2.p = dropout2_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()

    print(f"Training CNN with weight_decay={weight_decay}, dropout=({dropout1_rate}, {dropout2_rate}), epochs={epochs}")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss/len(train_loader):.4f}")

    # 评估模型结果
    y_true, y_pred = evaluate_dl_model(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    print("CNN Model Metrics with Regularization and Dropout:")
    for metric, score in metrics.items():
        print(f"   {metric.capitalize()}: {score:.4f}")

    # 可视化误分类情况——使用两种降维方式
    images_tensor, _ = prepare_tensor_dataset(test_loader)
    images_flat = images_tensor.view(images_tensor.size(0), -1).numpy()
    error_flags = np.where(y_true != y_pred, 1, 0)
    print(f"Test samples: {images_flat.shape[0]}, error_flags shape: {error_flags.shape}")
    
    # 使用 t-SNE 可视化错误样本
    title_tsne = f"CNN Errors (TSNE): wd={weight_decay}, d1={dropout1_rate}, d2={dropout2_rate}"
    save_path_tsne = os.path.join(IMAGE_DIR, f"cnn_errors_tsne_wd{weight_decay}_d1{dropout1_rate}_d2{dropout2_rate}.png")
    visualize_with_dim_reduction(images_flat, error_flags, method='tsne', title=title_tsne, save_path=save_path_tsne)
    
    # 使用 UMAP 可视化错误样本
    title_umap = f"CNN Errors (UMAP): wd={weight_decay}, d1={dropout1_rate}, d2={dropout2_rate}"
    save_path_umap = os.path.join(IMAGE_DIR, f"cnn_errors_umap_wd{weight_decay}_d1{dropout1_rate}_d2{dropout2_rate}.png")
    visualize_with_dim_reduction(images_flat, error_flags, method='umap', title=title_umap, save_path=save_path_umap)

def explore_dl_models():
    """
    实验不同正则化和 Dropout 参数配置下 CNN 模型的表现，探究如何缓解过拟合现象。
    用户可通过修改 experiment_configs 增加更多配置，比较不同正则化强度和 Dropout 组合对
    模型训练和泛化能力的影响。
    """
    experiment_configs = [
        {"weight_decay": 0.0,    "dropout1_rate": 0.25, "dropout2_rate": 0.5, "epochs": 5},
        {"weight_decay": 0.001,  "dropout1_rate": 0.5,  "dropout2_rate": 0.7, "epochs": 5},
    ]
    for config in experiment_configs:
        train_cnn_with_regularization(weight_decay=config["weight_decay"],
                                      dropout1_rate=config["dropout1_rate"],
                                      dropout2_rate=config["dropout2_rate"],
                                      epochs=config["epochs"])

##########################################
# 主接口函数：灵活调用不同模型进行实验
##########################################
def main():
    print("========== Task 5: 欠拟合与过拟合现象探究 ==========")
    print("--- Part 1: 机器学习模型错误样本分析 ---")
    analyze_ml_models()

    print("--- Part 2: 深度学习模型正则化与 Dropout 防止过拟合探究 ---")
    explore_dl_models()

if __name__ == '__main__':
    main()