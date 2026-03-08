import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# 尝试导入 KMedoids
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None

# 配置保存路径和参数
save_dir = '/home/nkd/ouyangzl/Project/feature'
os.makedirs(save_dir, exist_ok=True)
DATA_ROOT = '/home/nkd/ouyangzl/Project/data/MNIST'
batch_size = 512
n_samples = 5000  # 使用前5000个样本

# 加载 MNIST 数据集，取前5000个样本，并转换为一维向量
transform = transforms.ToTensor()
mnist_ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
subset_idx = list(range(n_samples))
mnist_subset = Subset(mnist_ds, subset_idx)
loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=False)

all_features = []
all_labels = []
for imgs, labels in loader:
    # imgs shape: (batch, 1, 28, 28) -> flatten to (batch, 784)
    flat_imgs = imgs.view(imgs.size(0), -1).numpy()
    all_features.append(flat_imgs)
    all_labels.append(labels.numpy())
features = np.concatenate(all_features, axis=0)  # (n_samples, 784)
y_true = np.concatenate(all_labels, axis=0)

# 使用 t-SNE 降维至 2D
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(features)  # (n_samples, 2)

def cluster_and_visualize(X, y_true, method_key, title, save_prefix):
    """
    根据 method_key 构造聚类器，对 X 进行聚类，
    绘制聚类结果（左图）和真实标签（右图），
    计算 ARI 和 NMI 指标，并保存图像。
    """
    # 动态构建聚类器
    if method_key == 'kmeans':
        clusterer = KMeans(n_clusters=10, random_state=0)
    elif method_key == 'kmedoids':
        if KMedoids is None:
            print("KMedoids not installed, skip")
            return None, None
        clusterer = KMedoids(n_clusters=10, random_state=0)
    elif method_key == 'agg':
        clusterer = AgglomerativeClustering(n_clusters=10)
    elif method_key == 'dbscan':
        clusterer = DBSCAN(eps=3, min_samples=5, n_jobs=-1)
    elif method_key == 'gmm':
        clusterer = GaussianMixture(n_components=10, random_state=0)
    elif method_key == 'spectral':
        clusterer = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', random_state=0, n_jobs=-1)
    else:
        print(f"Method {method_key} not implemented, skip")
        return None, None

    # 特殊处理 GMM
    if method_key == 'gmm':
        clusterer.fit(X)
        y_pred = clusterer.predict(X)
    else:
        y_pred = clusterer.fit_predict(X)

    # 计算一致性指标
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # 绘图：左图为聚类结果，右图为真实标签
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', s=2)
    plt.title(f'{title} (ARI={ari:.3f}, NMI={nmi:.3f})')
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=2)
    plt.title('True Labels')
    plt.tight_layout()

    # 保存图像
    if save_prefix:
        fname = f"{save_prefix}_{method_key}_ari{ari:.3f}_nmi{nmi:.3f}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=200)
    plt.show()
    print(f"{title} using {method_key} => ARI: {ari:.4f}, NMI: {nmi:.4f}")
    return ari, nmi

# 动态构建 methods，保证未安装的模块不报错
methods = [
    ('kmeans', 'KMeans'),
    ('kmedoids', 'KMedoids'),
    ('agg', 'Agglomerative'),
    ('dbscan', 'DBSCAN'),
    ('gmm', 'GMM'),
    ('spectral', 'Spectral'),
]

results = []
for method_key, method_name in methods:
    print(f"\nClustering with {method_name}")
    ari, nmi = cluster_and_visualize(X_tsne, y_true, method_key, title=f"TSNE + {method_name}", save_prefix="tsne")
    if ari is not None and nmi is not None:
        results.append((method_name, ari, nmi))

def plot_metric_comparison(results, metric_idx, metric_name, save_name):
    """绘制聚类方法对比条形图"""
    names = [r[0] for r in results]
    values = [r[metric_idx] if r[metric_idx] is not None else 0 for r in results]
    plt.figure(figsize=(10, 4))
    plt.bar(names, values, color='skyblue')
    plt.ylabel(metric_name)
    plt.title(f'Clustering {metric_name}')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name), dpi=200)
    plt.show()

# 绘制 ARI 和 NMI 对比条形图
plot_metric_comparison(results, 1, 'ARI', 'clustering_ARI.png')
plot_metric_comparison(results, 2, 'NMI', 'clustering_NMI.png')