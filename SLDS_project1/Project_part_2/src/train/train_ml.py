from Project.src.models.ml_models import train_decision_tree, train_random_forest, train_svm, train_logistic_regression, train_knn
from data.mnist_loader import load_mnist
from Project.src.utils.metrics import compute_metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)

def main():
    batch_size = 512
    # 只需使用完整numpy数组用于sklearn算法
    _, _, train_images, train_labels = load_mnist(batch_size=batch_size)
    # 这里简单以一部分数据作为测试集（通常应分离独立的测试集，此处示例简化处理）
    split = int(0.8 * train_images.shape[0])
    X_train, X_test = train_images[:split], train_images[split:]
    y_train, y_test = train_labels[:split], train_labels[split:]    
    
    models = {
        'Decision Tree': train_decision_tree,
        'Random Forest': train_random_forest,
        'SVM': train_svm,
        'Logistic Regression': train_logistic_regression
    }
    
    for name, func in models.items():
        print(f'Training {name}...')
        model = func(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        print(f"{name} Results:")
        for metric, score in metrics.items():
            print(f"\t{metric.capitalize()}: {score:.4f}")
        print('\n')

if __name__ == '__main__':
    main()
        