from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    return metrics

# Task 3.1说明：
# Accuracy: 正确分类样本数量 / 总样本数量
# Precision: 正确预测为正类的数量 / 预测为正类的总数
# Recall: 正确预测为正类的数量 / 实际正类总数
# F1 Score: Precision和Recall的调和平均数，用于综合反映模型的分类效果