import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.data.mnist_loader import load_mnist
from src.models.cnn_model import CNN
from src.utils.metrics import compute_metrics

def prepare_dataset(loader):
    # 将DataLoader中的所有batch数据收集起来，并调整格式
    images_list, labels_list = [], []
    for img, label in loader:
        images_list.append(img)
        labels_list.append(label)
    images = torch.cat(images_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return images, labels

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1, keepdim=True)  # 获取预测的标签
            all_preds.append(preds.cpu())
            all_labels.append(target.cpu())
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        return compute_metrics(y_true, y_pred)

def main():
    batch_size = 64
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader, _, _ = load_mnist(batch_size=batch_size)
    
    model = CNN().to(device)
    optimizer = optim.Adan(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Training CNN...")
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    metrics = evaluate_model(model, test_loader, device)
    print("CNN Model Results:")
    
    for metric, score in metrics.items():
        print(f"\t{metric.capitalize()}: {score:.4f}")

if __name__ == '__main__':
    main()