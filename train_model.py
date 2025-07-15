import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from data_loader_test import PlantDataset  # 假设你前面那段数据加载代码放在 dataset_loader.py

# 超参数
num_classes = 4
num_epochs = 5
batch_size = 16
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
DATA_DIR = r"C:\Users\Yinzi\OneDrive\桌面\Grade5\2025Summer\data"

# 图像预处理（和前面一样）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# 数据集和加载器
dataset = PlantDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改最后一层
model = model.to(device)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss:.3f} | Accuracy: {acc:.2f}%")

print("训练完成 ✅")
torch.save(model.state_dict(), "plant_model.pth")
print("模型已保存 ✅")