import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# 修改为你的图像路径
DATA_DIR = r"C:\Users\Yinzi\OneDrive\桌面\Grade5\2025Summer\data"

# 自定义数据集类
class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_path = os.path.join(root_dir, cls)
            for img_file in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_file)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 图像变换（含增强）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# 创建数据集和数据加载器
dataset = PlantDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 测试可视化一批图像
def show_batch(loader):
    images, labels = next(iter(loader))
    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()

# 显示一批图像
show_batch(dataloader)
