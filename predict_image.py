import torch
from torchvision import models, transforms
from PIL import Image
import os

# 类别名称（按训练时文件夹顺序）
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# 加载模型结构
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 和训练时保持一致
model.load_state_dict(torch.load("plant_model.pth"))  # 加载权重
model.eval()

# 图像预处理步骤（和训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 要预测的图片路径（你可以换成别的）
image_path = r"C:\Users\Yinzi\OneDrive\桌面\Grade5\2025Summer\data\Common_Rust\Corn_Common_Rust (471).jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # 加上 batch 维度

# 预测
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_label = class_names[predicted.item()]

print(f"预测结果：{predicted_label} ✅")
