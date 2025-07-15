from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# 文件上传路径
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 类别名称（与你训练时一致）
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# 构建模型结构并加载权重
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 类分类
model.load_state_dict(torch.load('plant_model.pth', map_location='cpu'))
model.eval()

# 图像预处理步骤（与训练一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取上传文件
        file = request.files['file']
        if not file:
            return "No file uploaded", 400

        # 保存文件到 static/uploads
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 图像处理与预测
        image = Image.open(filepath).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = class_names[predicted.item()]

        # 显示结果
        return render_template('result.html', label=label, image='/' + filepath)

    return render_template('index.html')

