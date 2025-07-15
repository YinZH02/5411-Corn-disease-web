from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# 文件上传目录
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 类别标签（与你训练时一致）
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# 模型结构与权重加载
try:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load('plant_model.pth', map_location='cpu'))
    model.eval()
    print("✅ 模型加载成功")
except Exception as e:
    print("❌ 模型加载失败:", e)

# 图像预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file or file.filename == '':
                print("❌ 没有上传文件")
                return "No file uploaded", 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            print(f"📥 收到文件: {filepath}")

            image = Image.open(filepath).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                label = class_names[predicted.item()]
                print(f"🔍 预测结果: {label}")

            return render_template('result.html', label=label, image='/' + filepath)

        except Exception as e:
            print("❌ 处理失败:", e)
            return f"<h3>❌ Error: {e}</h3>", 500

    return render_template('index.html')
