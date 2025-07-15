from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# 上传目录
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 类别标签（与你训练时一致）
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# ✅ 加载轻量模型结构 + 权重（与你训练一致）
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 4)
model.load_state_dict(torch.load('plant_model.pth', map_location='cpu'))
model.eval()

# 图像预处理（和训练一致）
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
                return "No file uploaded", 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            image = Image.open(filepath).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                label = class_names[predicted.item()]

            return render_template('result.html', label=label, image='/' + filepath)

        except Exception as e:
            print("❌ 出现错误：", e)
            return f"<h3>❌ Error: {e}</h3>", 500

    return render_template('index.html')
