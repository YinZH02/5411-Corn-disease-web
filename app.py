from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载你的模型（修改为自己的模型路径）
model = torch.load('plant_model.pth', map_location='cpu')
model.eval()

# 预测用的预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return 'No file uploaded', 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        image = Image.open(filepath)
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax().item()

        label = f"Predicted class: {predicted_class}"  # 这里你可以替换为标签名
        return render_template('result.html', label=label, image='/' + filepath)

    return render_template('index.html')
