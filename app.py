from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
import io
import base64

from model import BinaryClassificationModel

def load_model(model_path, device):
    model = BinaryClassificationModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_path = "dolphin_binary_classification.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, device)

# Ініціалізація Flask
app = Flask(__name__)

# Трансформації для зображень
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file part'
            return render_template('index.html', error=error)

        file = request.files['file']
        if file.filename == '':
            error = 'No selected file'
            return render_template('index.html', error=error)

        # Зчитування даних файлу
        try:
            file_data = file.read()
            img = Image.open(io.BytesIO(file_data)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            error = f"Image processing error: {str(e)}"
            return render_template('index.html', error=error)

        # Прогноз
        try:
            with torch.no_grad():
                outputs = model(img_tensor)
                confidence = outputs.item()  # Припускаємо, що вихід - скаляр
                prediction = 1 if confidence > 0.5 else 0
        except Exception as e:
            error = f"Prediction error: {str(e)}"
            return render_template('index.html', error=error)

        # Маппінг результату
        class_name = "Yes, this dolphin is special!" if prediction == 1 else "No, this dolphin is not special."

        # Кодування зображення в base64
        encoded_image = base64.b64encode(file_data).decode('utf-8')
        mime_type = file.content_type  # Наприклад, 'image/jpeg'

        return render_template('index.html', result=class_name, confidence=confidence, image_data=encoded_image, mime_type=mime_type)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
