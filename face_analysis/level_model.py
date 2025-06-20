# face_analysis/level_model.py

import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2

model_path = "mobilenet_skin_best.pth"  # 모델 파일 경로는 프로젝트 루트 기준
num_classes = 6  # 정상(0) + 1~5레벨 = 6개

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_acne_level(cv2_img):
    # cv2_img: OpenCV BGR 이미지
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence
