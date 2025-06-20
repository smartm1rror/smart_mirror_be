# skin_evaluate.py
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. 폴더 경로만 바꿔준다
test_dir = r'..\data\skin_dataset\val'  # validation 폴더

model_path = 'mobilenet_skin_best.pth'
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

y_true = []
y_pred = []
y_probs = []
img_paths = []

start_time = time.time()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())
        batch_indices = list(range(len(img_paths), len(img_paths)+len(labels)))
        img_paths.extend([test_dataset.samples[idx][0] for idx in batch_indices])
end_time = time.time()

total_time = end_time - start_time
avg_time_per_image = total_time / len(test_dataset)
accuracy = np.mean(np.array(y_true) == np.array(y_pred))

print(f"\n== 평가 결과 ==")
print(f"전체 이미지 개수: {len(test_dataset)}")
print(f"정확도(Accuracy): {accuracy:.4f}")
print(f"전체 분석 시간: {total_time:.2f}초")
print(f"평균 추론 시간(1장): {avg_time_per_image:.3f}초")
print("\n== 클래스별 정밀 리포트 ==")
print(classification_report(y_true, y_pred, target_names=class_names))
print("\n== 혼동 행렬 (Confusion Matrix) ==")
print(confusion_matrix(y_true, y_pred))

with open('skin_eval_detail.txt', 'w', encoding='utf-8') as f:
    f.write("img_path\tlabel\tpred\tprob\n")
    for path, true, pred, probs in zip(img_paths, y_true, y_pred, y_probs):
        prob_pred = probs[pred]
        f.write(f"{path}\t{class_names[true]}\t{class_names[pred]}\t{prob_pred:.3f}\n")

print("\n(상세 결과: skin_eval_detail.txt 파일 참고)")
