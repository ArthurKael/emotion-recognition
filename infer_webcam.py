import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms


MODEL_PATH = "models/emotion_resnet18_gray.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 96

EMA_ALPHA = 0.25
CONF_THRESHOLD = 0.35
MARGIN_THRESHOLD = 0.10


def build_resnet18_grayscale(num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]

model = build_resnet18_grayscale(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được webcam. Thử đổi cv2.VideoCapture(0) thành 1 hoặc 2.")
    exit()

ema_probs = None

print("Nhấn Q để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame từ webcam.")
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )

    if len(faces) > 0:
        largest_face = max(faces, key=lambda box: box[2] * box[3])
        x, y, w, h = largest_face
        face = gray_frame[y:y+h, x:x+w]

        try:
            face_input = transform(face).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(face_input)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            if ema_probs is None:
                ema_probs = probs
            else:
                ema_probs = EMA_ALPHA * probs + (1 - EMA_ALPHA) * ema_probs

            sorted_idx = np.argsort(ema_probs)[::-1]
            top1_idx = sorted_idx[0]
            top2_idx = sorted_idx[1]

            top1_conf = float(ema_probs[top1_idx])
            top2_conf = float(ema_probs[top2_idx])
            margin = top1_conf - top2_conf

            label = class_names[top1_idx]

            if top1_conf < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
                label_to_show = "uncertain"
                color = (0, 165, 255)
            else:
                label_to_show = label
                color = (0, 255, 0)

            text = f"{label_to_show}: {top1_conf * 100:.1f}%"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
            cv2.putText(
                frame,
                text,
                (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            info1 = f"1) {class_names[top1_idx]}: {top1_conf * 100:.1f}%"
            info2 = f"2) {class_names[top2_idx]}: {top2_conf * 100:.1f}%"
            info3 = f"margin: {margin * 100:.1f}%"

            cv2.putText(frame, info1, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, info2, (x, y + h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, info3, (x, y + h + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        except Exception as e:
            print("Lỗi xử lý khuôn mặt:", e)

    else:
        ema_probs = None

    cv2.imshow("Emotion Recognition - ResNet18 Gray + EMA + Margin", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()