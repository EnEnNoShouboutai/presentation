import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# yolov5-lite 라이브러리 import를 위해
# 디렉토리를 PYTHONPATH에 추가
sys.path.append(str(Path("yolov5-lite")))

# yolov5-lite에서 필요한 모듈 import
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# 설정
weights_path = "weights/best.pt" # 모델 가중치 경로
image_path = "images/test.jpg" # 테스트 이미지 경로
# 학습에 사용할 디바이스 선택
device = "cuda" if torch.cuda.is_available() else "cpu"
# 신뢰도 임계값 설정
conf_threshold = 0.4

# 1. 모델 로드
model = DetectMultiBackend(weights_path, device=device)
model.eval()

# 2. 이미지 불러오기 및 전처리
img0 = cv2.imread(image_path)  # BGR
img = letterbox(img0, new_shape=640)[0]
img = img.transpose((2, 0, 1))[::-1]  # BGR -> RGB -> CHW
img = torch.from_numpy(img).float().to(device) / 255.0
img = img.unsqueeze(0)

# 3. 추론
with torch.no_grad():
	pred = model(img)
	pred = non_max_suppression(pred, conf_threshold, 0.45)

# 4. 결과 표시
for det in pred:  # 감지 결과
	if det is not None and len(det):
		det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

		for *xyxy, conf, cls in det:
			label = f'Fire {conf:.2f}'

			cv2.rectangle(
				img0,
				(int(xyxy[0]), int(xyxy[1])),
				(int(xyxy[2]), int(xyxy[3])),
				(0, 0, 255),
				2
			)
			cv2.putText(
				img0,
				label,
				(int(xyxy[0]), int(xyxy[1]) - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.9,
				(0, 0, 255),
				2
			)