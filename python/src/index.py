import torch # PyTorch: 딥러닝 모델을 실행할 프레임워크
import cv2 # OpenCV: 이미지 처리 및 시각화를 위한 라이브러리
from pathlib import Path
import sys

# yolov5-lite 폴더 안의 기능들을을 사용하기 위해 경로 추가
# 모델 구조가 별도의 디렉토리에 있으므로 이를 import할 수 있도록 함
sys.path.append(str(Path("yolov5-lite")))

# YOLOv5-Lite에서 제공하는 추론 관련 모듈 로드
from models.common import DetectMultiBackend # 다양한 백엔드에서 모델을 실행할 수 있도록 도와주는 클래스
from utils.general import non_max_suppression # 중복된 박스 제거하는 후처리 함수
from utils.datasets import letterbox # 이미지 비율을 유지한 채 크기를 모델 입력 크기로 조정하는 함수

# 기본 경로 및 설정 정의
weights = 'weights/best.pt' # 학습된 산불 감지용 모델 가중치 파일 경로
image_path = 'images/test.jpg' # 테스트용 이미지 경로
device = 'cpu' # 데모용이므로 GPU 없이 CPU 사용 (느리지만 설정 간단)
conf_thres = 0.4 # 감지 신뢰도 임계값 (0.4 이상만 표시)

# 모델 로드 및 초기화
model = DetectMultiBackend(weights, device=device) # 학습된 모델 불러오기
model.eval() # 추론 모드로 전환 (학습 관련 기능 비활성화)

# 이미지 불러오기
img0 = cv2.imread(image_path) # 이미지 파일을 OpenCV로 읽음 (기본은 BGR 채널 순서)

# 이미지 전처리: 모델 입력 크기(640x640)에 맞게 리사이징 및 패딩
img = letterbox(img0, 640)[0] # 비율을 유지하면서 크기 조정 (중간에 여백이 생길 수 있음)

# 이미지 포맷 변경: CHW 순서로 변경하고 정규화
img = img.transpose((2, 0, 1))[::-1] # 이미지 채널 순서 변경 (HWC → CHW), BGR → RGB
img = torch.from_numpy(img).float() / 255.0 # 넘파이 배열을 텐서로 바꾸고 [0, 1]로 정규화
img = img.unsqueeze(0).to(device) # 배치 차원 추가 → 모델 입력 형식 (1, 3, 640, 640)

# 모델에 이미지 입력 → 객체 감지 수행
with torch.no_grad(): # 추론 시에는 학습 관련 계산(역전파 등)을 비활성화하여 속도 향상
	preds = model(img) # 모델로부터 raw 예측 결과 얻기
	preds = non_max_suppression(preds, conf_thres, 0.45) # NMS(비최대 억제)로 중복 감지 박스 제거

# 감지 결과를 원본 이미지에 표시
for det in preds:
	if det is not None:
		for *box, conf, cls in det:
			# 박스 좌표 unpack (x1, y1, x2, y2): 감지된 객체 영역
			x1, y1, x2, y2 = map(int, box)  # float 좌표를 정수로 변환
			# 감지된 영역에 빨간색 사각형 그리기
			cv2.rectangle(
				img0,
				(x1, y1),
				(x2, y2),
				(0, 0, 255),
				2
			)
			# 사각형 위에 라벨 텍스트 표시 ("Fire")
			cv2.putText(
				img0,
				"Fire",
				(x1, y1 - 5),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.9,
				(0, 0, 255),
				2
			)