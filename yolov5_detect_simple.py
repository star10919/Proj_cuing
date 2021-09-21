import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
 
   
SOURCE = 'data/images/haein.jpg'
WEIGHTS = 'yolov5s.pt'      # yolov5s, yolov5m, yolov5l, yolov5x
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25  # prediction이후 바운딩 박스를 조절하는 NMS(Non-Max-Suppresion)에 사용되는 threshold 값.
IOU_THRES = 0.45   # prediction이후 바운딩 박스를 조절하는 NMS(Non-Max-Suppresion)에 사용되는 threshold 값.
CLASSES = None # CLASSES는 분류 필터링을 할 경우 사용하고 AGNOSTIC_NMS는 Classification없이 물체의 바운딩 박스만을 찾고 싶을때 사용
AGNOSTIC_NMS = False

 
def detect():
    source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE

    # Initialize
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # CUDA에서 지원(device.type 과 'cpu'의 값이 같음을 확인 후 같으면 False 다르면 True)
    print('device:', device)

    # Load model
    model = attempt_load(weights, map_location=device)  # FP32 모델 로드(가중치?)
    stride = int(model.stride.max())  
    imgsz = check_img_size(imgsz, s=stride)  # 이미지 사이즈체크
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names     # hasattr: model의 존재 확인
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    '''
    디바이스 선택 후 모델을 로드하고 이미지 사이즈를 stride로 나눌 수 있는지 체크 ////////////////// stride : 커널 이동 보폭
    그 다음엔 class name을 설정해주고 각 클래스 별로 RGB 컬러를 랜덤으로 지정
    이후 torch zero Tensor를 생성하여 Inference
    '''



    # Load image
    img0 = cv2.imread(source)  # opencv2에서는 BGR체계로 불러와짐
    assert img0 is not None, 'Image Not Found ' + source # 아까 설정한 source에서 이미지를 읽고 이미지가 없을 경우 예외처리

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0] # letterbox를 이용해 패딩을 해줌

    # Convert 
    img = img[:, :, ::-1].transpose(2, 0, 1)  #BGR to RGB, to 3x416x416 pytorch의 경우 모델에 입력할 경우 채널차원이 맨 앞에 있어야해서 transpose를 적용
    img = np.ascontiguousarray(img)     # 메모리에 연속배열 반환

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)      # 차원 늘림
    # numpy array에서 torch Tensor형식으로 변환하고, torch 모델의 입력은 항상 배치형태로 받기 때문에 맨 앞에 차원을 하나 넣어줌
    # 최종적으로는 1 x 3 x IMG_COL x IMG_ROW의 사이즈가 출력

    # Inference
    t0 = time_synchronized()
    pred = model(img, augment=AUGMENT)[0] # model에 이미지를 입력하면 pred가 출력됨
    print('pred shape:', pred.shape) # pred의 형태는 torch.Size([1, 18900, 85])출력

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
    # Yolo 모델은 각 이미지를 그리드 셀로 나누어 바운딩 박스들의 위치와 Confidence, Class 확률정보 출력

    # Process detections
    det = pred[0]
    print('det shape:', det.shape)
    
    '''
    그리드 셀에서 18900개의 바운딩 박스를 예측 확인
    pred를 직접 출력하면 index 0~3은 바운딩 박스의 위치, index 4는 바운딩 박스의 Confidence Score, 나머지 80개는 클래스들의 확률을 나타냄
    '''




    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string

    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

  

        print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')

    # det에서 나온 xy좌표들을 img0 사이즈에 맞게 리스케일링 해주고 img0 이미지에 예측한 클래스들과 정확도 등을 plot_one_box 함수를 이용하여 그려줌

    # Stream results
    print(s)
    cv2.imshow(source, img0)
    cv2.waitKey(0)  # 1 millisecond /////////waitkey에 대해서 알아보기
    # 마지막으로 opencv를 이용해 예측된 이미지 출력


if __name__ == '__main__':
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
            detect()