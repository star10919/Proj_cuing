import tensorflow as tf
from tensorflow.python import saved_model
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np

# kite/person(실습)

MODEL_PATH = 'C:/Users/bit/yolov4/tensorflow-yolov4-tflite/checkpoints/yolov4-416'
IOU_THRESHOLD = 0.45        # 정답과 예측값이 얼마나 겹치는지를 나타내는 지표
SCORE_THRESHOLD = 0.25      # 예측한 box안에 실제 object가 얼마나 존재하는지
INPUT_SIZE = 416

# load model
saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']        # model load

def main(img_path):     # 이미지 전처리
    img = cv2.imread(img_path)      # 이미지 읽어오기-cv2로 읽어오면 numpy타입으로 받아짐
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # 이미지 컬러 시스템 변경(BGR -> RGB)   opencv:BGR /  pillow:RGB

    img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))       # 이미지 크기 변경
    img_input = img_input / 255.
    img_input = img_input[np.newaxis, ...].astype(np.float32)   # newaxis : 차원 1개 추가
    img_input = tf.constant(img_input)      # numpy array를 tensor 상수로 바꿔줌

    pred_bbox = infer(img_input)        # 로드한 model에 이미지 넣어줌

    for key, value in pred_bbox.items():       # bounding box 후처리
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    #iou_threshold, score_threshold 넘는 부분 제거
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(  
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),    # [batch_size, num_boxes, q, 4]
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),    # [batch_size, num_boxes, num_classes]
        max_output_size_per_class=50,   # Tensor클래스당 비최대 억제에 의해 선택되는 최대 상자 수를 나타내는 정수 스칼라
        max_total_size=50,  # 	모든 클래스에 대해 유지되는 최대 상자 수를 나타내는 int32 스칼라입니다. 이 값을 크게 설정하면 시스템 작업 부하에 따라 OOM 오류가 발생할 수 있습니다.
        iou_threshold=IOU_THRESHOLD,    # IOU에 대해 상자가 너무 많이 겹치는지 여부를 결정하기 위한 임계값
        score_threshold=SCORE_THRESHOLD)   # 점수에 따라 상자를 제거할 시기를 결정하기 위한 임계값

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    result = utils.draw_bbox(img, pred_bbox)        # 결과값을 bounding box로 그림

    result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)  # BGR에서 RGB로 변환
    cv2.imwrite('result.png', result)   # opencv로 저장하려면 윗줄 과정 필요
    

if __name__ == '__main__':
    img_path = 'C:/Users/bit/yolov4/tensorflow-yolov4-tflite/data/kite.jpg'
    main(img_path)

