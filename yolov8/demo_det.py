import os
import cv2
import torch
import numpy as np

from model.yolov8_det import Yolov8Detector
from utils import post_process_det, val_transform, scale_bboxes


# COCO labels
coco_class_labels = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',  'traffic light',  'fire hydrant',  'stop sign',  'parking meter',  'bench',  'bird',  'cat',  'dog',  'horse',  'sheep',  'cow',  'elephant',  'bear',  'zebra',  'giraffe',  'backpack',  'umbrella',  'handbag',  'tie',  'suitcase',  'frisbee',  'skis',  'snowboard',  'sports ball',  'kite',  'baseball bat',  'baseball glove',  'skateboard',  'surfboard',  'tennis racket',  'bottle',  'wine glass',  'cup',  'fork',  'knife',  'spoon',  'bowl',  'banana',  'apple',  'sandwich',  'orange',  'broccoli',  'carrot',  'hot dog',  'pizza',  'donut',  'cake',  'chair',  'couch',  'potted plant',  'bed',  'dining table',  'toilet',  'tv',  'laptop',  'mouse',  'remote',  'keyboard',  'cell phone',  'microwave',  'oven',  'toaster',  'sink',  'refrigerator',  'book',  'clock',  'vase',  'scissors',  'teddy bear',  'hair drier',  'toothbrush')

# build YOLO model
model_scale = "s"
yolov8 = Yolov8Detector(model_scale=model_scale, num_classes=80).eval()

# load ckpt
ckpt = torch.load("checkpoints/yolov8{}_det_ckpt.pth".format(model_scale), map_location="cpu")
yolov8.load_state_dict(ckpt.pop("model")) 

# run the demo
for file_name in os.listdir("data"):
    img_file = os.path.join("data", file_name)
    img = cv2.imread(img_file)
    img_h, img_w = img.shape[:2]

    # data process
    x, ratio = val_transform(img, img_size=640)
    x = x.unsqueeze(0).float()

    # inference
    with torch.no_grad():
        outputs = yolov8(x)
    xyxy_preds  = outputs[0, :4, :].permute(1, 0)
    score_preds = outputs[0, 4:, :].permute(1, 0)

    # post-process
    bboxes, scores, labels = post_process_det(
        score_preds, xyxy_preds, conf_thresh=0.2, nms_thresh=0.5, num_classes=80)

    # rescale bboxes
    bboxes = scale_bboxes(bboxes, [img_w, img_h], ratio)

    # Color for beautiful visualization
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255))
                     for _ in range(80)]

    for box, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = box
        color = class_colors[label]
        label = coco_class_labels[label]
        mess = "{} | {:.2f}".format(label, score)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, mess, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)

    cv2.imshow("detec", img)
    cv2.waitKey(0)
