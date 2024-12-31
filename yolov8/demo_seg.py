import os
import cv2
import torch
import numpy as np

from model.yolov8_seg import Yolov8Segmentor
from utils import val_transform
from utils import post_process_seg, scale_bboxes, decode_masks


# COCO labels
coco_class_labels = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',  'traffic light',  'fire hydrant',  'stop sign',  'parking meter',  'bench',  'bird',  'cat',  'dog',  'horse',  'sheep',  'cow',  'elephant',  'bear',  'zebra',  'giraffe',  'backpack',  'umbrella',  'handbag',  'tie',  'suitcase',  'frisbee',  'skis',  'snowboard',  'sports ball',  'kite',  'baseball bat',  'baseball glove',  'skateboard',  'surfboard',  'tennis racket',  'bottle',  'wine glass',  'cup',  'fork',  'knife',  'spoon',  'bowl',  'banana',  'apple',  'sandwich',  'orange',  'broccoli',  'carrot',  'hot dog',  'pizza',  'donut',  'cake',  'chair',  'couch',  'potted plant',  'bed',  'dining table',  'toilet',  'tv',  'laptop',  'mouse',  'remote',  'keyboard',  'cell phone',  'microwave',  'oven',  'toaster',  'sink',  'refrigerator',  'book',  'clock',  'vase',  'scissors',  'teddy bear',  'hair drier',  'toothbrush')

# build YOLO model
model_scale = "n"
yolov8 = Yolov8Segmentor(model_scale=model_scale, num_classes=80, num_masks=32).eval()

# load ckpt
ckpt = torch.load("checkpoints/yolov8{}_seg_ckpt.pth".format(model_scale), map_location="cpu")
yolov8.load_state_dict(ckpt.pop("model")) 

# run the demo
for file_name in os.listdir("data"):
    img_file = os.path.join("data", file_name)
    img = cv2.imread(img_file)
    orig_img_h, orig_img_w = img.shape[:2]

    # data process
    x, ratio = val_transform(img, img_size=640)
    x = x.unsqueeze(0).float()

    # inference
    with torch.no_grad():
        outputs, protos = yolov8(x)
    xyxy_preds  = outputs[0, :4, :].permute(1, 0)   # [hw, 4]
    score_preds = outputs[0, 4:84, :].permute(1, 0) # [hw, 80]
    mask_preds  = outputs[0, 84:, :].permute(1, 0)  # [hw, mask_dim]
    proto_preds = protos[0].cpu().numpy()           # [mask_dim, h, w]

    # post-process
    bboxes, masks, scores, labels = post_process_seg(
        score_preds, xyxy_preds, mask_preds, conf_thresh=0.2, nms_thresh=0.5, num_classes=80)

    # rescale bboxes
    bboxes = scale_bboxes(bboxes, [orig_img_w, orig_img_h], ratio)

    # decode masks with the protos
    masks = decode_masks(proto_preds, masks, [orig_img_w, orig_img_h])

    # Color for beautiful visualization
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255))
                     for _ in range(80)]

    for box, mask, score, cls_idx in zip(bboxes, masks, scores, labels):
        x1, y1, x2, y2 = box
        label = coco_class_labels[cls_idx]

        # crop the mask according to the bbox
        crop_mask = np.zeros_like(mask)
        crop_mask[int(y1):int(y2), int(x1):int(x2)] = mask[int(y1):int(y2), int(x1):int(x2)]

        # threshold the masks
        crop_mask = (crop_mask > 0.5) * 1.0

        color = class_colors[cls_idx]
        mess = "{} | {:.2f}".format(label, score)

        # draw the bbox and the label info.
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, mess, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)

        # draw the instance mask
        crop_mask = np.repeat(crop_mask[..., None], 3, axis=-1)
        mask_rgb = crop_mask * color * 0.6

        inv_alph_mask = (1 - crop_mask * 0.6)

        img = (img * inv_alph_mask +  mask_rgb).astype(np.uint8)

    cv2.imshow("detec", img)
    cv2.waitKey(0)