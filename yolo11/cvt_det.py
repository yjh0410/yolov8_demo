import os
import cv2
import torch
import numpy as np
from model.yolo11_det import Yolo11Detector

model_scale = "l"
ckpt = torch.load(f'checkpoints/official/yolo11{model_scale}.pt', map_location="cpu")
yolo11 = ckpt.pop("model")
yolo11 = yolo11.float()

my_yolo11 = Yolo11Detector(model_scale=model_scale, num_classes=80).eval()
my_yolo11.load_state_dict(yolo11.state_dict()) 
torch.save({"model": my_yolo11.state_dict()}, os.path.join("checkpoints/yolo11{}_det_ckpt.pth".format(model_scale)))
my_yolo11.load_state_dict(torch.load(os.path.join("checkpoints/yolo11{}_det_ckpt.pth".format(model_scale)), map_location="cpu")["model"])

for file_name in os.listdir("data"):
    img_file = os.path.join("data", file_name)
    img = cv2.imread(img_file)
    img = cv2.resize(img, (640, 640))
    img_1 = img.copy()
    img_2 = img.copy()
    x = torch.as_tensor(img / 255.).permute(2, 0, 1).contiguous()
    x = x.unsqueeze(0).float()

    # ---------------- Official YOLO11 ----------------
    outputs, _ = yolo11(x)
    xywh_preds = outputs[0, :4, :].permute(1, 0)
    score_preds = outputs[0, 4:, :].permute(1, 0)

    for box, score in zip(xywh_preds, score_preds):
        cx, cy, bw, bh = box
        score = torch.max(score)
        
        if score > 0.2:
            cv2.rectangle(img_1,
                        (int(cx - 0.5 * bw), int(cy - 0.5 * bh)),
                        (int(cx + 0.5 * bw), int(cy + 0.5 * bh)),
                        [0, 0, 255],
                        )

    # ---------------- My YOLO11 ----------------
    outputs = my_yolo11(x)
    xyxy_preds  = outputs[0, :4, :].permute(1, 0)
    score_preds = outputs[0, 4:, :].permute(1, 0)

    for box, score in zip(xyxy_preds , score_preds):
        x1, y1, x2, y2 = box
        score = torch.max(score)
        
        if score > 0.2:
            cv2.rectangle(img_2,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        [0, 0, 255],
                        )

    cv2.imshow("detec", np.concatenate([img_1, img_2], axis=1))
    cv2.waitKey(0)
