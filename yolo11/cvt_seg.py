import os
import cv2
import torch
import numpy as np
from model.yolo11_seg import Yolo11Seg

model_scale = "n"
ckpt = torch.load(f"checkpoints/official/yolo11{model_scale}-seg.pt", map_location="cpu")
yolo11 = ckpt.pop("model")
yolo11 = yolo11.float()

# print(yolo11)
# exit()

my_yolo11 = Yolo11Seg(model_scale=model_scale, num_classes=80, num_masks=32).eval()
my_yolo11.load_state_dict(yolo11.state_dict()) 
torch.save({"model": my_yolo11.state_dict()}, os.path.join("checkpoints/yolo11{}_seg_ckpt.pth".format(model_scale)))
my_yolo11.load_state_dict(torch.load(os.path.join("checkpoints/yolo11{}_seg_ckpt.pth".format(model_scale)), map_location="cpu")["model"])

for file_name in os.listdir("data"):
    img_file = os.path.join("data", file_name)
    img = cv2.imread(img_file)
    img = cv2.resize(img, (640, 640))
    img_1 = img.copy()
    img_2 = img.copy()
    x = torch.as_tensor(img / 255.).permute(2, 0, 1).contiguous()
    x = x.unsqueeze(0).float()

    # ---------------- Official YOLo11 ----------------
    yolo11.model[-1].export = True
    outputs, protos = yolo11(x)

    xywh_preds = outputs[0, :4, :].permute(1, 0)     # [bs, 4,  hw]
    score_preds = outputs[0, 4:84, :].permute(1, 0)  # [bs, 80, hw]
    mask_preds = outputs[0, 84:, :].permute(1, 0)    # [bs, 32, hw]

    for box, score in zip(xywh_preds, score_preds):
        cx, cy, bw, bh = box
        score = torch.max(score)
        
        if score > 0.2:
            cv2.rectangle(img_1,
                        (int(cx - 0.5 * bw), int(cy - 0.5 * bh)),
                        (int(cx + 0.5 * bw), int(cy + 0.5 * bh)),
                        [0, 0, 255],
                        )

    # ---------------- My YOLo11 ----------------
    outputs, protos = my_yolo11(x)
    xyxy_preds = outputs[0, :4, :].permute(1, 0)     # [bs, 4,  hw]
    score_preds = outputs[0, 4:84, :].permute(1, 0)  # [bs, 80, hw]
    mask_preds = outputs[0, 84:, :].permute(1, 0)    # [bs, 32, hw]

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