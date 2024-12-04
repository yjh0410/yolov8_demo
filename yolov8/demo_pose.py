import os
import cv2
import torch
import numpy as np

from model.yolov8_pose import Yolov8Pose
from utils import post_process_pose, val_transform, scale_bboxes, scale_keypoints


# build YOLO model
model_scale = "l"
yolov8 = Yolov8Pose(model_scale=model_scale, num_classes=1).eval()

# load ckpt
ckpt = torch.load("checkpoints/yolov8{}_pose_ckpt.pth".format(model_scale), map_location="cpu")
yolov8.load_state_dict(ckpt.pop("model"))

num_kpt = yolov8.num_kepoints
num_kdim = yolov8.num_kptdims

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
    score_preds = outputs[0, 4:5, :].permute(1, 0)
    pose_preds  = outputs[0, 5:, :].permute(1, 0)

    # post-process
    bboxes, poses, scores, labels = post_process_pose(
        score_preds, xyxy_preds, pose_preds, conf_thresh=0.2, nms_thresh=0.5, num_classes=80)

    # rescale bboxes
    bboxes = scale_bboxes(bboxes, [img_w, img_h], ratio)
    poses = scale_keypoints(poses, [img_w, img_h], ratio, kdim=num_kdim)

    # Color for beautiful visualization
    np.random.seed(0)
    instance_colors = [(np.random.randint(255),
                        np.random.randint(255),
                        np.random.randint(255))
                        for _ in range(100)]

    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        score = scores[i]
        color = instance_colors[i]
        pose = poses[i].reshape(num_kpt, num_kdim)

        if score > 0.2:
            # draw bbox of the person
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # draw the keypoint
            for j in range(pose.shape[0]):
                point_position = (int(pose[j, 0]), int(pose[j, 1]))
                radius = 2
                vis = pose[j, 2]
                if vis > 0.5:
                    cv2.circle(img, point_position, radius, color, 2)

    cv2.imshow("detec", img)
    cv2.waitKey(0)
