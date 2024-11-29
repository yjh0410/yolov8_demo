import cv2
import torch
import numpy as np


def val_transform(image, img_size=640):
    # --------------- Resize image ---------------
    orig_h, orig_w = image.shape[:2]
    ratio = img_size / max(orig_h, orig_w)
    if ratio != 1: 
        new_shape = (int(round(orig_w * ratio)), int(round(orig_h * ratio)))
        image = cv2.resize(image, new_shape)
    image = torch.as_tensor(image / 255.).permute(2, 0, 1).contiguous()

    # --------------- Pad Image ---------------
    img_h0, img_w0 = image.shape[1:]
    dh = img_h0 % 32
    dw = img_w0 % 32
    dh = dh if dh == 0 else 32 - dh
    dw = dw if dw == 0 else 32 - dw
    
    pad_img_h = img_h0 + dh
    pad_img_w = img_w0 + dw
    pad_image = torch.ones([image.size(0), pad_img_h, pad_img_w]).float() * 0.5
    pad_image[:, :img_h0, :img_w0] = image

    return pad_image, ratio

def post_process(cls_preds: torch.Tensor,
                 box_preds: torch.Tensor,
                 conf_thresh: float = 0.2,
                 nms_thresh: float = 0.45,
                 num_classes:int = 80,
                 ):
    """
    We process predictions at each scale hierarchically
    Input:
        cls_preds: torch.Tensor -> [B, M, C], B=1
        box_preds: torch.Tensor -> [B, M, 4], B=1
    Output:
        bboxes: np.array -> [N, 4]
        scores: np.array -> [N,]
        labels: np.array -> [N,]
    """
    
    # [M,]
    scores, labels = torch.max(cls_preds, dim=1)

    # topk candidates
    predicted_prob, topk_idxs = scores.sort(descending=True)

    # filter out the proposals with low confidence score
    keep_idxs = predicted_prob > conf_thresh
    scores = predicted_prob[keep_idxs]
    topk_idxs = topk_idxs[keep_idxs]

    labels = labels[topk_idxs]
    bboxes = box_preds[topk_idxs]

    # to cpu & numpy
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    bboxes = bboxes.cpu().numpy()

    # nms
    scores, labels, bboxes = multiclass_nms(
        scores, labels, bboxes, nms_thresh, num_classes)
    
    return bboxes, scores, labels

def rescale_bboxes(bboxes, origin_size, ratio):
    # rescale bboxes
    bboxes /= ratio

    # clip bboxes
    bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=origin_size[0])
    bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=origin_size[1])

    return bboxes


# --------------------- NMS ops ---------------------
def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int32)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1
    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)
