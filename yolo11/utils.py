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

def post_process_det(cls_preds: torch.Tensor,
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
    scores, labels, bboxes, _ = multiclass_nms(
        scores, labels, bboxes, None, nms_thresh, num_classes)
    
    return bboxes, scores, labels

def post_process_seg(cls_preds: torch.Tensor,
                     box_preds: torch.Tensor,
                     mask_preds: torch.Tensor,
                     conf_thresh: float = 0.2,
                     nms_thresh: float = 0.45,
                     num_classes:int = 80,
                     ):
    """
    We process predictions at each scale hierarchically
    Input:
        cls_preds: torch.Tensor  -> [bs, m, nc], bs=1
        box_preds: torch.Tensor  -> [bs, m, 4],  bs=1
        mask_preds: torch.Tensor -> [bs, m, nm], bs=1
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
    masks = mask_preds[topk_idxs]

    # to cpu & numpy
    scores = scores.cpu().numpy()  # [N,]
    labels = labels.cpu().numpy()  # [N,]
    bboxes = bboxes.cpu().numpy()  # [N, 4]
    masks  = masks.cpu().numpy()   # [N, mask_dim]

    # nms
    scores, labels, bboxes, masks = multiclass_nms(
        scores, labels, bboxes, masks, nms_thresh, num_classes)
    
    return bboxes, masks, scores, labels

def post_process_pose(cls_preds: torch.Tensor,
                     box_preds: torch.Tensor,
                     pose_preds: torch.Tensor,
                     conf_thresh: float = 0.2,
                     nms_thresh: float = 0.45,
                     num_classes:int = 80,
                     ):
    """
    We process predictions at each scale hierarchically
    Input:
        cls_preds: torch.Tensor  -> [bs, m, nc], bs=1
        box_preds: torch.Tensor  -> [bs, m, 4],  bs=1
        pose_preds: torch.Tensor -> [bs, m, nm], bs=1
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
    poses = pose_preds[topk_idxs]

    # to cpu & numpy
    scores = scores.cpu().numpy()  # [N,]
    labels = labels.cpu().numpy()  # [N,]
    bboxes = bboxes.cpu().numpy()  # [N, 4]
    poses  = poses.cpu().numpy()   # [N, mask_dim]

    # nms
    scores, labels, bboxes, poses = multiclass_nms(
        scores, labels, bboxes, poses, nms_thresh, num_classes)
    
    return bboxes, poses, scores, labels

def stable_sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


# --------------------- bbox & mask ops ---------------------
def scale_bboxes(bboxes, origin_size, ratio):
    """
        Inputs:
        - params: bboxes - np.ndarray or torch.Tensor with shape of [n, 4].
        - params: origin_size - a list[w, h] of original image width and height.
        - params: ratio - scaling factor.
    """
    # rescale bboxes
    bboxes /= ratio

    # clip bboxes
    if isinstance(bboxes, np.ndarray):
        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=origin_size[0])
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=origin_size[1])
    elif isinstance(bboxes, torch.Tensor):
        bboxes[..., [0, 2]] = torch.clamp(bboxes[..., [0, 2]], min=0., max=origin_size[0])
        bboxes[..., [1, 3]] = torch.clamp(bboxes[..., [1, 3]], min=0., max=origin_size[1])

    return bboxes

def scale_masks(masks, origin_size):
    """
    Rescale segment masks to shape.

    Args:
        masks: torch.Tensor with shape of [n, h, w].
        origin_size a list[w, h] of original image width and height.
    """
    mh, mw = masks.shape[1:]

    # calculate the padding value
    gain = min(mw / origin_size[0], mh / origin_size[1])  # gain  = old / new
    pad_w = int(mw - origin_size[0] * gain)
    pad_h = int(mh - origin_size[1] * gain)

    # crop the masks with the padding
    masks = masks[:, :int(mh - pad_h + 0.1), :int(mw - pad_w + 0.1)]
    
    # resize the masks
    masks = np.transpose(masks, (1, 2, 0))  # [n, h, w] -> [h, w, n]
    masks = cv2.resize(masks, origin_size, interpolation=cv2.INTER_CUBIC)
    masks = np.transpose(masks, (2, 0, 1))  # [h, w, n] -> [n, h, w]

    return masks

def decode_masks(protos, masks_in, origin_size):
    """
        Inputs:
        - params: protos - np.ndarray with shape of [mask_dim, h, w].
        - params: masks_in - np.ndarray with shape of [n, mask_dim], n is number of masks after nms.
        - params: origin_size - a list[w, h] of original image width and height.
        - params: ratio - scaling factor.
    """
    c, mh, mw = protos.shape

    # decode masks: [n, mask_dim] x [mask_dim, hw] = [n, hw] -> [n, h, w]
    masks_out = stable_sigmoid(masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)

    # scale masks: [n, h, w] -> [n, oh, ow]
    masks_out = scale_masks(masks_out, origin_size)

    return masks_out

def scale_keypoints(poses, origin_size, ratio, kdim=3):
    """
        Inputs:
        - params: poses - np.ndarray or torch.Tensor with shape of [n, 4].
        - params: origin_size - a list[w, h] of original image width and height.
        - params: ratio - scaling factor.
    """
    # rescale poses
    poses /= ratio

    # clip poses
    if isinstance(poses, np.ndarray):
        poses[..., 0::kdim] = np.clip(poses[..., 0::kdim], a_min=0., a_max=origin_size[0])
        poses[..., 1::kdim] = np.clip(poses[..., 1::kdim], a_min=0., a_max=origin_size[1])
    elif isinstance(poses, torch.Tensor):
        poses[..., 0::kdim] = torch.clamp(poses[..., 0::kdim], min=0., max=origin_size[0])
        poses[..., 1::kdim] = torch.clamp(poses[..., 1::kdim], min=0., max=origin_size[1])

    return poses


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

def multiclass_nms_class_agnostic(scores, labels, bboxes, masks=None, nms_thresh=0.50):
    # nms
    keep = nms(bboxes, scores, nms_thresh)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    if masks is not None:
        masks = masks[keep]

    return scores, labels, bboxes, masks

def multiclass_nms_class_aware(scores, labels, bboxes, masks=None, nms_thresh=0.50, num_classes=80):
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

    if masks is not None:
        masks = masks[keep]

    return scores, labels, bboxes, masks

def multiclass_nms(scores, labels, bboxes, masks=None, nms_thresh=0.50, num_classes=80, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, masks, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, masks, nms_thresh, num_classes)
