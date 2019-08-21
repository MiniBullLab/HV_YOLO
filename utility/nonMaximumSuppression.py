import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import numpy as np
import torch
import torch.nn.functional as F

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        # cross-class NMS (experimental)
        cross_class_nms = False
        if cross_class_nms:
            a = pred.clone()
            _, indices = torch.sort(-a[:, 4], 0)  # sort best to worst
            a = a[indices]
            radius = 30  # area to search for cross-class ious
            for i in range(len(a)):
                if i >= len(a) - 1:
                    break

                close = (torch.abs(a[i, 0] - a[i + 1:, 0]) < radius) & (torch.abs(a[i, 1] - a[i + 1:, 1]) < radius)
                close = close.nonzero()

                if len(close) > 0:
                    close = close + i + 1
                    iou = bbox_iou(a[i:i + 1, :4], a[close.squeeze(), :4].reshape(-1, 4), x1y1x2y2=False)
                    bad = close[iou > nms_thres]

                    if len(bad) > 0:
                        mask = torch.ones(len(a)).type(torch.ByteTensor)
                        mask[bad] = 0
                        a = a[mask]
            pred = a

        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] = multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        class_prob, class_pred = torch.max(F.softmax(pred[:, 5:], 1), 1)

        v = ((pred[:, 4] > conf_thres) & (class_prob > .3))  # TODO examine arbitrary 0.3 thres here
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]
        class_prob = class_prob[v]
        class_pred = class_pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_prob, class_pred)
        detections = torch.cat((pred[:, :5], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda(prediction.device)

        nms_style = 'OR'  # 'AND' or 'OR' (classical)
        for c in unique_labels:
            # Get the detections with the particular class
            det_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(det_class[:, 4], descending=True)
            det_class = det_class[conf_sort_index]
            # Perform non-maximum suppression
            det_max = []

            if nms_style == 'OR':  # Classical NMS
                while det_class.shape[0]:
                    # Get detection with highest confidence and save as max detection
                    det_max.append(det_class[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(det_class) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = bbox_iou(det_max[-1], det_class[1:])

                    # Remove detections with IoU >= NMS threshold
                    det_class = det_class[1:][ious < nms_thres]

            elif nms_style == 'AND':  # 'AND'-style NMS: >=2 boxes must share commonality to pass, single boxes erased
                while det_class.shape[0]:
                    if len(det_class) == 1:
                        break

                    ious = bbox_iou(det_class[:1], det_class[1:])

                    if ious.max() > 0.5:
                        det_max.append(det_class[0].unsqueeze(0))

                    # Remove detections with IoU >= NMS threshold
                    det_class = det_class[1:][ious < nms_thres]

            elif nms_style == 'linear':
                while det_class.shape[0]:
                    # Get detection with highest confidence and save as max detection
                    det_max.append(det_class[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(det_class) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = bbox_iou(det_max[-1], det_class[1:])

                    # Remove detections with IoU >= NMS threshold
                    weight = (ious > nms_thres).type(torch.cuda.FloatTensor) * (1 - ious) + (ious < nms_thres).type(torch.cuda.FloatTensor)
                    det_class[1:, 4] *= weight
                    det_class = det_class[1:]
                    det_class = det_class[det_class[:, 4] > conf_thres]
                    # Stop if we're at the last detection
                    if len(det_class) == 0:
                        break

                    _, conf_sort_index = torch.sort(det_class[:, 4], descending=True)
                    det_class = det_class[conf_sort_index]

            elif nms_style == 'gaussian':
                sigma = 0.5
                while det_class.shape[0]:
                    # Get detection with highest confidence and save as max detection
                    det_max.append(det_class[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(det_class) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = bbox_iou(det_max[-1], det_class[1:])

                    # Remove detections with IoU >= NMS threshold
                    weight = np.exp( - (ious * ious) / sigma)
                    det_class[1:, 4] *= weight.type(torch.cuda.FloatTensor)
                    det_class = det_class[1:]
                    det_class = det_class[det_class[:, 4] > conf_thres]
                    # Stop if we're at the last detection
                    if len(det_class) == 0:
                        break

                    _, conf_sort_index = torch.sort(det_class[:, 4], descending=True)
                    det_class = det_class[conf_sort_index]

            if len(det_max) > 0:
                det_max = torch.cat(det_max).data
                # Add max detections to outputs
                output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output