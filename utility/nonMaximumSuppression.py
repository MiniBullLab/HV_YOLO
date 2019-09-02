import os
import numpy as np
import torch

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou
    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
        c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
        return iou - (c_area - union_area) / c_area  # GIoU

    return iou

def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.45, nms_style="OR", useNewConf=False,
                        GIoU=False):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):

        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)

        if useNewConf:
            pred[:, 4] *= class_conf # improves mAP from 0.549 to 0.551

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        adIndex = np.argsort((-pred[:, 4]).cpu().numpy())
        pred = pred[adIndex]

        det_max = []
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # Non-maximum suppression
            if nms_style == 'OR':  # default

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:], GIoU=GIoU)  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:], GIoU=GIoU)  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold
            
            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc, GIoU=GIoU) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif nms_style == 'SOFT_Linear':
                while dc.shape[0]:
                    # Get detection with highest confidence and save as max detection
                    det_max.append(dc[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(dc) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = bbox_iou(det_max[-1], dc[1:], GIoU=GIoU)

                    # Remove detections with IoU >= NMS threshold
                    weight = (ious > nms_thres).type_as(dc) * (1 - ious) + (ious < nms_thres).type_as(dc)
                    dc[1:, 4] *= weight
                    dc = dc[1:]
                    dc = dc[dc[:, 4] > conf_thres]
                    # Stop if we're at the last detection
                    if len(dc) == 0:
                        break

                    _, conf_sort_index = torch.sort(dc[:, 4], descending=True)
                    dc = dc[conf_sort_index]

            elif nms_style == 'SOFT_Gaussian':
                sigma = 0.5
                while dc.shape[0]:
                    # Get detection with highest confidence and save as max detection
                    det_max.append(dc[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(dc) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = bbox_iou(det_max[-1], dc[1:], GIoU=GIoU)

                    # Remove detections with IoU >= NMS threshold
                    weight = np.exp( - (ious * ious) / sigma)
                    dc[1:, 4] *= weight.type_as(dc)
                    dc = dc[1:]
                    dc = dc[dc[:, 4] > conf_thres]
                    # Stop if we're at the last detection
                    if len(dc) == 0:
                        break

                    _, conf_sort_index = torch.sort(dc[:, 4], descending=True)
                    dc = dc[conf_sort_index]

        # mark 当框的重合度大于一定值时，被认为是同一个目标，那么较小置信度的目标会被抑制

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            detIndex = np.argsort((-det_max[:, 4]).cpu().numpy())
            output[image_i] = det_max[detIndex]  # sort

    return output