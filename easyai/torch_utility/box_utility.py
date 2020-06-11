#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch


def torch_box2d_rect_corner(boxes):
    result = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                        boxes[:, :2] + boxes[:, 2:] / 2], 1)
    return result


def torch_rect_box_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.
    Args:
        boxes1 (torch.Tensor): sized [N,4]; [[xc, yc, w, h], ...]
        boxes2 (torch.Tensor): sized [N,4]; [[xc, yc, w, h], ...]
    Return:
      iou(tensor): sized [N,M].
    """
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions


def torch_corners_box2d_ious(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1(tensor): bounding boxes, sized [N,4]; [[xmin, ymin, xmax, ymax], ...]
      box2(tensor): bounding boxes, sized [M,4].
    Return:
      iou(tensor): sized [N,M].
    """
    if len(box1.size()) == 1:
        box1 = box1.unsqueeze(0)

    if len(box2.size()) == 1:
        box2 = box2.unsqueeze(0)

    N = box1.size(0)
    M = box2.size(0)

    # max(xmin, ymin).
    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
    )

    # min(xmax, ymax)
    rb = torch.min(
        box1[:, 2:4].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:4].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
    area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou



