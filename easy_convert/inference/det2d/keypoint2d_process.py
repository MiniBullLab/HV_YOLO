#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


class KeyPoint2dProcess():
    def __init__(self, image_size, thresh_conf, num_classes, point_count=9):
        self.image_size = image_size
        self.thresh_conf = thresh_conf
        self.num_classes = num_classes
        self.point_count = point_count
        self.loc_count = point_count * 2
        self.only_objectness = 1

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def softmax(self, x):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        s = x_exp / x_sum
        return s

    def postprocess(self, output):
        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = self.get_keypoint_box(output)
        corners2D = self.resize_keypoint_objects(all_boxes)
        return corners2D

    def resize_keypoint_objects(self, boxes):
        best_conf_est = -1
        # If the prediction has the highest confidence,
        # choose it as our prediction for single object pose estimation
        for j in range(len(boxes)):
            if (boxes[j][18] > best_conf_est):
                box_pr = boxes[j]
                best_conf_est = boxes[j][18]

        corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.image_size[0]
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.image_size[1]
        return corners2D_pr

    def get_keypoint_box(self, output):
        N, C, H, W = output.shape
        pred_boxes, confs, cls = self.get_region_result(output)

        cls_max_confs = np.max(cls, 1)
        cls_max_ids = np.argmax(cls, 1)

        # Boxes filter
        boxes = []
        max_conf = -1
        for cy in range(H):
            for cx in range(W):
                ind = cy * W + cx
                det_conf = confs[ind]
                if self.only_objectness:
                    conf = confs[ind]
                else:
                    conf = confs[ind] * cls_max_confs[ind]

                if conf > max_conf:
                    max_conf = conf
                    max_ind = ind
                if conf > self.thresh_conf:
                    box = []
                    for j in range(0, self.point_count):
                        box.append(pred_boxes[ind, 2 * j] / W)
                        box.append(pred_boxes[ind, 2 * j + 1] / H)
                    cls_max_conf = cls_max_confs[ind]
                    cls_max_id = cls_max_ids[ind]
                    box.append(det_conf)
                    box.append(cls_max_conf)
                    box.append(cls_max_id)
                    boxes.append(box)

        if len(boxes) == 0:
            for j in range(0, self.point_count):
                box.append(pred_boxes[max_ind, 2 * j] / W)
                box.append(pred_boxes[max_ind, 2 * j + 1] / H)
            cls_max_conf = cls_max_confs[max_ind]
            cls_max_id = cls_max_ids[max_ind]
            box.append(det_conf)
            box.append(cls_max_conf)
            box.append(cls_max_id)
            boxes.append(box)
        return boxes

    def get_region_result(self, output):
        N, C, H, W = output.shape
        output = output.reshape(N, (self.loc_count + 1 + self.num_classes),
                              H * W)
        output = np.transpose(output, (0, 2, 1))

        x_point = []
        y_point = []
        x_point.append(self.sigmoid(output[:, :, 0]))
        y_point.append(self.sigmoid(output[:, :, 1]))
        for index in range(2, self.point_count*2, 2):
            x_point.append(output[:, :, index])
            y_point.append(output[:, :, index + 1])
        conf = self.sigmoid(output[:, :, self.loc_count]).reshape(-1, 1)
        cls = output[:, :, self.loc_count + 1:self.loc_count + 1 + self.num_classes]. \
            reshape(-1, self.num_classes)

        # Create pred boxes
        point_all_count = N * H * W
        pred_corners = np.zeros((point_all_count, 2*self.point_count), dtype='float')
        grid_x = np.linspace(0, W-1, W).reshape(1, W).repeat(H, 0).reshape(point_all_count)
        grid_y = np.linspace(0, H-1, H).reshape(1, H).repeat(W, 1).reshape(point_all_count)
        for i in range(0, self.point_count):
            pred_corners[:, 2 * i] = (x_point[i].reshape(point_all_count) + grid_x)
            pred_corners[:, 2 * i + 1] = (y_point[i].reshape(point_all_count) + grid_y)

        cls = self.softmax(cls)
        return pred_corners, conf, cls