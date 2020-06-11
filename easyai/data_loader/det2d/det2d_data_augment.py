#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import math
import numpy as np
from easyai.helper.dataType import Rect2D
from easyai.data_loader.utility.image_data_augment import ImageDataAugment


class DetectionDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.is_lr_flip = True
        self.is_up_flip = False
        self.image_augment = ImageDataAugment()

    def augment(self, image_rgb, labels):
        image = image_rgb[:]
        targets = labels[:]
        image_size = (image_rgb.shape[1], image_rgb.shape[0])
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            image, matrix, degree = self.image_augment.augment_affine(image)
            targets = self.labels_affine(targets, matrix, degree, image_size)
        if self.is_lr_flip:
            image, is_lr = self.image_augment.augment_lr_flip(image)
            targets = self.labels_lr_flip(targets, is_lr, image_size)
        if self.is_up_flip:
            image, is_up = self.image_augment.augment_up_flip(image)
            targets = self.labels_up_flip(targets, is_up, image_size)
        return image, targets

    def labels_affine(self, labels, matrix, degree, image_size):
        targets = []
        for object in labels:
            points = np.array(object.getVector())
            area0 = (points[2] - points[0]) * (points[3] - points[1])
            xy = np.ones((4, 3))
            # x1y1, x2y2, x1y2, x2y1
            xy[:, :2] = points[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)
            xy = np.squeeze((xy @ matrix.T)[:, :2].reshape(1, 8))

            # create new boxes
            x = xy[[0, 2, 4, 6]]
            y = xy[[1, 3, 5, 7]]
            xy = np.array([x.min(), y.min(), x.max(), y.max()])

            # apply angle-based reduction
            radians = degree * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[2] + xy[0]) / 2
            y = (xy[3] + xy[1]) / 2
            w = (xy[2] - xy[0]) * reduction
            h = (xy[3] - xy[1]) * reduction
            xy = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

            # reject warped points outside of image
            np.clip(xy, 0, image_size[0], out=xy)
            w = xy[2] - xy[0]
            h = xy[3] - xy[1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)
            if i:
                rect = Rect2D()
                rect.class_id = object.class_id
                rect.min_corner.x = xy[0]
                rect.min_corner.y = xy[1]
                rect.max_corner.x = xy[2]
                rect.max_corner.y = xy[3]
                targets.append(rect)
        return targets

    def labels_lr_flip(self, labels, is_lr, image_size):
        # left-right flip
        if is_lr:
            for object in labels:
                temp = object.min_corner.x
                object.min_corner.x = image_size[0] - object.max_corner.x
                object.max_corner.x = image_size[0] - temp
        return labels

    def labels_up_flip(self, labels, is_up, image_size):
        # up-down flip
        if is_up:
            for object in labels:
                temp = object.min_corner.y
                object.min_corner.y = image_size[1] - object.max_corner.y
                object.max_corner.y = image_size[1] - temp
        return labels
