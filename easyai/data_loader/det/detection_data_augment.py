#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import math
import numpy as np
import random
from easyai.helper.dataType import Rect2D
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.utility.image_data_augment import ImageDataAugment


class DetectionDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.is_lr_flip = True
        self.is_up_flip = False
        self.dataset_process = ImageDataSetProcess()
        self.image_augment = ImageDataAugment()

    def augment(self, image_rgb, labels):
        image = image_rgb[:]
        targets = labels[:]
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            image, targets = self.augment_affine(image, targets)
        if self.is_lr_flip:
            image, targets = self.augment_lr_flip(image, targets)
        if self.is_up_flip:
            image, targets = self.augment_up_flip(image, targets)
        return image, targets

    def augment_affine(self, src_image, labels):
        image_size = (src_image.shape[1], src_image.shape[0])
        matrix, degree = self.dataset_process.affine_matrix(image_size,
                                                    degrees=(-5, 5),
                                                    translate=(0.1, 0.1),
                                                    scale=(0.8, 1.1),
                                                    shear=(-3, 3))
        image = self.dataset_process.image_affine(src_image, matrix,
                                                  border_value=(127.5, 127.5, 127.5))
        # Return warped points also
        if labels is not None:
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
            return image, targets
        else:
            return image, None

    def augment_lr_flip(self, src_image, labels):
        # random left-right flip
        image_size = (src_image.shape[1], src_image.shape[0])
        image = src_image[:]
        if random.random() > 0.5:
            image = np.fliplr(image)
            for object in labels:
                temp = object.min_corner.x
                object.min_corner.x = image_size[0] - object.max_corner.x
                object.max_corner.x = image_size[0] - temp
        return image, labels

    def augment_up_flip(self, src_image, labels):
        # random up-down flip
        image_size = (src_image.shape[1], src_image.shape[0])
        image = src_image[:]
        if random.random() > 0.5:
            image = np.flipud(image)
            for object in labels:
                temp = object.min_corner.y
                object.min_corner.y = image_size[1] - object.max_corner.y
                object.max_corner.y = image_size[1] - temp
        return image, labels
