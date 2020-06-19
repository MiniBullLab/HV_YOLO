#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.data_loader.det2d.det2d_data_augment import DetectionDataAugment
from easyai.data_loader.seg.segment_data_augment import SegmentDataAugment
from easyai.data_loader.utility.image_data_augment import ImageDataAugment


class Det2dSegDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.is_lr_flip = True
        self.image_augment = ImageDataAugment()
        self.det2d_augment = DetectionDataAugment()
        self.seg_augment = SegmentDataAugment()

    def augment(self, image_rgb, boxes_label, segment_label):
        image_size = (image_rgb.shape[1], image_rgb.shape[0])  # [width, height]
        image = image_rgb[:]
        boxes = boxes_label[:]
        segment_image = segment_label[:]
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            image, matrix, degree = self.image_augment.augment_affine(image)
            boxes = self.det2d_augment.labels_affine(boxes, matrix, degree, image_size)
            segment_image = self.seg_augment.label_affine(segment_image, matrix)
        if self.is_lr_flip:
            image, is_lr = self.image_augment.augment_lr_flip(image)
            boxes = self.det2d_augment.labels_lr_flip(boxes, is_lr, image_size)
            segment_image = self.seg_augment.label_lr_flip(segment_image, is_lr)
        return image, boxes, segment_image
