#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.helper.dataType import Rect2D
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class KeyPoint2dDataSetProcess(BaseDataSetProcess):

    def __init__(self, points_count):
        super().__init__()
        self.points_count = points_count
        self.dataset_process = ImageDataSetProcess()
        self.image_pad_color = (0, 0, 0)

    def normaliza_dataset(self, src_image, labels=None, image_size=None):
        image = self.dataset_process.image_normaliza(src_image)
        image = self.dataset_process.numpy_transpose(image)
        result = None
        if labels is not None:
            result = np.zeros((len(labels), self.points_count), dtype=np.float32)
            for index, rect2d in enumerate(labels):
                temp_data = []
                class_id = rect2d.class_id
                temp_data.append(class_id)
                key_points = rect2d.get_key_points()
                for point in key_points:
                    x, y, = point.x, point.y
                    x /= image_size[0]
                    y /= image_size[1]
                    temp_data.append(x)
                    temp_data.append(y)
                result[index, :] = np.array(temp_data)
        return image, result

    def resize_dataset(self, src_image, image_size, boxes=None, class_name=None):
        labels = []
        image, ratio, pad = self.dataset_process.image_resize_square(src_image,
                                                                     image_size,
                                                                     color=self.image_pad_color)
        if boxes is not None:
            for box in boxes:
                if box.name in class_name:
                    rect2d = Rect2D()
                    rect2d.class_id = class_name.index(box.name)
                    key_points = box.get_key_points()
                    for point in key_points:
                        point.x = ratio * point.x + pad[0] // 2
                        point.y = ratio * point.y + pad[0] // 2
                        rect2d.add_key_points(point)
                    labels.append(rect2d)
        return image, labels

    def change_outside_labels(self, labels):
        # reject warped points outside of image (0.999 for the image boundary)
        for i, label in enumerate(labels):
            for index in range(1, self.points_count+1):
                if label[index] >= float(1):
                    label[index] = 0.999
        return labels