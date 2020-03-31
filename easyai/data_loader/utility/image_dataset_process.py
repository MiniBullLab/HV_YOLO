#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
import numpy as np
import random
import math
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess


class ImageDataSetProcess(BaseDataSetProcess):

    def __init__(self):
        super().__init__()

    def image_normaliza(self, image):
        return image / 255.0

    def numpy_normaliza(self, input_data, mean, std):
        result = (input_data - mean) / std
        return result

    def numpy_transpose(self, images, dtype=np.float32):
        result = None
        if images is None:
            result = None
        elif images.ndim == 3:
            image = images.transpose(2, 0, 1)
            result = np.ascontiguousarray(image, dtype=dtype)
        elif images.ndim == 4:
            img_all = images.transpose(0, 3, 1, 2)
            result = np.ascontiguousarray(img_all, dtype=dtype)
        return result

    def image_resize(self, src_image, image_size):
        image = cv2.resize(src_image, image_size, interpolation=cv2.INTER_NEAREST)
        return image

    def image_resize_square(self, src_image, image_size, color=(0, 0, 0)): # src_image h, w, c
        shape = src_image.shape[:2]  # shape = [height, width]
        src_size = (shape[1], shape[0])
        ratio, pad_size = self.resize_square_size(src_size, image_size)
        new_shape = (round(shape[0] * ratio), round(shape[1] * ratio))
        top = pad_size[1] // 2
        bottom = pad_size[1] - (pad_size[1] // 2)
        left = pad_size[0] // 2
        right = pad_size[0] - (pad_size[0] // 2)
        image = cv2.resize(src_image, (new_shape[1], new_shape[0]),
                           interpolation=cv2.INTER_AREA)  # resized, no border
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
        return image, ratio, pad_size

    def resize_square_size(self, src_size, dst_size):
        # ratio  = old / new
        ratio = min(float(dst_size[0]) / src_size[0], float(dst_size[1]) / src_size[1])
        new_shape = (round(src_size[0] * ratio), round(src_size[1] * ratio))
        dw = dst_size[0] - new_shape[0]  # width padding
        dh = dst_size[1] - new_shape[1]  # height padding
        pad_size = (dw, dh)
        return ratio, pad_size

    def image_affine(self, src_image, matrix, border_value=250):
        result = None
        width = src_image.shape[1]
        height = src_image.shape[0]
        if src_image is not None:
            result = cv2.warpPerspective(src_image, matrix,
                                         dsize=(width, height),
                                         flags=cv2.INTER_LINEAR,
                                         borderValue=border_value)
        return result

    def affine_matrix(self, image_size, degrees=(-10, 10),
                      translate=(.1, .1), scale=(.9, 1.1),
                      shear=(-3, 3)):
        width = image_size[0]
        height = image_size[1]
        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[1] * width
        T[1, 2] = (random.random() * 2 - 1) * translate[0] * height

        # Shear
        S = np.eye(3)
        # x shear (deg)
        S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
        # y shear (deg)
        S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!
        return M, a
