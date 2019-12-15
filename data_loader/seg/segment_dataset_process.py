#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import numpy as np
from data_loader.utility.image_dataset_process import ImageDataSetProcess


class SegmentDatasetProcess():
    def __init__(self):
        self.dataset_process = ImageDataSetProcess()

    def normaliza_dataset(self, src_image):
        image = self.dataset_process.image_normaliza(src_image)
        image = self.dataset_process.image_transpose(image)
        return image

    def resize_dataset(self, src_image, image_size, label,
                       volid_label_seg=None, valid_label_seg=None):
        image, ratio, pad = self.dataset_process.image_resize_square(src_image,
                                                                     image_size,
                                                                     color=(127.5, 127.5, 127.5))
        target = self.encode_segmap(np.array(label, dtype=np.uint8),
                                    volid_label_seg, valid_label_seg)
        target, ratio, pad = self.dataset_process.image_resize_square(target,
                                                                      image_size,
                                                                      250)
        return image, target

    def change_label(self, label, valid_label_seg):
        valid_masks = np.zeros(label.shape)
        for l in range(0, len(valid_label_seg)):
            valid_mask = label == l  # set false to position of seg that not in valid_label_seg
            valid_masks += valid_mask  # set 0.0 to position of seg that not in valid_label_seg
        valid_masks[valid_masks == 0] = -1
        seg = np.float32(label) * valid_masks
        seg[seg < 0] = 250
        seg = np.uint8(seg)
        return seg

    def encode_segmap(self, mask, volid_label, valid_label):
        classes = -np.ones([100, 100])
        valid = [x for j in valid_label for x in j]
        for i in range(0, len(valid_label)):
            classes[i, :len(valid_label[i])] = valid_label[i]
        for label in volid_label:
            mask[mask == label] = 250
        for validc in valid:
            mask[mask == validc] = np.uint8(np.where(classes == validc)[0])

        return mask
