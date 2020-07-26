#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch


def detection_data_merge(batch_list):
    list_labels = []
    list_images = []
    for torch_image, torch_label in batch_list:
        list_images.append(torch_image)
        list_labels.append(torch_label)
    return torch.stack(list_images), list_labels
