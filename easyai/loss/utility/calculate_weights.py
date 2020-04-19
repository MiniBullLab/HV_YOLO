#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from sklearn.utils import compute_class_weight


def numpy_compute_weight(labels):
    # compute class weights
    cls_weight_list = []
    batch_size = labels.shape[0]
    for i in range(batch_size):
        y = labels[i].reshape(-1)
        lb = np.unique(y)  # 0., 1
        cls_weight = compute_class_weight('balanced', lb, y)
        cls_weight_list.append(cls_weight)
    del y
    cls_weight_list = np.asarray(cls_weight_list)
    result = np.sum(cls_weight_list, axis=0) / batch_size
    # print(result)
    return result


def enet_weighing(labels, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    for label in labels:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    class_count = class_count[:num_classes]
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights
