#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


class SegmentionMetric():

    def __init__(self, num_class):
        self.number_class = num_class
        self.confusion_matrix = np.zeros((self.number_class, self.number_class))

    def reset(self):
        self.confusion_matrix = np.zeros((self.number_class, self.number_class))

    def eval(self, label_pred, label_gt):
        label_pred = label_pred.astype(label_gt.dtype)
        self.confusion_matrix += self.fast_hist(label_gt.flatten(), label_pred.flatten(), self.number_class)

    def batch_eval(self, label_preds, label_gts):
        label_preds = label_preds.astype(label_gts.dtype)
        for gt, pred in zip(label_gts, label_preds):
            self.confusion_matrix += self.fast_hist(gt.flatten(), pred.flatten(), self.number_class)

    def get_score(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.number_class), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu,}, cls_iu

    def fast_hist(self, gt, pred, n_class):
        mask = (pred >= 0) & (pred < n_class)
        hist = np.bincount(n_class * gt[mask].astype(int) +
                           pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

