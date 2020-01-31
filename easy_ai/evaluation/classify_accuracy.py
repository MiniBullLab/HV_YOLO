#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easy_ai.helper.average_meter import AverageMeter


class ClassifyAccuracy():

    def __init__(self):
        self.top1 = AverageMeter()
        self.topK = AverageMeter()

    def compute_topk(self, output, target, top=(1, 5)):
        precision = self.accuracy(output, target, top)
        if len(precision) > 1:
            self.top1.update(precision[0], 1)
            self.topK.update(precision[1], 1)
        else:
            self.top1.update(precision[0], 1)

    def get_top1(self):
        return self.top1.avg

    def get_topK(self):
        return self.topK.avg

    def accuracy(self, output, target, top=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(top)
        batch_size = target.size(0)
        _, pred = output.float().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
