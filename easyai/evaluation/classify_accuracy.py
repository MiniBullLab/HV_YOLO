#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.average_meter import AverageMeter


class ClassifyAccuracy():

    def __init__(self, top_k=(1, 5)):
        self.top1 = AverageMeter()
        self.topK = AverageMeter()
        self.param_top = top_k

    def torch_eval(self, output, target):
        precision = self.accuracy(output, target, self.param_top)
        batch_size = target.size(0)
        if len(precision) > 1:
            self.top1.update(precision[0].item(), batch_size)
            self.topK.update(precision[1].item(), batch_size)
        else:
            self.top1.update(precision[0].item(), batch_size)

    def eval(self, outputs, targets):
        pass
        # batch_size = (outputs.shape)[0]
        # maxk = max(self.param_top)
        # preds = -outputs
        # preds = preds.argsort()[:maxk]
        # targets = targets.repeat(preds.shape, axis=1)
        # correct = preds == targets
        # res = []
        # for k in self.param_top:
        #     correct_k = correct[:k].sum(0)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        # if len(res) > 1:
        #     self.top1.update(res[0], 1)
        #     self.topK.update(res[1], 1)
        # else:
        #     self.top1.update(res[0], 1)

    def clean_data(self):
        self.top1.reset()
        self.topK.reset()

    def get_top1(self):
        return self.top1.avg

    def get_topK(self):
        return self.topK.avg

    def accuracy(self, output, target, top=(1, 5)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(top)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
