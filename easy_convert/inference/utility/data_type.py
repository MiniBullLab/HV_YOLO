#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


class Point():
    def __init__(self, x=0., y=0.):
        self.x = x
        self.y = y

    def __str__(self):
        return "({},{})".format(self.x, self.y)


class Rectangle():
    def __init__(self, posn, w, h):
        self.corner = posn
        self.width = w
        self.height = h

    def __str__(self):
        return "({0},{1},{2})".format(self.corner, self.width, self.height)

    def iou(self, rect):
        return self.intersection(rect) / self.union(rect)

    def intersection(self, rect):
        w = self.overlap(self.corner.x, self.width, rect.corner.x, rect.width)
        h = self.overlap(self.corner.y, self.height, rect.corner.y, rect.height)
        if w < 0 or h < 0:
            return 0
        area = w * h
        return area

    def union(self, rect):
        i = self.intersection(rect)
        u = self.width * self.height + rect.width * rect.height - i
        return u

    def overlap(self, x1, w1, x2, w2):
        l1 = x1 - w1 / 2
        l2 = x2 - w2 / 2
        left = l1 if l1 > l2 else l2
        r1 = x1 + w1 / 2
        r2 = x2 + w2 / 2
        right = r1 if r1 < r2 else r2
        return right - left


class ObjectBox():
    def __init__(self, rect, prob=np.zeros(100), objectness=-1):
        self.rect = rect
        self.prob = prob
        self.objectness = objectness

    def __str__(self):
        return "({0},{1},{2})".format(self.rect, self.prob, self.objectness)

    def iou(self, box2):
        return self.rect.iou(box2.rect)