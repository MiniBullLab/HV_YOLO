#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class LossType():

    CrossEntropy2d = "crossEntropy2d"
    BinaryCrossEntropy2d = "bceLoss"
    MeanSquaredErrorLoss = "mseLoss"

    SmoothCrossEntropy = "smoothCrossEntropy"
    FocalLoss = "focalLoss"
    FocalBinaryLoss = "focalBinaryLoss"

    # det2d
    Region2dLoss = "Region2dLoss"
    YoloV3Loss = "YoloV3Loss"
    MultiBoxLoss = "MultiBoxLoss"
    KeyPoints2dRegionLoss = "KeyPoints2dRegionLoss"

    # det3d

    # seg
    EncNetLoss = "encNetLoss"
    OhemCrossEntropy2d = "ohemCrossEntropy2d"
