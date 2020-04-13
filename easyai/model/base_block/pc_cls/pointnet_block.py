#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from torch.autograd import Variable
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock1d
from easyai.model.base_block.utility.utility_block import FcBNActivationBlock


class PointNetBlockName():

    MaxPool1dBlock = "maxpool1d"
    STNBlock = "stn"
    TransformBlock = "transform"
    PointNetRouteBlock = "pointRoute"


class MaxPool1dBlock(BaseBlock):

    def __init__(self, input_channel):
        super().__init__(PointNetBlockName.MaxPool1dBlock)
        self.input_channle = input_channel

    def forward(self, x):
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.input_channle)
        return x


class STNBlock(BaseBlock):

    def __init__(self, input_channle, bn_name=NormalizationType.BatchNormalize1d,
                 activation_name = ActivationType.ReLU):
        super().__init__(PointNetBlockName.STNBlock)
        self.input_channle = input_channle
        self.conv1 = ConvBNActivationBlock1d(in_channels=input_channle,
                                             out_channels=64,
                                             kernel_size=1,
                                             bnName=bn_name,
                                             activationName=activation_name)
        self.conv2 = ConvBNActivationBlock1d(in_channels=64,
                                             out_channels=128,
                                             kernel_size=1,
                                             bnName=bn_name,
                                             activationName=activation_name)
        self.conv3 = ConvBNActivationBlock1d(in_channels=128,
                                             out_channels=1024,
                                             kernel_size=1,
                                             bnName=bn_name,
                                             activationName=activation_name)
        self.maxpool = MaxPool1dBlock(input_channel=1024)
        self.fc1 = FcBNActivationBlock(1024, 512,
                                       bnName=bn_name,
                                       activationName=activation_name)
        self.fc2 = FcBNActivationBlock(512, 256,
                                       bnName=bn_name,
                                       activationName=activation_name)
        self.fc3 = nn.Linear(256, input_channle * input_channle)

        one_value = np.eye(self.input_channle).flatten().astype(np.float32)
        self.count = self.input_channle * self.input_channle
        self.tensor_one = torch.from_numpy(one_value)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = Variable(self.tensor_one).view(1, self.count).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.input_channle, self.input_channle)
        return x


class TransformBlock(BaseBlock):

    def __init__(self, input_channle, bn_name=NormalizationType.BatchNormalize1d,
                 activation_name=ActivationType.ReLU):
        super().__init__(PointNetBlockName.TransformBlock)
        self.stn = STNBlock(input_channle=input_channle,
                            bn_name=bn_name,
                            activation_name=activation_name)

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        return x, trans


class PointNetRouteBlock(BaseBlock):

    def __init__(self, layer):
        super().__init__(PointNetBlockName.PointNetRouteBlock)
        self.layer = layer

    def forward(self, x, base_outputs):
        point_feat = base_outputs[self.layer]
        n_pts = point_feat.size()[2]
        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        x = torch.cat([x, point_feat], 1)
        return x
