#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import NormalizeLayer, ActivationLayer, MeanLayer
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class EncNetBlockName():

    SeperableConv2dNActivation = "SeperableConv2dNActivation"
    JPUBlock = "JPUBlock"
    Encoding = "Encoding"
    EncBlock = "EncBlock"
    FCNHeadBlock = "FCNHeadBlock"


class SeperableConv2dNActivation(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(EncNetBlockName.SeperableConv2dNActivation)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation, in_channels, bias=bias)
        self.norm_layer = NormalizeLayer(bn_name, in_channels)
        self.pointwise = ConvBNActivationBlock(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=1,
                                               bias=bias,
                                               bnName=bn_name,
                                               activationName=activation_name)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.norm_layer(x)
        x = self.pointwise(x)
        return x


class JPUBlock(BaseBlock):
    def __init__(self, layers, in_planes, width=512,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(EncNetBlockName.JPUBlock)
        self.layers = [int(x) for x in layers.split(',') if x]
        assert len(self.layers) == 4
        self.conv5 = ConvBNActivationBlock(in_channels=in_planes[-1],
                                           out_channels=width,
                                           kernel_size=3,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv4 = ConvBNActivationBlock(in_channels=in_planes[-2],
                                           out_channels=width,
                                           kernel_size=3,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv3 = ConvBNActivationBlock(in_channels=in_planes[-3],
                                           out_channels=width,
                                           kernel_size=3,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)

        self.up1 = Upsample(scale_factor=2, mode='bilinear')
        self.up2 = Upsample(scale_factor=4, mode='bilinear')

        self.dilation1 = SeperableConv2dNActivation(3 * width, width,
                                                    kernel_size=3,
                                                    padding=1,
                                                    dilation=1,
                                                    bias=False,
                                                    bn_name=bn_name,
                                                    activation_name=activation_name)
        self.dilation2 = SeperableConv2dNActivation(3 * width, width,
                                                    kernel_size=3,
                                                    padding=2,
                                                    dilation=2,
                                                    bias=False,
                                                    bn_name=bn_name,
                                                    activation_name=activation_name)
        self.dilation3 = SeperableConv2dNActivation(3 * width, width,
                                                    kernel_size=3,
                                                    padding=4,
                                                    dilation=4,
                                                    bias=False,
                                                    bn_name=bn_name,
                                                    activation_name=activation_name)
        self.dilation4 = SeperableConv2dNActivation(3*width, width,
                                                    kernel_size=3,
                                                    padding=8,
                                                    dilation=8,
                                                    bias=False,
                                                    bn_name=bn_name,
                                                    activation_name=activation_name)

    def forward(self, layer_outputs, base_outputs):
        # print(self.layers)
        inputs = [layer_outputs[i] if i < 0 else base_outputs[i] for i in self.layers]
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        feats[-2] = self.up1(feats[-2])
        feats[-3] = self.up2(feats[-3])
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat),
                          self.dilation3(feat), self.dilation4(feat)], dim=1)
        return feat


class Encoding(BaseBlock):
    def __init__(self, D, K):
        super().__init__(EncNetBlockName.Encoding)
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert (X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN -> BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW -> Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(self.scale_l2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = self.aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x' + str(self.D) + '=>' + str(self.K) + 'x' \
               + str(self.D) + ')'

    @staticmethod
    def scale_l2(X, C, S):
        S = S.view(1, 1, C.size(0), 1)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        SL = S * (X - C)
        SL = SL.pow(2).sum(3)
        return SL

    @staticmethod
    def aggregate(A, X, C):
        A = A.unsqueeze(3)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        E = A * (X - C)
        E = E.sum(1)
        return E


class EncBlock(BaseBlock):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(EncNetBlockName.EncBlock)
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            ConvBNActivationBlock(in_channels=in_channels,
                                  out_channels=in_channels,
                                  kernel_size=1,
                                  bias=False,
                                  bnName=bn_name,
                                  activationName=activation_name),
            Encoding(D=in_channels, K=ncodes),
            NormalizeLayer(NormalizationType.BatchNormalize1d, ncodes),
            ActivationLayer(activation_name),
            MeanLayer(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())

        self.activate = ActivationLayer(activation_name)

        if self.se_loss:
            self.se_layer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        out = self.activate(x + x * y)
        if self.se_loss:
            loss_out = self.se_layer(en)
            return out, loss_out
        else:
            return out, None


class FCNHeadBlock(BaseBlock):

    def __init__(self, in_channels, class_number, scale_factor,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(EncNetBlockName.FCNHeadBlock)
        inter_channels = in_channels // 4
        self.conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                           out_channels=inter_channels,
                                           kernel_size=3,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.dropout = nn.Dropout2d(0.1, False)
        self.conv2 = nn.Conv2d(inter_channels, class_number, 1)
        self.up = Upsample(scale_factor=scale_factor, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.up(x)
        return x
