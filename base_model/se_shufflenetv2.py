import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utils.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicUnit(nn.Module):
    def __init__(self, inplanes, outplanes, c_tag=0.5, BatchNorm=nn.BatchNorm2d, activation=nn.ReLU, SE=False, residual=False, groups=2, dilation=1):
        super(BasicUnit, self).__init__()
        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part
        self.conv1 = nn.Conv2d(self.right_part_in, self.right_part_out, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(self.right_part_out)
        self.conv2 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=3, padding=dilation, bias=False,
                               groups=self.right_part_out, dilation=dilation)
        self.bn2 = BatchNorm(self.right_part_out)
        self.conv3 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(self.right_part_out)
        self.activation = activation

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.residual = residual
        self.groups = groups
        self.SE = SE
        if self.SE:
            self.SELayer = SELayer(self.right_part_out, 2)  # TODO

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.conv1(right)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        if self.SE:
            out = self.SELayer(out)
        if self.residual and self.inplanes == self.outplanes:
            out += right

        return channel_shuffle(torch.cat((left, out), 1), self.groups)


class DownsampleUnit(nn.Module):
    def __init__(self, inplanes, out_channels, c_tag=0.5, BatchNorm=nn.BatchNorm2d, activation=nn.ReLU, groups=2):
        super(DownsampleUnit, self).__init__()
        mid_channels = out_channels // 2

        self.conv1r = nn.Conv2d(inplanes, mid_channels, kernel_size=1, bias=False)
        self.bn1r = BatchNorm(mid_channels)
        self.conv2r = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False, groups=mid_channels)
        self.bn2r = BatchNorm(mid_channels)
        self.conv3r = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3r = BatchNorm(mid_channels)

        self.conv1l = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)
        self.bn1l = BatchNorm(inplanes)
        self.conv2l = nn.Conv2d(inplanes, mid_channels, kernel_size=1, bias=False)
        self.bn2l = BatchNorm(mid_channels)
        self.activation = activation

        self.groups = groups
        self.inplanes = inplanes

    def forward(self, x):
        out_r = self.conv1r(x)
        out_r = self.bn1r(out_r)
        out_r = self.activation(out_r)

        out_r = self.conv2r(out_r)
        out_r = self.bn2r(out_r)

        out_r = self.conv3r(out_r)
        out_r = self.bn3r(out_r)
        out_r = self.activation(out_r)

        out_l = self.conv1l(x)
        out_l = self.bn1l(out_l)

        out_l = self.conv2l(out_l)
        out_l = self.bn2l(out_l)
        out_l = self.activation(out_l)

        return channel_shuffle(torch.cat((out_r, out_l), 1), self.groups)


class SEShuffleNetV2(nn.Module):
    """SEShuffleNetV2 implementation.
    """

    def __init__(self, scale=1.0, in_channels=3, c_tag=0.5, BatchNorm=nn.BatchNorm2d, activation=nn.ReLU,
                 SE=False, residual=False, groups=2, dilationType=0):
        """
        SEShuffleNetV2 constructor
        :param scale:
        :param in_channels:
        :param c_tag:
        :param activation:
        :param SE:
        :param residual:
        :param groups:
        :param dilationType: diltation
        """

        super(SEShuffleNetV2, self).__init__()

        self.scale = scale
        self.c_tag = c_tag
        self.residual = residual
        self.SE = SE
        self.groups = groups

        self.activation_type = activation
        self.activation = activation
        self.BatchNorm = BatchNorm
        self.dilationType = dilationType

        self.num_of_channels = {0.5: [24, 48, 96, 192, 1024], 1: [24, 116, 232, 464, 1024],
                                1.5: [24, 176, 352, 704, 1024], 2: [24, 244, 488, 976, 2048]}
        self.c = [_make_divisible(chan, groups) for chan in self.num_of_channels[scale]]
        self.n = [3, 7, 3]  # TODO: should be [3,7,3]
        self.num_of_dilation = {0: [1, 1, 1], 1: [1, 2, 4]}
        self.d = self.num_of_dilation[dilationType]  # TODO: dilation should be [1,2,4]
        self.out_channels = self.c[:-1]

        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=2, padding=1)
        self.bn1 = BatchNorm(self.c[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[0], d=self.d[0],
                                  stage=0)
        self.layer2 = self._make_stage(inplanes=self.c[1], outplanes=self.c[2], n=self.n[1], d=self.d[1],
                                  stage=1)
        self.layer3 = self._make_stage(inplanes=self.c[2], outplanes=self.c[3], n=self.n[2], d=self.d[2],
                                  stage=2)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, self.BatchNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, d, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit{}".format(stage)

        # First module is the only one utilizing stride
        first_module = DownsampleUnit(inplanes=inplanes, out_channels=outplanes, BatchNorm=self.BatchNorm, activation=self.activation_type, c_tag=self.c_tag,
                                      groups=self.groups)
        modules["DownsampleUnit"] = first_module
        for i in range(n):
            name = stage_name + "_{}".format(i)
            module = BasicUnit(inplanes=outplanes, outplanes=outplanes, BatchNorm=self.BatchNorm, activation=self.activation_type,
                               c_tag=self.c_tag, SE=self.SE, residual=self.residual, groups=self.groups, dilation=d)
            modules[name] = module

        return nn.Sequential(modules)

    # def _make_shuffles(self):
    #     shuffleModules = []
    #     # modules = OrderedDict()
    #     stage_name = "ShuffleConvs"
    #
    #     for i in range(len(self.c) - 2):
    #         name = stage_name + "_{}".format(i)
    #         module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i], d=self.d[i],
    #                                       stage=i)
    #         # modules[name] = module
    #         shuffleModules.append(module)
    #
    #     return shuffleModules

    def forward(self, x):
        blocks = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        blocks.append(x)
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)

        return blocks


def test():
    """Testing
    """

    model2 = SEShuffleNetV2(scale=1, in_channels=3, c_tag=0.5, BatchNorm=nn.BatchNorm2d, activation=nn.LeakyReLU(0.1),
                          SE=True, residual=False, dilationType=1)
    print(model2)
    x = torch.randn(3, 3, 640, 352)
    y = model2(x)
    for e_y in y:
        print(e_y.shape)

# test()
