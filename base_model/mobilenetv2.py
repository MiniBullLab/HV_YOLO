# MobileNetV2 in PyTorch
import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
# from base_block.utilityBlock import InvertedResidual
from .baseModelName import BaseModelName
from .baseModel import *

from base_block.blockName import BatchNormType, ActivationType, BlockType, LossType
from base_block.utilityBlock import ConvBNActivationBlock, ConvActivationBlock
from base_block.activationFunction import ActivationFunction

__all__ = ['mobilenet_v2_1_0']

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1, dilation=1,
                  bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU6, **kwargs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            convBNReLU1 = ConvBNActivationBlock(in_channels=in_channels,
                                                out_channels=inter_channels,
                                                kernel_size=1,
                                                bnName=bnName,
                                                activationName=activationName)
            layers.append(convBNReLU1)
        layers.extend([
            # dw
            ConvBNActivationBlock(in_channels = inter_channels,
                                  out_channels = inter_channels,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=dilation,
                                  dilation=dilation,
                                  groups = inter_channels,
                                  bnName=bnName,
                                  activationName=activationName),
            # pw-linear
            ConvBNActivationBlock(in_channels=inter_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  bnName=bnName,
                                  activationName=ActivationType.Linear)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(BaseModel):
    def __init__(self, data_channel=3, num_blocks=[1,2,3,4,3,3,1], out_channels=[16,24,32,64,96,160,320], stride=[1,2,2,2,1,2,1],
                 dilation=[1,1,1,1,1,1,1], bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU6, expand_ratios=[1,6,6,6,6,6,6], **kwargss):
        super().__init__()
        self.setModelName(BaseModelName.MobileNetV2)
        self.data_channel = data_channel
        self.num_blocks = num_blocks # [3, 7, 3]
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.activationName = activationName
        self.bnName = bnName
        self.expand_ratios = expand_ratios

        self.layer1 = ConvBNActivationBlock(in_channels=data_channel,
                                            out_channels=self.out_channels[2],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bnName=bnName,
                                            activationName=activationName)

        self.in_channels = self.out_channels[2]
        self.layer2 = self._make_layer(self.out_channels[0], self.num_blocks[0], self.stride[0], self.dilation[0],
                                       bnName, activationName, self.expand_ratios[0])

        self.layer3 = self._make_layer(self.out_channels[1], self.num_blocks[1], self.stride[1], self.dilation[1],
                                       bnName, activationName, self.expand_ratios[1])

        self.layer4 = self._make_layer(self.out_channels[2], self.num_blocks[2], self.stride[2], self.dilation[2],
                                       bnName, activationName, self.expand_ratios[2])

        self.layer5 = nn.Sequential(self._make_layer(self.out_channels[3], self.num_blocks[3], self.stride[3], self.dilation[3],
                                       bnName, activationName, self.expand_ratios[3]),
                                    self._make_layer(self.out_channels[4], self.num_blocks[4], self.stride[4], self.dilation[4],
                                       bnName, activationName, self.expand_ratios[4]))

        self.layer6 = nn.Sequential(self._make_layer(self.out_channels[5], self.num_blocks[5], self.stride[5], self.dilation[5],
                                       bnName, activationName, self.expand_ratios[5]),
                                    self._make_layer(self.out_channels[6], self.num_blocks[6], self.stride[6], self.dilation[6],
                                       bnName, activationName, self.expand_ratios[6]))

        self._init_weight()

    def _make_layer(self, out_channels, num_blocks, stride, dilation, bnName, activationName, expand_ratio):
        layers = [InvertedResidual(self.in_channels, out_channels, stride=stride, expand_ratio=expand_ratio,
                                   dilation=dilation, bnName=bnName, activationName=activationName)]
        self.in_channels = out_channels
        for i in range(num_blocks - 1):
            layers.append(InvertedResidual(self.in_channels, out_channels, stride=1, expand_ratio=expand_ratio,
                                           dilation=1, bnName=bnName, activationName=activationName))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        blocks = []
        x = self.layer1(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)
        x = self.layer5(x)
        blocks.append(x)
        x = self.layer6(x)
        blocks.append(x)
        return blocks

    def _init_weight(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def get_mobilenet_v2(pretrained=False, **kwargs):
    model = MobileNetV2()

    if pretrained:
        raise ValueError("Not support pretrained")
    return model

def mobilenet_v2_1_0(**kwargs):
    return get_mobilenet_v2(**kwargs)


