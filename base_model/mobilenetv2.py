# MobileNetV2 in PyTorch
import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from .baseModelName import BaseModelName
from .baseModel import *
from base_block.blockName import BatchNormType, ActivationType
from base_block.utilityBlock import ConvBNActivationBlock, InvertedResidual

__all__ = ['mobilenet_v2_1_0']

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

        self.outChannelList = []
        self.index = 0

        self.create_block_list()

    def create_block_list(self):
        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.out_channels[2],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.addBlockList(layer1.getModelName(), layer1, self.out_channels[2])

        self.in_channels = self.out_channels[2]
        self.make_mobile_layer(self.out_channels[0], self.num_blocks[0], self.stride[0], self.dilation[0],
                               self.bnName, self.activationName, self.expand_ratios[0])

        self.in_channels = self.out_channels[0]
        self.make_mobile_layer(self.out_channels[1], self.num_blocks[1], self.stride[1], self.dilation[1],
                               self.bnName, self.activationName, self.expand_ratios[1])

        self.in_channels = self.out_channels[1]
        self.make_mobile_layer(self.out_channels[2], self.num_blocks[2], self.stride[2], self.dilation[2],
                               self.bnName, self.activationName, self.expand_ratios[2])

        self.in_channels = self.out_channels[2]
        self.make_mobile_layer(self.out_channels[3], self.num_blocks[3], self.stride[3], self.dilation[3],
                               self.bnName, self.activationName, self.expand_ratios[3])

        self.in_channels = self.out_channels[3]
        self.make_mobile_layer(self.out_channels[4], self.num_blocks[4], self.stride[4], self.dilation[4],
                               self.bnName, self.activationName, self.expand_ratios[4])

        self.in_channels = self.out_channels[4]
        self.make_mobile_layer(self.out_channels[5], self.num_blocks[5], self.stride[5], self.dilation[5],
                               self.bnName, self.activationName, self.expand_ratios[5])

        self.in_channels = self.out_channels[5]
        self.make_mobile_layer(self.out_channels[6], self.num_blocks[6], self.stride[6], self.dilation[6],
                               self.bnName, self.activationName, self.expand_ratios[6])

    def make_mobile_layer(self, out_channels, num_blocks, stride, dilation,
                          bnName, activationName, expand_ratio):
        down_layers = InvertedResidual(self.in_channels, out_channels, stride=stride, expand_ratio=expand_ratio,
                                   dilation=dilation, bnName=bnName, activationName=activationName)
        name = "down_%s" % down_layers.getModelName()
        self.addBlockList(name, down_layers, out_channels)
        for _ in range(num_blocks - 1):
            layer = InvertedResidual(out_channels, out_channels, stride=1, expand_ratio=expand_ratio,
                                           dilation=1, bnName=bnName, activationName=activationName)
            self.addBlockList(layer.getModelName(), layer, out_channels)

    def addBlockList(self, blockName, block, out_channel):
        blockName = "base_%s_%d" % (blockName, self.index)
        self.add_module(blockName, block)
        self.outChannelList.append(out_channel)
        self.index += 1

    def getOutChannelList(self):
        return self.outChannelList

    def printBlockName(self):
        for key in self._modules.keys():
            print(key)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list

def get_mobilenet_v2(pretrained=False, **kwargs):
    model = MobileNetV2()

    if pretrained:
        raise ValueError("Not support pretrained")
    return model

def mobilenet_v2_1_0(**kwargs):
    return get_mobilenet_v2(**kwargs)


