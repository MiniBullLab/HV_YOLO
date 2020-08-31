#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.utility.utility_loss import MeanSquaredErrorLoss
from easyai.model.utility.base_model import *
from easyai.model.base_block.utility.utility_block import ConvActivationBlock
from easyai.model.utility.registry import REGISTERED_SR_MODEL


@REGISTERED_SR_MODEL.register_module(ModelName.SmallSRNet)
class SmallSRNet(BaseModel):
    def __init__(self, data_channel=1, upscale_factor=3):
        super().__init__(data_channel)
        self.set_name(ModelName.SmallSRNet)
        self.upscale_factor = upscale_factor
        self.activation_name = ActivationType.ReLU

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        conv1 = ConvActivationBlock(in_channels=self.data_channel,
                                    out_channels=64,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2),
                                    bias=True,
                                    activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 64)

        conv2 = ConvActivationBlock(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    bias=True,
                                    activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, 64)

        conv3 = ConvActivationBlock(in_channels=64,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    bias=True,
                                    activationName=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, 32)

        out_channel = self.data_channel * self.upscale_factor ** 2
        conv4 = nn.Conv2d(32, out_channel, (3, 3), (1, 1), (1, 1))
        self.add_block_list(LayerType.Convolutional, conv4, self.upscale_factor ** 2)

        pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        self.add_block_list(LayerType.PixelShuffle, pixel_shuffle, 1)

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = MeanSquaredErrorLoss()
        self.add_block_list(LossType.MeanSquaredErrorLoss, loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LossType.MeanSquaredErrorLoss in key:
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
            print(key, x.shape)
        return output


if __name__ == "__main__":
    pass
