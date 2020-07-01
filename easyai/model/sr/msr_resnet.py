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
from easyai.model.base_block.sr.msr_resnet_block import ResidualBlockNoBN


class MSRResNet(BaseModel):

    def __init__(self, data_channel=1, upscale_factor=3):
        super().__init__()
        self.set_name(ModelName.MSRResNet)
        self.data_channel = data_channel
        self.upscale_factor = upscale_factor
        self.out_channel = 64
        self.num_block = 3
        self.activation_name = ActivationType.LeakyReLU
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        conv_first = ConvActivationBlock(in_channels=self.data_channel,
                                         out_channels=self.out_channel,
                                         kernel_size=3,
                                         padding=1,
                                         stride=1,
                                         bias=True,
                                         activationName=self.activation_name)
        self.add_block_list(conv_first.get_name(), conv_first, self.out_channel)

        self.make_layer(self.num_block, self.out_channel)

        last_conv = ConvActivationBlock(in_channels=self.out_channel,
                                        out_channels=32,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=True,
                                        activationName=ActivationType.ReLU)
        self.out_channel = 32
        self.add_block_list(last_conv.get_name(), last_conv, self.out_channel)

        self.out_channel = self.out_channel * self.upscale_factor ** 2
        upconv1 = nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1, bias=True)
        self.add_block_list(LayerType.Convolutional, upconv1, self.out_channel)

        pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        self.add_block_list(LayerType.PixelShuffle, pixel_shuffle, 1)

    def make_layer(self, number_block, in_channel):
        for _ in range(number_block):
            temp_block = ResidualBlockNoBN(in_channel)
            self.add_block_list(temp_block.get_name(), temp_block, in_channel)

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
