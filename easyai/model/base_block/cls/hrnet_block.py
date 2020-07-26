#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class HRNetBlockName():

    BasicBlock = "BasicBlock"
    Bottleneck = "Bottleneck"
    TransitionBlock = "TransitionBlock"
    HighResolutionBlock = "HighResolutionBlock"
    ClassificationHeadBlock = "ClassificationHeadBlock"


class BasicBlock(BaseBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(HRNetBlockName.BasicBlock)
        self.conv1 = ConvBNActivationBlock(in_channels=inplanes,
                                           out_channels=planes,
                                           kernel_size=3,
                                           stride=stride,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv2 = ConvBNActivationBlock(in_channels=planes,
                                           out_channels=planes,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=ActivationType.Linear)
        self.downsample = downsample
        self.activate = ActivationLayer(activation_name=activation_name)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activate(out)
        return out


class Bottleneck(BaseBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(HRNetBlockName.Bottleneck)
        self.conv1 = ConvBNActivationBlock(in_channels=inplanes,
                                           out_channels=planes,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv2 = ConvBNActivationBlock(in_channels=planes,
                                           out_channels=planes,
                                           kernel_size=3,
                                           stride=stride,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv3 = ConvBNActivationBlock(in_channels=planes,
                                           out_channels=planes * self.expansion,
                                           kernel_size=3,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=ActivationType.Linear)
        self.downsample = downsample
        self.activate = ActivationLayer(activation_name=activation_name)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activate(out)
        return out


class TransitionBlock(BaseBlock):

    def __init__(self, block_type, num_channels_pre_layer, num_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(HRNetBlockName.TransitionBlock)
        block = None
        if block_type == 0:
            block = BasicBlock
        elif block_type == 1:
            block = Bottleneck
        num_channels_cur_layer = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(ConvBNActivationBlock(in_channels=num_channels_pre_layer[i],
                                                                   out_channels=num_channels_cur_layer[i],
                                                                   kernel_size=3,
                                                                   stride=1,
                                                                   padding=1,
                                                                   bias=False,
                                                                   bnName=bn_name,
                                                                   activationName=activation_name))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(ConvBNActivationBlock(in_channels=inchannels,
                                                          out_channels=outchannels,
                                                          kernel_size=3,
                                                          stride=2,
                                                          padding=1,
                                                          bias=False,
                                                          bnName=bn_name,
                                                          activationName=activation_name))
                transition_layers.append(nn.Sequential(*conv3x3s))

        self.transition_layers = nn.ModuleList(transition_layers)

    def forward(self, input_list):
        x_list = []
        for index, layer in enumerate(self.transition_layers):
            if layer is not None:
                x_list.append(layer(input_list[-1]))
            elif len(input_list) == 1:
                x_list.append(input_list[-1])
            else:
                x_list.append(input_list[index])
        return x_list


class HighResolutionBlock(BaseBlock):
    def __init__(self, num_branches, block, num_blocks, num_in_channels,
                 num_channels, fuse_method, multi_scale_output=True,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(HRNetBlockName.HighResolutionBlock)
        self.check_branches(num_branches, num_blocks,
                            num_in_channels, num_channels)
        self.bn_name = bn_name
        self.activation_name = activation_name
        self.num_in_channels = num_in_channels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self.make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self.make_fuse_layers()
        self.activate = ActivationLayer(activation_name=self.activation_name)

    def check_branches(self, num_branches, num_blocks,
                       num_in_channels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_in_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_IN_CHANNELS({})'.format(
                num_branches, len(num_in_channels))
            raise ValueError(error_msg)

    def make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self.make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        ConvBNActivationBlock(in_channels=self.num_in_channels[j],
                                              out_channels=self.num_in_channels[i],
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              bias=False,
                                              bnName=self.bn_name,
                                              activationName=ActivationType.Linear),
                        Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            conv3x3 = ConvBNActivationBlock(in_channels=self.num_in_channels[j],
                                                            out_channels=self.num_in_channels[i],
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            bias=False,
                                                            bnName=self.bn_name,
                                                            activationName=ActivationType.Linear)
                            conv3x3s.append(conv3x3)
                        else:
                            conv3x3 = ConvBNActivationBlock(in_channels=self.num_in_channels[j],
                                                            out_channels=self.num_in_channels[j],
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            bias=False,
                                                            bnName=self.bn_name,
                                                            activationName=self.activation_name)
                            conv3x3s.append(conv3x3)
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        layers = []
        if stride != 1 or \
           self.num_in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = ConvBNActivationBlock(in_channels=self.num_in_channels[branch_index],
                                               out_channels=num_channels[branch_index] * block.expansion,
                                               kernel_size=1,
                                               stride=stride,
                                               padding=0,
                                               bias=False,
                                               bnName=self.bn_name,
                                               activationName=ActivationType.Linear)
        layers.append(block(self.num_in_channels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_in_channels[branch_index],
                                num_channels[branch_index]))
        return nn.Sequential(*layers)

    def forward(self, input_list):
        if self.num_branches == 1:
            return [self.branches[0](input_list[0])]

        for i in range(self.num_branches):
            input_list[i] = self.branches[i](input_list[i])

        x_fuse = []
        for i, layer in enumerate(self.fuse_layers):
            y = input_list[0] if i == 0 else layer[0](input_list[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + input_list[j]
                else:
                    y = y + layer[j](input_list[j])
            x_fuse.append(self.activate(y))
        return x_fuse


class ClassificationHeadBlock(BaseBlock):

    def __init__(self, pre_stage_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(HRNetBlockName.ClassificationHeadBlock)
        self.activation_name = activation_name
        self.bn_name = bn_name
        head_channels = [32, 64, 128, 256]
        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_modules.append(self.make_layer(channels, head_channels[i], 1, stride=1))
        self.incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * Bottleneck.expansion
            out_channels = head_channels[i + 1] * Bottleneck.expansion
            downsamp_module = ConvBNActivationBlock(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    bias=True,
                                                    bnName=self.bn_name,
                                                    activationName=self.activation_name)
            downsamp_modules.append(downsamp_module)
        self.downsamp_modules = nn.ModuleList(downsamp_modules)

    def make_layer(self, inplanes, planes, blocks, stride=1):
        out_channel = planes * Bottleneck.expansion
        downsample = None
        if stride != 1 or inplanes != out_channel:
            downsample = ConvBNActivationBlock(in_channels=inplanes,
                                               out_channels=out_channel,
                                               kernel_size=1,
                                               stride=stride,
                                               bias=False,
                                               bnName=self.bn_name,
                                               activationName=ActivationType.Linear)

        layers = [Bottleneck(inplanes, planes, stride, downsample)]
        for i in range(1, blocks):
            temp_block = Bottleneck(out_channel, planes, bn_name=self.bn_name,
                                    activation_name=self.activation_name)
            layers.append(temp_block)
        return nn.Sequential(*layers)

    def forward(self, input_list):
        y = self.incre_modules[0](input_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](input_list[i + 1]) + self.downsamp_modules[i](y)
        return y
