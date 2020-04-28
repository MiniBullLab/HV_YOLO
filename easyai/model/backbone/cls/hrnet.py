#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.hrnet_block import HRNetBlockName
from easyai.model.base_block.cls.hrnet_block import BasicBlock, Bottleneck, TransitionBlock
from easyai.model.base_block.cls.hrnet_block import HighResolutionBlock, ClassificationHeadBlock

__all__ = ['hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w18',
           'hrnet_w30', 'hrnet_w32', 'hrnet_w40', 'hrnet_w44',
           'hrnet_w48', 'hrnet_w64']


class HighResolutionNet(BaseBackbone):

    def __init__(self, cfgs, data_channel=3,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.HRnet_w18_small)
        self.data_channel = data_channel
        self.cfgs = cfgs
        self.activation_name = activation_name
        self.bn_name = bn_name
        self.in_channel = 64
        self.num_features = 2048

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        stem_width = self.cfgs['STEM_WIDTH']
        conv1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                      out_channels=stem_width,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, stem_width)

        conv2 = ConvBNActivationBlock(in_channels=stem_width,
                                      out_channels=64,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, 64)
        self.in_channel = 64

        stage1_cfg = self.cfgs['STAGE1']
        num_channels = stage1_cfg['NUM_CHANNELS'][0]
        block_type = stage1_cfg['BLOCK']
        num_blocks = stage1_cfg['NUM_BLOCKS'][0]
        self.make_layer(block_type, self.in_channel, num_channels, num_blocks)

        stage2_cfg = self.cfgs['STAGE2']
        num_channels = stage2_cfg['NUM_CHANNELS']
        block_type = stage2_cfg['BLOCK']
        transition1 = TransitionBlock(block_type, (self.in_channel,), num_channels,
                                      bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(transition1.get_name(), transition1, -1)

        pre_stage_channels = self.make_stage(stage2_cfg, num_channels)

        stage3_cfg = self.cfgs['STAGE3']
        num_channels = stage3_cfg['NUM_CHANNELS']
        block_type = stage3_cfg['BLOCK']
        transition2 = TransitionBlock(block_type, pre_stage_channels, num_channels,
                                      bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(transition2.get_name(), transition2, -1)

        pre_stage_channels = self.make_stage(stage3_cfg, num_channels)

        stage4_cfg = self.cfgs['STAGE4']
        num_channels = stage4_cfg['NUM_CHANNELS']
        block_type = stage4_cfg['BLOCK']
        transition3 = TransitionBlock(block_type, pre_stage_channels, num_channels,
                                      bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(transition3.get_name(), transition3, -1)

        pre_stage_channels = self.make_stage(stage4_cfg, num_channels, multi_scale_output=True)

        # Classification Head
        self.make_head(pre_stage_channels)

    def make_head(self, pre_stage_channels):
        head_block = ClassificationHeadBlock(pre_stage_channels, bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(head_block.get_name(), head_block, 256 * Bottleneck.expansion)

        final_layer = ConvBNActivationBlock(in_channels=256 * Bottleneck.expansion,
                                            out_channels=self.num_features,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=True,
                                            bnName=self.bn_name,
                                            activationName=self.activation_name)
        self.add_block_list(final_layer.get_name(), final_layer, self.num_features)

    def make_layer(self, block_type, inplanes, planes, blocks, stride=1):
        block = None
        if block_type == 0:
            block = BasicBlock
        elif block_type == 1:
            block = Bottleneck
        out_channel = planes * block.expansion
        downsample = None
        if stride != 1 or inplanes != out_channel:
            downsample = ConvBNActivationBlock(in_channels=inplanes,
                                               out_channels=out_channel,
                                               kernel_size=1,
                                               stride=stride,
                                               padding=0,
                                               bias=False,
                                               bnName=self.bn_name,
                                               activationName=ActivationType.Linear)

        down_block = block(inplanes, planes, stride, downsample,
                           bn_name=self.bn_name, activation_name=self.activation_name)
        name = "down_" + down_block.get_name()
        self.add_block_list(name, down_block, out_channel)
        for i in range(1, blocks):
            temp_block = block(out_channel, planes, bn_name=self.bn_name,
                               activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, out_channel)
        self.in_channel = out_channel

    def make_stage(self, layer_config, num_in_channels, multi_scale_output=True):
        block_type = layer_config['BLOCK']
        block = None
        if block_type == 0:
            block = BasicBlock
        elif block_type == 1:
            block = Bottleneck
        num_in_channels = [num_in_channels[i] * block.expansion for i in range(len(num_in_channels))]
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        fuse_method = layer_config['FUSE_METHOD']

        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            hr_block = HighResolutionBlock(num_branches, block, num_blocks,
                                           num_in_channels, num_channels,
                                           fuse_method, reset_multi_scale_output,
                                           bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(hr_block.get_name(), hr_block, -1)
            num_in_channels = hr_block.num_in_channels
        return num_in_channels

    def forward(self, x):
        output_list = []
        stage_input = []
        for key, block in self._modules.items():
            if HRNetBlockName.TransitionBlock in key:
                stage_input = [x] if len(stage_input) == 0 else stage_input
                stage_input = block(stage_input)
                output_list.append(stage_input[-1])
            elif HRNetBlockName.HighResolutionBlock in key:
                stage_input = block(stage_input)
                output_list.append(stage_input[-1])
            elif HRNetBlockName.ClassificationHeadBlock in key:
                x = block(stage_input)
                output_list.append(x)
            else:
                x = block(x)
                output_list.append(x)
            # print(key, output_list[-1].shape)
        return output_list


def hrnet_w18_small(data_channel):
    cfgs = dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK=1,
            NUM_BLOCKS=(1,),
            NUM_CHANNELS=(32,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK=0,
            NUM_BLOCKS=(2, 2),
            NUM_CHANNELS=(16, 32),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=3,
            BLOCK=0,
            NUM_BLOCKS=(2, 2, 2),
            NUM_CHANNELS=(16, 32, 64),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=4,
            BLOCK=0,
            NUM_BLOCKS=(2, 2, 2, 2),
            NUM_CHANNELS=(16, 32, 64, 128),
            FUSE_METHOD='SUM',
        ),
    )
    model = HighResolutionNet(cfgs, data_channel)
    model.set_name(BackboneName.HRnet_w18_small)
    return model


def hrnet_w18_small_v2(data_channel):
    cfgs = dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK=1,
            NUM_BLOCKS=(2,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK=0,
            NUM_BLOCKS=(2, 2),
            NUM_CHANNELS=(18, 36),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=3,
            BLOCK=0,
            NUM_BLOCKS=(2, 2, 2),
            NUM_CHANNELS=(18, 36, 72),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=2,
            NUM_BRANCHES=4,
            BLOCK=0,
            NUM_BLOCKS=(2, 2, 2, 2),
            NUM_CHANNELS=(18, 36, 72, 144),
            FUSE_METHOD='SUM',
        ),
    )
    model = HighResolutionNet(cfgs, data_channel)
    model.set_name(BackboneName.HRnet_w18_small_v2)
    return model


def hrnet_w18(data_channel):
    cfgs = dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK=1,
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK=0,
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(18, 36),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(18, 36, 72),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(18, 36, 72, 144),
            FUSE_METHOD='SUM',
        ),
    )
    model = HighResolutionNet(cfgs, data_channel)
    model.set_name(BackboneName.HRnet_w18)
    return model


def hrnet_w30(data_channel):
    cfgs = dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK=1,
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK=0,
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(30, 60),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(30, 60, 120),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(30, 60, 120, 240),
            FUSE_METHOD='SUM',
        ),
    )
    model = HighResolutionNet(cfgs, data_channel)
    model.set_name(BackboneName.HRnet_w30)
    return model


def hrnet_w32(data_channel):
    cfgs = dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK=1,
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK=0,
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(32, 64),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(32, 64, 128),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(32, 64, 128, 256),
            FUSE_METHOD='SUM',
        ),
    )
    model = HighResolutionNet(cfgs, data_channel)
    model.set_name(BackboneName.HRnet_w32)
    return model


def hrnet_w40(data_channel):
    cfgs = dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK=1,
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK=0,
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(40, 80),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(40, 80, 160),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(40, 80, 160, 320),
            FUSE_METHOD='SUM',
        ),
    )
    model = HighResolutionNet(cfgs, data_channel)
    model.set_name(BackboneName.HRnet_w40)
    return model


def hrnet_w44(data_channel):
    cfgs = dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK=1,
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK=0,
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(44, 88),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(44, 88, 176),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(44, 88, 176, 352),
            FUSE_METHOD='SUM',
        ),
    )
    model = HighResolutionNet(cfgs, data_channel)
    model.set_name(BackboneName.HRnet_w44)
    return model


def hrnet_w48(data_channel):
    cfgs = dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK=1,
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK=0,
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(48, 96),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(48, 96, 192),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(48, 96, 192, 384),
            FUSE_METHOD='SUM',
        ),
    )
    model = HighResolutionNet(cfgs, data_channel)
    model.set_name(BackboneName.HRnet_w48)
    return model


def hrnet_w64(data_channel):
    cfgs = dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK=1,
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK=0,
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(64, 128),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(64, 128, 256),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK=0,
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(64, 128, 256, 512),
            FUSE_METHOD='SUM',
        ),
    )
    model = HighResolutionNet(cfgs, data_channel)
    model.set_name(BackboneName.HRnet_w64)
    return model
