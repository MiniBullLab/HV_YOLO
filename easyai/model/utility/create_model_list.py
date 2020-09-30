#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.nn as nn
from collections import OrderedDict
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock, ConvActivationBlock
from easyai.model.base_block.utility.utility_layer import NormalizeLayer, ActivationLayer
from easyai.model.base_block.utility.utility_layer import MultiplyLayer, AddLayer
from easyai.model.base_block.utility.utility_layer import RouteLayer, ShortRouteLayer
from easyai.model.base_block.utility.utility_layer import ShortcutLayer
from easyai.model.base_block.utility.utility_layer import FcLayer
from easyai.model.base_block.utility.pooling_layer import MyMaxPool2d, GlobalAvgPool2d
from easyai.model.base_block.utility.pooling_layer import SpatialPyramidPooling
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.detection_block import Detection2dBlock
from easyai.model.base_block.cls.darknet_block import ReorgBlock, DarknetBlockName
from easyai.loss.cls.ce2d_loss import CrossEntropy2d
from easyai.loss.cls.bce_loss import BinaryCrossEntropy2d
from easyai.loss.seg.ohem_cross_entropy2d import OhemCrossEntropy2d
from easyai.loss.det2d.region2d_loss import Region2dLoss
from easyai.loss.det2d.yolov3_loss import YoloV3Loss
from easyai.loss.det2d.multibox_loss import MultiBoxLoss
from easyai.loss.det2d.key_points2d_region_loss import KeyPoints2dRegionLoss


class CreateModuleList():

    def __init__(self):
        self.index = 0
        self.outChannelList = []
        self.blockDict = OrderedDict()

        self.filters = 0
        self.input_channels = 0

    def set_start_index(self, index=0):
        self.index = index

    def getBlockList(self):
        return self.blockDict

    def getOutChannelList(self):
        return self.outChannelList

    def createOrderedDict(self, model_define, input_channels):
        self.blockDict.clear()
        self.outChannelList.clear()
        self.filters = 0
        self.input_channels = 0

        for block_def in model_define:
            if block_def['type'] == BlockType.InputData:
                data_channel = int(block_def['data_channel'])
                self.input_channels = data_channel
            elif block_def['type'] == DarknetBlockName.ReorgBlock:
                stride = int(block_def['stride'])
                block = ReorgBlock(stride=stride)
                self.filters = block.stride * block.stride * self.outChannelList[-1]
                self.add_block_list(DarknetBlockName.ReorgBlock, block, self.filters)
                self.input_channels = self.filters
            elif block_def['type'] == BlockType.SpatialPyramidPooling:
                pool_sizes = [int(x) for x in block_def['pool_sizes'].split(',') if x.strip()]
                block = SpatialPyramidPooling(pool_sizes=pool_sizes)
                self.filters = (len(pool_sizes) + 1) * self.outChannelList[-1]
                self.add_block_list(BlockType.SpatialPyramidPooling, block, self.filters)
                self.input_channels = self.filters
            elif block_def['type'] == BlockType.Detection2dBlock:
                anchor_number = int(block_def['anchor_number'])
                class_number = int(block_def['class_number'])
                block = Detection2dBlock(self.input_channels, anchor_number, class_number)
                self.filters = -1
                self.add_block_list(BlockType.Detection2dBlock, block, self.filters)
                self.input_channels = self.filters
            else:
                self.create_layer(block_def, input_channels)
                self.create_convolutional(block_def)
                self.create_loss(block_def)

    def create_layer(self, module_def, input_channels):
        if module_def['type'] == LayerType.MyMaxPool2d:
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            maxpool = MyMaxPool2d(kernel_size, stride)
            self.add_block_list(LayerType.MyMaxPool2d, maxpool, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.GlobalAvgPool:
            globalAvgPool = GlobalAvgPool2d()
            self.add_block_list(LayerType.GlobalAvgPool, globalAvgPool, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.FcLayer:
            num_output = int(module_def['num_output'])
            self.filters = num_output
            layer = FcLayer(self.input_channels, num_output)
            self.add_block_list(LayerType.FcLayer, layer, num_output)
            self.input_channels = num_output
        elif module_def['type'] == LayerType.Upsample:
            scale = int(module_def['stride'])
            mode = module_def.get('mode', 'bilinear')
            upsample = Upsample(scale_factor=scale, mode=mode)
            self.add_block_list(LayerType.Upsample, upsample, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.MultiplyLayer:
            block = MultiplyLayer(module_def['layers'])
            mult_index = block.layers[0]
            self.filters = input_channels[mult_index] if mult_index >= 0 \
                else self.outChannelList[mult_index]
            self.add_block_list(LayerType.MultiplyLayer, block, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.AddLayer:
            block = AddLayer(module_def['layers'])
            add_index = block.layers[0]
            self.filters = input_channels[add_index] if add_index >= 0 \
                else self.outChannelList[add_index]
            self.add_block_list(LayerType.AddLayer, block, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.RouteLayer:
            block = RouteLayer(module_def['layers'])
            self.filters = sum([input_channels[i] if i >= 0 else self.outChannelList[i]
                                for i in block.layers])
            self.add_block_list(LayerType.RouteLayer, block, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.ShortRouteLayer:
            block = ShortRouteLayer(module_def['from'], module_def['activation'])
            self.filters = self.outChannelList[block.layer_from] + \
                           self.outChannelList[-1]
            self.add_block_list(LayerType.ShortRouteLayer, block, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.ShortcutLayer:
            block = ShortcutLayer(module_def['from'], module_def['activation'])
            self.filters = self.outChannelList[block.layer_from]
            self.add_block_list(LayerType.ShortcutLayer, block, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.Dropout:
            probability = float(module_def['probability'])
            layer = nn.Dropout(p=probability, inplace=False)
            self.add_block_list(LayerType.Dropout, layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.NormalizeLayer:
            bn_name = module_def['batch_normalize'].strip()
            layer = NormalizeLayer(bn_name, self.filters)
            self.add_block_list(LayerType.NormalizeLayer, layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.ActivationLayer:
            activation_name = module_def['activation'].strip()
            layer = ActivationLayer(activation_name, inplace=False)
            self.add_block_list(LayerType.ActivationLayer, layer, self.filters)
            self.input_channels = self.filters

    def create_convolutional(self, module_def):
        if module_def['type'] == LayerType.Convolutional:
            self.filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = int(module_def.get("pad", None))
            if pad is None:
                pad = ((kernel_size - 1) // 2)
            assert pad == ((kernel_size - 1) // 2)
            dilation = int(module_def.get('dilation', 1))
            groups = int(module_def.get("groups", 1))
            if dilation > 1:
                pad = dilation
            block = nn.Conv2d(in_channels=self.input_channels,
                              out_channels=self.filters,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=pad,
                              dilation=dilation,
                              groups=groups,
                              bias=True)
            self.add_block_list(LayerType.Convolutional, block, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == BlockType.ConvActivationBlock:
            self.filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = int(module_def.get("pad", None))
            if pad is None:
                pad = ((kernel_size - 1) // 2)
            assert pad == ((kernel_size - 1) // 2)
            activationName = module_def['activation']
            dilation = int(module_def.get("dilation", 1))
            groups = int(module_def.get("groups", 1))
            if dilation > 1:
                pad = dilation
            block = ConvActivationBlock(in_channels=self.input_channels,
                                        out_channels=self.filters,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=pad,
                                        dilation=dilation,
                                        groups=groups,
                                        activationName=activationName)
            self.add_block_list(BlockType.ConvActivationBlock, block, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == BlockType.ConvBNActivationBlock:
            bnName = module_def['batch_normalize']
            self.filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = int(module_def.get("pad", None))
            if pad is None:
                pad = ((kernel_size - 1) // 2)
            assert pad == ((kernel_size - 1) // 2)
            activationName = module_def['activation']
            dilation = int(module_def.get("dilation", 1))
            groups = int(module_def.get("groups", 1))
            if dilation > 1:
                pad = dilation
            block = ConvBNActivationBlock(in_channels=self.input_channels,
                                          out_channels=self.filters,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=pad,
                                          bnName=bnName,
                                          dilation=dilation,
                                          groups=groups,
                                          activationName=activationName)
            self.add_block_list(BlockType.ConvBNActivationBlock, block, self.filters)
            self.input_channels = self.filters

    def create_loss(self, module_def):
        if module_def["type"] == LossType.CrossEntropy2d:
            weight_type = int(module_def.get("weight_type", 0))
            weight = module_def.get("weight", None)
            reduce = module_def.get("reduce", None)
            reduction = module_def.get("reduction", 'mean')
            ignore_index = int(module_def.get("ignore_index", 250))
            layer = CrossEntropy2d(weight_type=weight_type,
                                   weight=weight,
                                   reduce=reduce,
                                   reduction=reduction,
                                   ignore_index=ignore_index)
            self.add_block_list(LossType.CrossEntropy2d, layer, self.filters)
            self.input_channels = self.filters
        elif module_def["type"] == LossType.BinaryCrossEntropy2d:
            weight_type = int(module_def.get("weight_type", 0))
            weight = module_def.get("weight", None)
            reduce = module_def.get("reduce", None)
            reduction = module_def.get("reduction", 'mean')
            layer = BinaryCrossEntropy2d(weight_type=weight_type,
                                         weight=weight,
                                         reduce=reduce,
                                         reduction=reduction)
            self.add_block_list(LossType.BinaryCrossEntropy2d, layer, self.filters)
            self.input_channels = self.filters
        elif module_def["type"] == LossType.OhemCrossEntropy2d:
            ignore_index = int(module_def.get("ignore_index", 250))
            layer = OhemCrossEntropy2d(ignore_index=ignore_index)
            self.add_block_list(LossType.OhemCrossEntropy2d, layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LossType.Region2dLoss:
            anchor_sizes_str = (x for x in module_def['anchor_sizes'].split('|') if x.strip())
            anchor_sizes = []
            for data in anchor_sizes_str:
                temp_value = [float(x) for x in data.split(',') if x.strip()]
                anchor_sizes.append(temp_value)
            class_number = int(module_def['class_number'])
            reduction = int(module_def['reduction'])
            coord_weight = float(module_def['coord_weight'])
            noobject_weight = float(module_def['noobject_weight'])
            object_weight = float(module_def['object_weight'])
            class_weight = float(module_def['class_weight'])
            iou_threshold = float(module_def['iou_threshold'])
            loss_layer = Region2dLoss(class_number, anchor_sizes, reduction,
                                      coord_weight=coord_weight, noobject_weight=noobject_weight,
                                      object_weight=object_weight, class_weight=class_weight,
                                      iou_threshold=iou_threshold)
            self.add_block_list(LossType.Region2dLoss, loss_layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LossType.YoloV3Loss:
            anchor_sizes_str = (x for x in module_def['anchor_sizes'].split('|') if x.strip())
            anchor_mask = [int(x) for x in module_def['anchor_mask'].split(',')]
            anchor_sizes = []
            for data in anchor_sizes_str:
                temp_value = [float(x) for x in data.split(',') if x.strip()]
                anchor_sizes.append(temp_value)
            class_number = int(module_def['class_number'])
            reduction = int(module_def['reduction'])
            coord_weight = float(module_def['coord_weight'])
            noobject_weight = float(module_def['noobject_weight'])
            object_weight = float(module_def['object_weight'])
            class_weight = float(module_def['class_weight'])
            iou_threshold = float(module_def['iou_threshold'])
            yolo_layer = YoloV3Loss(class_number, anchor_sizes, anchor_mask, reduction,
                                    coord_weight=coord_weight, noobject_weight=noobject_weight,
                                    object_weight=object_weight, class_weight=class_weight,
                                    iou_threshold=iou_threshold)
            self.add_block_list(LossType.YoloV3Loss, yolo_layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LossType.MultiBoxLoss:
            class_number = int(module_def['class_number'])
            iou_threshold = float(module_def['iou_threshold'])
            input_size = (int(x) for x in module_def['input_size'].split(',') if x.strip())
            anchor_counts = (int(x) for x in module_def['anchor_counts'].split(',') if x.strip())
            anchor_sizes = (int(x) for x in module_def['anchor_sizes'].split(',') if x.strip())
            aspect_ratio_str = (x for x in module_def['aspect_ratio_list'].split('|') if x.strip())
            aspect_ratio_list = []
            for data in aspect_ratio_str:
                temp_value = [int(x) for x in data.split(',') if x.strip()]
                aspect_ratio_list.append(temp_value)
            loss_layer = MultiBoxLoss(class_number, iou_threshold,
                                      input_size=input_size, anchor_counts=anchor_counts,
                                      anchor_sizes=anchor_sizes, aspect_ratio_list=aspect_ratio_list)
            self.add_block_list(LossType.MultiBoxLoss, loss_layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LossType.KeyPoints2dRegionLoss:
            class_number = int(module_def['class_number'])
            point_count = int(module_def['point_count'])
            loss_layer = KeyPoints2dRegionLoss(class_number, point_count)
            self.add_block_list(LossType.KeyPoints2dRegionLoss, loss_layer, self.filters)
            self.input_channels = self.filters

    def add_block_list(self, block_name, block, out_channel):
        block_name = "%s_%d" % (block_name, self.index)
        self.blockDict[block_name] = block
        self.outChannelList.append(out_channel)
        self.index += 1
