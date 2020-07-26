#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from easyai.base_name.block_name import BlockType
from easyai.base_name.block_name import LayerType
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.model.utility.model_parse import ModelParse


class SlimPrune():

    def __init__(self, cfg_path, weight_path,
                 global_percent, layer_keep):
        assert cfg_path.endswith("cfg")
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.global_percent = global_percent
        self.layer_keep = layer_keep
        self.model_factory = ModelFactory()
        self.model_process = TorchModelProcess()
        self.model_write = ModelParse()

        path, file_name_and_post = os.path.split(cfg_path)
        self.save_cfg_path = os.path.join(path, "prune_%s" % file_name_and_post)
        path, file_name_and_post = os.path.split(weight_path)
        self.save_weight_path = os.path.join(path, "prune_%s" % file_name_and_post)

    def prune(self):
        torch_model = self.model_factory.get_model(self.cfg_path)
        self.model_process.loadLatestModelWeight(self.weight_path, torch_model)
        assert torch_model.backbone_name.endswith("cfg")

        base_module_list, task_module_list, \
            base_module_filters, task_module_filters = self.get_module_list(torch_model)
        threshold = self.get_bn_weights_threshold(base_module_list, task_module_list)
        base_filters_mask, task_filters_mask = self.get_filters_mask(base_module_list,
                                                                     task_module_list,
                                                                     threshold)
        self.merge_mask(base_module_list, task_module_filters,
                        base_filters_mask, task_filters_mask)

        self.prune_model_keep_size(base_module_list, task_module_filters,
                                   base_module_filters, task_module_filters,
                                   base_filters_mask, task_filters_mask)

        self.save_model_cfg(torch_model, base_filters_mask, task_filters_mask)
        self.save_model_weight(torch_model, self.save_cfg_path,
                               base_module_list, task_module_filters,
                               base_filters_mask, task_filters_mask)

    def get_module_list(self, model):
        base_module_list = []
        task_module_list = []
        base_module_filters = []
        task_module_filters = []
        for index, key, block in enumerate(model._modules.items()):
            if BlockType.BaseNet in key:
                base_module_filters = block.get_outchannel_list()
                for base_key, base_block in block._modules.items():
                    base_module_list.append((base_key, base_block))
            else:
                task_module_filters.append(model.block_out_channels[index])
                task_module_list.append((key, block))
        return base_module_list, task_module_list, base_module_filters, task_module_filters

    def get_bn_weights_threshold(self, base_module_list, task_module_list):
        prune_count = 0
        bn_weights = []
        for index, key, block in enumerate(base_module_list):
            if BlockType.ConvBNActivationBlock in key:
                next_key, next_block = base_module_list[index+1]
                if LayerType.Upsample not in next_key and \
                        LayerType.MyMaxPool2d not in next_key:
                    bn_weights.append(self.get_bn_weights(block))
                    prune_count += 1

        for index, key, block in enumerate(task_module_list):
            if BlockType.ConvBNActivationBlock in key:
                next_key, next_block = task_module_list[index+1]
                if LayerType.Upsample not in next_key and \
                        LayerType.MyMaxPool2d not in next_key:
                    bn_weights.append(self.get_bn_weights(block))
                    prune_count += 1

        bn_weights = torch.cat(bn_weights, 1)

        sorted_bn, sorted_index = torch.sort(bn_weights)
        threshold_index = int(len(bn_weights) * self.global_percent)
        threshold = sorted_bn[threshold_index].cuda()
        return threshold

    def get_filters_mask(self, base_module_list, task_module_list, threshold):
        pruned = 0
        total = 0
        base_filters_mask = OrderedDict()
        task_filters_mask = OrderedDict()
        for index, (key, block) in enumerate(base_module_list):
            if BlockType.ConvBNActivationBlock in key:
                next_key, next_block = base_module_list[index+1]
                if LayerType.Upsample not in next_key and \
                        LayerType.MyMaxPool2d not in next_key:
                    weight_copy = self.get_bn_weights(block)
                    channels = weight_copy.shape[0]  #
                    min_channel_num = int(channels * self.layer_keep) \
                        if int(channels * self.layer_keep) > 0 else 1
                    mask = weight_copy.gt(threshold).float()
                    if int(torch.sum(mask)) < min_channel_num:
                        _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                        mask[sorted_index_weights[:min_channel_num]] = 1.
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain
                else:
                    mask = torch.ones(self.get_bn_weights_shape(block))
                total += mask.shape[0]
                base_filters_mask[index] = mask.clone()

        for index, (key, block) in enumerate(task_module_list):
            if BlockType.ConvBNActivationBlock in key:
                next_key, next_block = task_module_list[index+1]
                if LayerType.Upsample not in next_key and \
                        LayerType.MyMaxPool2d not in next_key:
                    weight_copy = self.get_bn_weights(block)
                    channels = weight_copy.shape[0]  #
                    min_channel_num = int(channels * self.layer_keep) \
                        if int(channels * self.layer_keep) > 0 else 1
                    mask = weight_copy.gt(threshold).float()
                    if int(torch.sum(mask)) < min_channel_num:
                        _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                        mask[sorted_index_weights[:min_channel_num]] = 1.
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain
                else:
                    mask = torch.ones(self.get_bn_weights_shape(block))
                total += mask.shape[0]
                task_filters_mask[index] = mask.clone()

        prune_ratio = pruned / total
        print('Prune channels: {}\tPrune ratio: {}'.format(pruned, prune_ratio))
        return base_filters_mask, task_filters_mask

    def merge_mask(self, base_module_list, task_module_list,
                   base_filters_mask, task_filters_mask):
        base_is_access = {}
        task_is_access = {}
        for index, key, block in enumerate(base_module_list):
            if LayerType.ShortcutLayer in key:
                base_is_access[index] = False
        for index, key, block in enumerate(task_module_list):
            if LayerType.ShortcutLayer in key:
                task_is_access[index] = False
        end_count = len(task_module_list)
        for i in range(-1, -end_count-1, -1):
            key, block = task_module_list[i]
            if LayerType.ShortcutLayer in key:
                if task_is_access[end_count+i]:
                    continue
                merge_masks = []
                layer_i = i
                temp_key, temp_block = task_module_list[layer_i]
                while LayerType.ShortcutLayer in temp_key:
                    if layer_i < 0:
                        task_is_access[end_count+layer_i] = True
                    else:
                        base_is_access[layer_i] = True
                    pre_key, pre_block = task_module_list[layer_i-1]
                    if BlockType.ConvBNActivationBlock in pre_key:
                        merge_masks.append(task_filters_mask[end_count+layer_i-1].unsqueeze(0))
                    layer_i = temp_block.layer_from
                    if layer_i >= 0:
                        temp_key, temp_block = base_module_list[layer_i]
                        if BlockType.ConvBNActivationBlock in temp_block:
                            merge_masks.append(base_filters_mask[layer_i].unsqueeze(0))
                    else:
                        temp_key, temp_block = task_module_list[layer_i]
                        if BlockType.ConvBNActivationBlock in temp_block:
                            merge_masks.append(task_filters_mask[end_count+layer_i].unsqueeze(0))

                if len(merge_masks) > 1:
                    merge_masks = torch.cat(merge_masks, 0)
                    merge_mask = (torch.sum(merge_masks, dim=0) > 0).float()
                else:
                    merge_mask = merge_masks[0].float()

                layer_i = i
                temp_key, temp_block = task_module_list[layer_i]
                while LayerType.ShortcutLayer in temp_key:
                    pre_key, pre_block = task_module_list[layer_i - 1]
                    if BlockType.ConvBNActivationBlock in pre_key:
                        task_filters_mask[end_count+layer_i-1] = merge_mask

                    layer_i = temp_block.layer_from
                    if layer_i >= 0:
                        temp_key, temp_block = base_module_list[layer_i]
                        if BlockType.ConvBNActivationBlock in temp_block:
                            base_filters_mask[layer_i] = merge_mask

                    else:
                        temp_key, temp_block = task_module_list[layer_i]
                        if BlockType.ConvBNActivationBlock in temp_block:
                            task_filters_mask[end_count + layer_i] = merge_mask

        end_count = len(base_module_list)
        for i in range(-1, -end_count-1, -1):
            key, block = base_module_list[i]
            if LayerType.ShortcutLayer in key:
                if base_is_access[end_count+i]:
                    continue
                merge_masks = []
                layer_i = i
                temp_key, temp_block = base_module_list[layer_i]
                while LayerType.ShortcutLayer in temp_key:
                    base_is_access[end_count+layer_i] = True
                    pre_key, pre_block = base_module_list[layer_i-1]
                    if BlockType.ConvBNActivationBlock in pre_key:
                        merge_masks.append(base_filters_mask[end_count+layer_i-1].unsqueeze(0))
                    layer_i = temp_block.layer_from
                    if layer_i < 0:
                        temp_key, temp_block = base_module_list[layer_i]
                        if BlockType.ConvBNActivationBlock in temp_block:
                            merge_masks.append(base_filters_mask[end_count+layer_i].unsqueeze(0))
                    else:
                        print("merge channle error")
                        break

                if len(merge_masks) > 1:
                    merge_masks = torch.cat(merge_masks, 0)
                    merge_mask = (torch.sum(merge_masks, dim=0) > 0).float()
                else:
                    merge_mask = merge_masks[0].float()

                layer_i = i
                temp_key, temp_block = task_module_list[layer_i]
                while LayerType.ShortcutLayer in temp_key:
                    pre_key, pre_block = task_module_list[layer_i - 1]
                    if BlockType.ConvBNActivationBlock in pre_key:
                        base_filters_mask[end_count+layer_i-1] = merge_mask

                    layer_i = temp_block.layer_from
                    if layer_i < 0:
                        temp_key, temp_block = base_module_list[layer_i]
                        if BlockType.ConvBNActivationBlock in temp_block:
                            base_filters_mask[end_count+layer_i] = merge_mask
                        else:
                            print("merge channle error")
                            break

    def prune_model_keep_size(self, base_module_list, task_module_list,
                              base_module_filters, task_module_filters,
                              base_filters_mask, task_filters_mask):
        base_activations = []
        task_activations = []
        for index, (key, block) in enumerate(base_module_list):
            if LayerType.Convolutional in key:
                activation = torch.zeros(base_module_filters[index]).cuda()
                base_activations.append(activation)
            elif BlockType.ConvActivationBlock in key:
                activation = torch.zeros(base_module_filters[index]).cuda()
                base_activations.append(activation)
            elif BlockType.ConvBNActivationBlock in key:
                activation = torch.zeros(base_module_filters[index]).cuda()
                next_key, next_block = base_module_list[index + 1]
                if LayerType.Upsample not in next_key and \
                        LayerType.MyMaxPool2d not in next_key:
                    mask = base_filters_mask[index]
                    block.block[1].weight.data.mul_(mask)
                    activation = F.leaky_relu((1 - mask) * block.block[1].bias.data, 0.1)
                    self.update_next_block_activation(next_key, next_block, activation)
                    block.block[1].bias.data.mul_(mask)
                base_activations.append(activation)
            elif LayerType.ShortcutLayer in key:
                actv1 = base_activations[index - 1]
                from_layer = block.layer_from
                actv2 = base_activations[from_layer]
                activation = actv1 + actv2
                next_key, next_block = base_module_list[index + 1]
                self.update_next_block_activation(next_key, next_block, activation)
                base_activations.append(activation)
            elif LayerType.RouteLayer in key:
                # spp不参与剪枝，其中的route不用更新，仅占位
                from_layers = block.layers
                activation = None
                next_key, next_block = base_module_list[index + 1]
                if len(from_layers) == 1:
                    activation = base_activations[from_layers[0]]
                    self.update_next_block_activation(next_key, next_block, activation)
                elif len(from_layers) == 2:
                    actv1 = base_activations[from_layers[0]]
                    actv2 = base_activations[from_layers[1]]
                    activation = torch.cat((actv1, actv2))
                    self.update_next_block_activation(next_key, next_block, activation)
                base_activations.append(activation)
            elif LayerType.Upsample in key:
                base_activations.append(base_activations[index-1])
            else:
                base_activations.append(None)

        for index, (key, block) in enumerate(task_module_list):
            if LayerType.Convolutional in key:
                activation = torch.zeros(task_module_filters[index]).cuda()
                task_activations.append(activation)
            elif BlockType.ConvActivationBlock in key:
                activation = torch.zeros(task_module_filters[index]).cuda()
                task_activations.append(activation)
            elif BlockType.ConvBNActivationBlock in key:
                activation = torch.zeros(task_module_filters[index]).cuda()
                next_key, next_block = task_module_list[index + 1]
                if LayerType.Upsample not in next_key and \
                        LayerType.MyMaxPool2d not in next_key:
                    mask = task_filters_mask[index]
                    block.block[1].weight.data.mul_(mask)
                    activation = F.leaky_relu((1 - mask) * block.block[1].bias.data, 0.1)
                    self.update_next_block_activation(next_key, next_block, activation)
                    block.block[1].bias.data.mul_(mask)
                task_activations.append(activation)
            elif LayerType.ShortcutLayer in key:
                actv1 = task_activations[index - 1]
                from_layer = block.layer_from
                if from_layer >= 0:
                    actv2 = base_activations[from_layer]
                else:
                    actv2 = task_activations[from_layer]
                activation = actv1 + actv2
                next_key, next_block = task_module_list[index + 1]
                self.update_next_block_activation(next_key, next_block, activation)
                task_activations.append(activation)
            elif LayerType.RouteLayer in key:
                # spp不参与剪枝，其中的route不用更新，仅占位
                from_layers = block.layers
                activation = None
                next_key, next_block = task_module_list[index + 1]
                if len(from_layers) == 1:
                    if from_layers[0] >= 0:
                        activation = base_activations[from_layers[0]]
                    else:
                        activation = task_activations[from_layers[0]]
                    self.update_next_block_activation(next_key, next_block, activation)
                elif len(from_layers) == 2:
                    if from_layers[0] >= 0:
                        actv1 = base_activations[from_layers[0]]
                    else:
                        actv1 = task_activations[from_layers[0]]
                    if from_layers[1] >= 0:
                        actv2 = base_activations[from_layers[1]]
                    else:
                        actv2 = task_activations[from_layers[1]]
                    activation = torch.cat((actv1, actv2))
                    self.update_next_block_activation(next_key, next_block, activation)
                task_activations.append(activation)
            elif LayerType.Upsample in key:
                task_activations.append(task_activations[index-1])
            else:
                task_activations.append(None)

    def save_model_cfg(self, model, base_filters_mask, task_filters_mask):
        prune_backbone_defines = None
        for key, block in model._modules.items():
            if BlockType.BaseNet in key:
                prune_backbone_defines = deepcopy(block.model_defines)
        prune_task_defines = deepcopy(model.model_defines)
        path, file_name_and_post = os.path.split(self.save_cfg_path)
        backbone_name = "prune_%s" % model.backbone_name
        backbone_cfg_path = os.path.join(path, backbone_name)

        for key, mask in base_filters_mask.items():
            index = key + 1
            filters = int(mask.sum())
            assert prune_backbone_defines[index]['type'] == BlockType.ConvBNActivationBlock
            prune_backbone_defines[index]['filters'] = filters

        for key, mask in task_filters_mask.items():
            index = key + 1
            filters = int(mask.sum())
            assert prune_task_defines[index]['type'] == BlockType.ConvBNActivationBlock
            prune_task_defines[index]['filters'] = filters

        task_backbone = {'type': BlockType.BaseNet,
                         'name': backbone_name}
        prune_task_defines.insert(0, task_backbone)

        self.model_write.write_cfg_file(prune_backbone_defines, backbone_cfg_path)
        self.model_write.write_cfg_file(prune_task_defines, self.save_cfg_path)

    def save_model_weight(self, model, save_cfg_path,
                          base_module_list, task_module_list,
                          base_filters_mask, task_filters_mask):
        prune_model = self.model_factory.get_model(save_cfg_path)
        prune_dict = prune_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in model_dict.items():
            if k in prune_dict and v.shape == prune_dict[k].shape:
                pretrained_dict[k] = v
        # print("Load pretrained parameters:")
        # for k, v in pretrained_dict.items():
        #     print(k, v.shape)
        prune_dict.update(pretrained_dict)
        prune_model.load_state_dict(prune_dict)

        prune_base_module_list, prune_task_module_list, _, _ = self.get_module_list(prune_model)

        for key, mask in base_filters_mask.items():
            base_filters_mask[key] = mask.clone().cpu().numpy()

        for key, mask in task_filters_mask.items():
            task_filters_mask[key] = mask.clone().cpu().numpy()

        for index, (prune_key, prune_block) in enumerate(prune_base_module_list):
            old_key, old_block = base_module_list[index]
            numpy_mask = base_filters_mask[index]
            if BlockType.ConvBNActivationBlock in prune_key:
                # bn
                out_channel_idx = np.argwhere(numpy_mask)[:, 0].tolist()
                prune_block.block[1].weight.data = old_block.block[1].weight.data[out_channel_idx].clone()
                prune_block.block[1].bias.data = old_block.block[1].bias.data[out_channel_idx].clone()
                prune_block.block[1].running_mean.data = old_block.block[1].running_mean.data[out_channel_idx].clone()
                prune_block.block[1].running_var.data = old_block.block[1].running_var.data[out_channel_idx].clone()
                # conv
                input_mask = self.get_base_input_mask(index, base_module_list, base_filters_mask)
                in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
                tmp = old_block.block[0].weight.data[:, in_channel_idx, :, :].clone()
                prune_block.block[0].weight.data = tmp[out_channel_idx, :, :, :].clone()
            elif LayerType.Convolutional in prune_key:
                input_mask = self.get_base_input_mask(index, base_module_list, base_filters_mask)
                in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
                prune_block.weight.data = old_block.weight.data[:, in_channel_idx, :, :].clone()
                prune_block.bias.data = old_block.bias.data.clone()
            elif BlockType.ConvActivationBlock in prune_key:
                input_mask = self.get_base_input_mask(index, base_module_list, base_filters_mask)
                in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
                prune_block[0].weight.data = old_block[0].weight.data[:, in_channel_idx, :, :].clone()
                prune_block[0].bias.data = old_block[0].bias.data.clone()

        for index, (prune_key, prune_block) in enumerate(prune_task_module_list):
            old_key, old_block = task_module_list[index]
            numpy_mask = base_filters_mask[index]
            if BlockType.ConvBNActivationBlock in prune_key:
                # bn
                out_channel_idx = np.argwhere(numpy_mask)[:, 0].tolist()
                prune_block.block[1].weight.data = old_block.block[1].weight.data[out_channel_idx].clone()
                prune_block.block[1].bias.data = old_block.block[1].bias.data[out_channel_idx].clone()
                prune_block.block[1].running_mean.data = old_block.block[1].running_mean.data[out_channel_idx].clone()
                prune_block.block[1].running_var.data = old_block.block[1].running_var.data[out_channel_idx].clone()
                # conv
                input_mask = self.get_task_input_mask(index, task_module_list, task_filters_mask,
                                                      base_module_list, base_filters_mask)
                in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
                tmp = old_block.block[0].weight.data[:, in_channel_idx, :, :].clone()
                prune_block.block[0].weight.data = tmp[out_channel_idx, :, :, :].clone()
            elif LayerType.Convolutional in prune_key:
                input_mask = self.get_task_input_mask(index, task_module_list, task_filters_mask,
                                                      base_module_list, base_filters_mask)
                in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
                prune_block.weight.data = old_block.weight.data[:, in_channel_idx, :, :].clone()
                prune_block.bias.data = old_block.bias.data.clone()
            elif BlockType.ConvActivationBlock in prune_key:
                input_mask = self.get_task_input_mask(index, task_module_list, task_filters_mask,
                                                      base_module_list, base_filters_mask)
                in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
                prune_block[0].weight.data = old_block[0].weight.data[:, in_channel_idx, :, :].clone()
                prune_block[0].bias.data = old_block[0].bias.data.clone()

        self.model_process.saveLatestModel(self.save_weight_path, prune_model)

    def get_base_input_mask(self, index, base_module_list, base_filters_mask):
        if index == 0:
            return np.ones(3)
        pre_key, pre_block = base_module_list[index -1]
        if LayerType.Convolutional in pre_key:
            mask = base_filters_mask[index -1]
            return mask
        elif BlockType.ConvActivationBlock in pre_key:
            mask = base_filters_mask[index - 1]
            return mask
        elif BlockType.ConvBNActivationBlock in pre_key:
            mask = base_filters_mask[index - 1]
            return mask
        elif LayerType.ShortcutLayer in pre_key:
            mask = base_filters_mask[index - 2]
            return mask
        elif LayerType.RouteLayer in pre_key:
            from_layers = pre_block.layers
            if len(from_layers) == 1:
                return base_filters_mask[from_layers[0]]
            elif len(from_layers) == 2:
                mask1 = base_filters_mask[from_layers[0]]
                temp_key, temp_block = base_module_list[from_layers[1]]
                if LayerType.Convolutional in temp_key:
                    mask2 = base_filters_mask[from_layers[1]]
                elif BlockType.ConvActivationBlock in temp_key:
                    mask2 = base_filters_mask[from_layers[1]]
                elif BlockType.ConvBNActivationBlock in temp_key:
                    mask2 = base_filters_mask[from_layers[1]]
                else:
                    mask2 = base_filters_mask[from_layers[1]-1]
                return np.concatenate([mask1, mask2])
            elif len(from_layers) == 4:
                # spp结构中最后一个route
                mask = base_filters_mask[from_layers[-1]]
                return np.concatenate([mask, mask, mask, mask])
            else:
                print("Something wrong with route module!")
                raise Exception

    def get_task_input_mask(self, index,
                            task_module_list, task_filters_mask,
                            base_module_list, base_filters_mask):
        pre_key, pre_block = task_module_list[index - 1]
        if LayerType.Convolutional in pre_key:
            return task_filters_mask[index - 1]
        elif BlockType.ConvActivationBlock in pre_key:
            return task_filters_mask[index - 1]
        elif BlockType.ConvBNActivationBlock in pre_key:
            return task_filters_mask[index - 1]
        elif LayerType.ShortcutLayer in pre_key:
            return task_filters_mask[index - 2]
        elif LayerType.RouteLayer in pre_key:
            from_layers = pre_block.layers
            if len(from_layers) == 1:
                temp_index = from_layers[0]
                if temp_index >= 0:
                    return base_filters_mask[from_layers[0]]
                else:
                    return task_filters_mask[from_layers[0]]
            elif len(from_layers) == 2:
                temp_index1 = from_layers[0]
                if temp_index1 >= 0:
                    mask1 = base_filters_mask[temp_index1]
                else:
                    mask1 = task_filters_mask[temp_index1]

                temp_index2 = from_layers[1]
                if temp_index2 >= 0:
                    temp_key, temp_block = base_module_list[temp_index2]
                    if LayerType.Convolutional in temp_key:
                        mask2 = base_filters_mask[temp_index2]
                    elif BlockType.ConvActivationBlock in temp_key:
                        mask2 = base_filters_mask[temp_index2]
                    elif BlockType.ConvBNActivationBlock in temp_key:
                        mask2 = base_filters_mask[temp_index2]
                    else:
                        mask2 = base_filters_mask[temp_index2 - 1]
                else:
                    temp_key, temp_block = task_module_list[temp_index2]
                    if LayerType.Convolutional in temp_key:
                        mask2 = task_filters_mask[temp_index2]
                    elif BlockType.ConvActivationBlock in temp_key:
                        mask2 = task_filters_mask[temp_index2]
                    elif BlockType.ConvBNActivationBlock in temp_key:
                        mask2 = task_filters_mask[temp_index2]
                    else:
                        mask2 = task_filters_mask[temp_index2 - 1]
                return np.concatenate([mask1, mask2])
            elif len(from_layers) == 4:
                # spp结构中最后一个route
                temp_index = from_layers[-1]
                if temp_index >= 0:
                    mask = base_filters_mask[temp_index]
                else:
                    mask = task_filters_mask[temp_index]
                return np.concatenate([mask, mask, mask, mask])
            else:
                print("Something wrong with route module!")
                raise Exception

    def get_bn_weights(self, block):
        for m in block.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                return m.weight.data.abs().clone()

    def get_bn_weights_shape(self, block):
        for m in block.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                return m.weight.data.shape

    def update_next_block_activation(self, next_key, next_block, activation):
        if LayerType.Convolutional in next_key:
            conv_sum = next_block.weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            next_block.bias.data.add_(offset)  # bias add offset(conv * activation)
        elif BlockType.ConvActivationBlock in next_key:
            conv_sum = next_block.block[0].weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            next_block.block[0].bias.data.add_(offset)  # bias add offset(conv * activation)
        elif BlockType.ConvBNActivationBlock in next_key:
            conv_sum = next_block.block[0].weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            next_block.block[0].running_mean.data.sub_(offset)  # mean sub offset(conv * activation)

    def update_activation(self, i, pruned_model, activation, CBL_idx):
        next_idx = i + 1
        if pruned_model.module_defs[next_idx]['type'] == 'convolutional':
            next_conv = pruned_model.module_list[next_idx][0]
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            if next_idx in CBL_idx:
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)  # mean sub offset(conv * activation)
            else:
                # 这里需要注意的是，对于convolutionnal，如果有BN，则该层卷积层不使用bias，如果无BN，则使用bias
                next_conv.bias.data.add_(offset)  # bias add offset(conv * activation)
