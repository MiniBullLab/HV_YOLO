import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

class TorchModelPrune():

    def __init__(self, global_percent, layer_keep):
        self.global_percent = global_percent
        self.layer_keep = layer_keep

    def parse_module_defs(self, module_defs, ignore_idx):
        CBL_idx = [] # conv with bn
        Conv_idx = [] # all conv
        shortcut_idx = dict()
        shortcut_all = set()
        ignore_idx = set()
        for i, module_def in enumerate(module_defs):
            if module_def['type'] == 'convolutional':
                if module_def['batch_normalize'] == '1':
                    CBL_idx.append(i)
                else:
                    Conv_idx.append(i)
                if module_defs[i + 1]['type'] == 'maxpool':
                    # spp前一个CBL不剪
                    ignore_idx.add(i)

            elif module_def['type'] == 'upsample':
                # 上采样层前的卷积层不裁剪
                ignore_idx.add(i - 1)

            elif module_def['type'] == 'shortcut':
                identity_idx = (i + int(module_def['from']))
                if module_defs[identity_idx]['type'] == 'convolutional':

                    # ignore_idx.add(identity_idx)
                    shortcut_idx[i - 1] = identity_idx
                    shortcut_all.add(identity_idx)
                elif module_defs[identity_idx]['type'] == 'shortcut':

                    # ignore_idx.add(identity_idx - 1)
                    shortcut_idx[i - 1] = identity_idx - 1
                    shortcut_all.add(identity_idx - 1)
                shortcut_all.add(i - 1)

        prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

        return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all

    def gather_bn_weights(module_list, prune_idx):

        size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

        bn_weights = torch.zeros(sum(size_list))
        index = 0
        for idx, size in zip(prune_idx, size_list):
            bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
            index += size

        return bn_weights

    def gen_thresh(self, bn_weights):
        sorted_bn = torch.sort(bn_weights)[0]
        sorted_bn, sorted_index = torch.sort(bn_weights)
        thresh_index = int(len(bn_weights) * self.global_percent)
        thresh = sorted_bn[thresh_index].cuda()

        return thresh

    def obtain_filters_mask(self, model, thresh, CBL_idx, prune_idx):

        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]  #
                min_channel_num = int(channels * self.layer_keep) if int(channels * self.layer_keep) > 0 else 1
                mask = weight_copy.gt(thresh).float()

                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(idx, mask.shape[0], remain))
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.clone())

        prune_ratio = pruned / total
        print('Prune channels: {}\tPrune ratio: {}'.format(pruned, prune_ratio))

        return num_filters, filters_mask

    def merge_mask(self, model, CBLidx2mask, CBLidx2filters):
        for i in range(len(model.module_defs) - 1, -1, -1):
            mtype = model.module_defs[i]['type']
            if mtype == 'shortcut':
                if model.module_defs[i]['is_access']:
                    continue

                Merge_masks = []
                layer_i = i
                while mtype == 'shortcut':
                    model.module_defs[layer_i]['is_access'] = True

                    if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                        bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                        if bn:
                            Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))

                    layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                    mtype = model.module_defs[layer_i]['type']

                    if mtype == 'convolutional':
                        bn = int(model.module_defs[layer_i]['batch_normalize'])
                        if bn:
                            Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))

                if len(Merge_masks) > 1:
                    Merge_masks = torch.cat(Merge_masks, 0)
                    merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()
                else:
                    merge_mask = Merge_masks[0].float()

                layer_i = i
                mtype = 'shortcut'
                while mtype == 'shortcut':

                    if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                        bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                        if bn:
                            CBLidx2mask[layer_i - 1] = merge_mask
                            CBLidx2filters[layer_i - 1] = int(torch.sum(merge_mask).item())

                    layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                    mtype = model.module_defs[layer_i]['type']

                    if mtype == 'convolutional':
                        bn = int(model.module_defs[layer_i]['batch_normalize'])
                        if bn:
                            CBLidx2mask[layer_i] = merge_mask
                            CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())

    def get_input_mask(self, module_defs, idx, CBLidx2mask):

        if idx == 0:
            return np.ones(3)

        if module_defs[idx - 1]['type'] == 'convolutional':
            return CBLidx2mask[idx - 1]
        elif module_defs[idx - 1]['type'] == 'shortcut':
            return CBLidx2mask[idx - 2]
        elif module_defs[idx - 1]['type'] == 'route':
            route_in_idxs = []
            for layer_i in module_defs[idx - 1]['layers'].split(","):
                if int(layer_i) < 0:
                    route_in_idxs.append(idx - 1 + int(layer_i))
                else:
                    route_in_idxs.append(int(layer_i))

            if len(route_in_idxs) == 1:
                return CBLidx2mask[route_in_idxs[0]]

            elif len(route_in_idxs) == 2:
                # return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
                mask1 = CBLidx2mask[route_in_idxs[0] - 1]
                if module_defs[route_in_idxs[1]]['type'] == 'convolutional':
                    mask2 = CBLidx2mask[route_in_idxs[1]]
                else:
                    mask2 = CBLidx2mask[route_in_idxs[1] - 1]
                return np.concatenate([mask1, mask2])

            elif len(route_in_idxs) == 4:
                # spp结构中最后一个route
                mask = CBLidx2mask[route_in_idxs[-1]]
                return np.concatenate([mask, mask, mask, mask])

            else:
                print("Something wrong with route module!")
                raise Exception

    def update_activation(self, i, pruned_model, activation, CBL_idx):
        next_idx = i + 1
        if pruned_model.module_defs[next_idx]['type'] == 'convolutional':
            next_conv = pruned_model.module_list[next_idx][0]
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            if next_idx in CBL_idx:
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset) # mean sub offset(conv * activation)
            else:
                # 这里需要注意的是，对于convolutionnal，如果有BN，则该层卷积层不使用bias，如果无BN，则使用bias
                next_conv.bias.data.add_(offset) # bias add offset(conv * activation)

    def init_weights_from_loose_model(self, compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):

        for idx in CBL_idx:
            compact_CBL = compact_model.module_list[idx]
            loose_CBL = loose_model.module_list[idx]
            out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

            compact_bn, loose_bn = compact_CBL[1], loose_CBL[1]
            compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
            compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
            compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
            compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

            input_mask = self.get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
            in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
            compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
            tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
            compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

        for idx in Conv_idx:
            compact_conv = compact_model.module_list[idx][0]
            loose_conv = loose_model.module_list[idx][0]

            input_mask = self.get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
            in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
            compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
            compact_conv.bias.data = loose_conv.bias.data.clone()

    def prune_model_keep_size(self, model, prune_idx, CBL_idx, CBLidx2mask):
        pruned_model = deepcopy(model)
        activations = []
        for i, model_def in enumerate(model.module_defs):

            if model_def['type'] == 'convolutional':
                activation = torch.zeros(int(model_def['filters'])).cuda()
                if i in prune_idx:
                    mask = torch.from_numpy(CBLidx2mask[i]).cuda()
                    bn_module = pruned_model.module_list[i][1]
                    bn_module.weight.data.mul_(mask)
                    activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
                    self.update_activation(i, pruned_model, activation, CBL_idx)
                    bn_module.bias.data.mul_(mask)
                activations.append(activation)

            elif model_def['type'] == 'shortcut':
                actv1 = activations[i - 1]
                from_layer = int(model_def['from'])
                actv2 = activations[i + from_layer]
                activation = actv1 + actv2
                self.update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)

            elif model_def['type'] == 'route':
                # spp不参与剪枝，其中的route不用更新，仅占位
                from_layers = [int(s) for s in model_def['layers'].split(',')]
                activation = None
                if len(from_layers) == 1:
                    activation = activations[i + from_layers[0]]
                    self.update_activation(i, pruned_model, activation, CBL_idx)
                elif len(from_layers) == 2:
                    actv1 = activations[i + from_layers[0]]
                    actv2 = activations[from_layers[1]]
                    activation = torch.cat((actv1, actv2))
                    self.update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)

            elif model_def['type'] == 'upsample':
                # activation = torch.zeros(int(model.module_defs[i - 1]['filters'])).cuda()
                activations.append(activations[i - 1])

            elif model_def['type'] == 'yolo':
                activations.append(None)

            elif model_def['type'] == 'maxpool':
                activations.append(None)

        return pruned_model

    def write_cfg(self, cfg_file, module_defs):

        with open(cfg_file, 'w') as f:
            for module_def in module_defs:
                f.write("[{}]\n".format(module_def['type']))
                for key, value in module_def.items():
                    if key == 'batch_normalize' and value == 0:
                        continue

                    if key != 'type':
                        if key == 'anchors':
                            value = ', '.join(','.join(str(int(i)) for i in j) for j in value)
                        f.write("{}={}\n".format(key, value))
                f.write("\n")