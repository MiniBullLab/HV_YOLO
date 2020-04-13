import torch


class ModelPrune():
    def __init__(self, cfg_path, gpu_id, global_percent, layer_keep, val_path):
        self.torchModelPrune = TorchModelPrune(global_percent, layer_keep)
        self.torchModelProcess = TorchModelProcess()

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)

        self.cfg_path = cfg_path
        self.pruned_cfg_path = os.path.join('./prune_{}_keep_{}'.format(global_percent, layer_keep), "prune.cfg")

        self.mAP = 0
        self.global_percent = global_percent
        self.layer_keep = layer_keep
        self.val_path = val_path

    def test(self, cfg_path, weight_path, gpu_id, val_path):
        self.detect_test = DetectionTest(cfg_path, gpu_id)
        self.detect_test.load_weights(weight_path)
        self.mAP, self.aps = self.detect_test.test(val_path)

    def prune(self):
        # test mAP before prune
        self.test(self.cfg_path, weight_path, gpu_id, self.val_path)
        mAP_prune_before = self.mAP

        # step 1: parse model to find prune_idx
        CBL_idx, Conv_idx, prune_idx, _, _= self.torchModelPrune.parse_module_defs(self.model.module_defs)

        # step 2: sort bn_weight to get thresh
        bn_weights = self.torchModelPrune.gather_bn_weights(self.model.module_list, prune_idx)
        thresh = self.torchModelPrune.gen_thresh(bn_weights)
        print('Global Threshold should be less than {}.'.format(thresh))

        num_filters, filters_mask = self.torchModelPrune.obtain_filters_mask(self.model, thresh, CBL_idx, prune_idx)
        CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
        CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}

        # add attrib is_access
        for i in self.model.module_defs:
            if i['type'] == 'shortcut':
                i['is_access'] = False

        # step 3: merge the mask of layers connected to shortcut
        print('merge the mask of layers connected to shortcut!')
        self.torchModelPrune.merge_mask(self.model, CBLidx2mask, CBLidx2filters)

        # prune and eval
        for i in CBLidx2mask:
            CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()

        # step 4: add offset of BN beta to following layers
        # 该函数有很重要的意义：
        # ①先用深拷贝将原始模型拷贝下来，得到model_copy
        # ②将model_copy中，BN层中低于阈值的α参数赋值为0
        # ③在BN层中，输出y=α*x+β，由于α参数的值被赋值为0，因此输入仅加了一个偏置β
        # ④很神奇的是，network slimming中是将α参数和β参数都置0，该处只将α参数置0，但效果却很好：其实在另外一篇论文中，已经提到，可以先将β参数的效果移到
        # 下一层卷积层，再去剪掉本层的α参数

        # 该函数用最简单的方法，让我们看到了，如何快速看到剪枝后的效果
        pruned_model = self.torchModelPrune.prune_model_keep_size(self.model, prune_idx, CBL_idx, CBLidx2mask)
        print("now prune the model but keep size,(actually add offset of BN beta to following "
              "layers), let's see how the mAP goes")

        # remove attrib is_access
        for i in self.model.module_defs:
            if i['type'] == 'shortcut':
                i.pop('is_access')

        compact_module_defs = deepcopy(self.model.module_defs)
        for idx in CBL_idx:
            assert compact_module_defs[idx]['type'] == 'convolutional'
            compact_module_defs[idx]['filters'] = str(CBLidx2filters[idx])

        self.torchModelPrune.write_cfg(self.pruned_cfg_path, [self.model.hyperparams.copy()] + compact_module_defs)
        print('Config file has been saved.')

        compact_model = self.torchModelProcess.initModel(self.pruned_cfg_path, gpu_id)
        # 使用prune model上的参数来测试
        self.torchModelPrune.init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

        compact_model_name = '/prune_{}_keep_{}'.format(self.global_percent, self.layer_keep)
        checkpoint = {'epoch': None,
                      'best_mAP': None,
                      'model': compact_model.state_dict(),
                      'optimizer': None}
        torch.save(checkpoint, compact_model_name)
        print('Compact model has been saved.')

        # test mAP after prune
        self.test(self.pruned_cfg_path, compact_model_name, gpu_id, self.val_path)
        mAP_prune_after = self.mAP

        # metric_table = [
        #     ["Metric", "Before", "After"],
        #     ["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric[0][2]:.6f}'],
        #     ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        #     ["Macs", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
        # ]
        # print(AsciiTable(metric_table).table)