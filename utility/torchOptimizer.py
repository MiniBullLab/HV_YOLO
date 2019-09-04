import torch

from model.modelParse import ModelParse

class torchOptimizer():
    def __init__(self, model, epoch, config):
        self.optimizers = {
            'SGD': torch.optim.SGD,
            'ASGD': torch.optim.ASGD,
            'Adam': torch.optim.Adam,
            'Adamax': torch.optim.Adamax,
            'Adagrad': torch.optim.Adagrad,
            'Adadelta': torch.optim.Adadelta,
            'Rprop': torch.optim.Rprop,
            'RMSprop': torch.optim.RMSprop
        }

        self.model = model
        self.epoch = epoch
        self.config = config
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, \
                                momentum=0.9, weight_decay=5e-4)

    def adjust_optimizer(self):
        """Reconfigures the optimizer according to epoch and config dict"""
        def modify_optimizer(optimizer, setting):
            if 'optimizer' in setting:
                optimizer = self.optimizers[setting['optimizer']](
                    optimizer.param_groups)
                print('OPTIMIZER - setting method = %s' %
                              setting['optimizer'])
            for i_group, param_group in enumerate(optimizer.param_groups):
                for key in param_group.keys():
                    if key in setting:
                        param_group[key] = setting[key]
                        print('OPTIMIZER - group %s setting %s = %s' %
                                  (i_group, key, param_group[key]))
            return optimizer

        # if callable(self.config):
        #     optimizer = modify_optimizer(self.optimizer, self.config(self.epoch), self.epoch)
        # else:

        # select the true epoch to adjust the optimizer
        for e in self.config.keys():
            if self.epoch >= e:
                em = e

        optimizer = modify_optimizer(self.optimizer, self.config[em])

        return optimizer

    #################################################################################################
    # def multiLrMethod(self, model, hyperparams):
    #     stopLayer = "data"
    #     for k, v in hyperparams.items():
    #         if ("conv" in k and v!='None') or ("innerProduct"in k and v!='None'):
    #             stopLayer = k
    #             stopType = v
    #     if stopLayer != "data":
    #         if stopType == "finetune":
    #             print(stopLayer)
    #             for name, p in model.named_parameters():
    #                 #for layerName in stopLayer:
    #                 if stopLayer in name:
    #                     p.requires_grad = True
    #                     break
    #                 else:
    #                     p.requires_grad = False
    #
    #             # define optimizer
    #             optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    #
    #         elif stopType == "diffLRate":
    #             base_params = []
    #             for name, p in model.named_parameters():
    #                 #for layerName in stopLayer:
    #                 if stopLayer in name:
    #                     base_params.append(p)
    #                 else:
    #                     break
    #
    #             base_layer_params = list(map(id, base_params))
    #             special_layers_params = filter(lambda p: id(p) not in base_layer_params, model.parameters())
    #
    #             # define optimizer
    #             optimizer = torch.optim.SGD([{'params': base_params},
    #                                      {'params': special_layers_params, 'lr': 1e-2}], lr=1e-3)
    #
    #     else:
    #         optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    #
    #
    #     return optimizer

def main():
    optimizerM = {0: {'optimizer': 'SGD',
                     'lr': 1e-2,
                     'momentum': 0.9,
                     'weight_decay': 5e-4},
                 2: {'optimizer': 'Adam',
                     'lr': 1e-2,
                     'momentum': 0.9,
                     'weight_decay': 5e-4},
                 4: {'optimizer': 'SGD',
                     'lr': 1e-3,
                     'momentum': 0.9,
                     'weight_decay': 5e-4}
                }

    modelParse = ModelParse()
    model = modelParse.getModel("MobileV2FCN")

    for epoch in range(0, 5):
        print("epoch {}...............".format(epoch))
        optimizerMethod = torchOptimizer(model, epoch, optimizerM)
        optimizerMethod.adjust_optimizer()

if __name__ == "__main__":
    main()