import torch

class TorchOptimizer():
    def __init__(self, config):
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
        self.config = config

    def getLatestModelOptimizer(self, model, checkpoint):
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,
                                         momentum=0.9, weight_decay=5e-4)
        if checkpoint:
            if checkpoint.get('optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        return self.optimizer

    def adjust_optimizer(self, epoch, lr):
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

        # select the true epoch to adjust the optimizer
        for e in self.config.keys():
            if epoch >= e:
                em = e

        optimizer = modify_optimizer(self.optimizer, self.config[em])
        self.adjustLr(optimizer, lr)

        return optimizer

    def adjustLr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def main():
    from model.modelParse import ModelParse
    optimizerM = {0: {'optimizer': 'SGD',
                     'lr': 1e-2,
                     'momentum': 0.9,
                     'weight_decay': 5e-4},
                 2: {'optimizer': 'Adam',
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
        optimizerMethod = TorchOptimizer(model, epoch, optimizerM)
        optimizerMethod.adjust_optimizer()

if __name__ == "__main__":
    main()