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
        self.optimizer = None

    def createOptimizer(self, epoch, model, base_lr):
        em = 0
        for e in self.config.keys():
            if epoch >= e:
                em = e
        setting = self.config[em]
        self.optimizer = self.optimizers[setting['optimizer']](filter(lambda p: p.requires_grad, model.parameters()),
                                                          lr=base_lr)
        self.adjust_param(self.optimizer, setting)
        self.adjust_lr(self.optimizer, base_lr)

    def adjust_optimizer(self, epoch, lr):
        # select the true epoch to adjust the optimizer
        em = 0
        for e in self.config.keys():
            if epoch >= e:
                em = e
        setting = self.config[em]
        optimizer = self.modify_optimizer(self.optimizer, setting)
        self.adjust_param(optimizer, setting)
        self.adjust_lr(optimizer, lr)
        return optimizer

    def getLatestModelOptimizer(self, checkpoint):
        if checkpoint:
            if checkpoint.get('optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self.optimizer

    def modify_optimizer(self, optimizer, setting):
        result = None
        if 'optimizer' in setting:
            result = self.optimizers[setting['optimizer']](optimizer.param_groups)
            print('OPTIMIZER - setting method = %s' % setting['optimizer'])
        return result

    def adjust_param(self, optimizer, setting):
        for i_group, param_group in enumerate(optimizer.param_groups):
            for key in param_group.keys():
                if key in setting:
                    param_group[key] = setting[key]
                    print('OPTIMIZER - group %s setting %s = %s' %
                          (i_group, key, param_group[key]))

    def adjust_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == "__main__":
    pass