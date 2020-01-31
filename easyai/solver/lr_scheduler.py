import math
from easyai.solver.base_lr_secheduler import BaseLrSecheduler


class LinearIncreaseLR(BaseLrSecheduler):
    def __init__(self, baseLr, endLr, total_iters):
        super().__init__()
        self.baseLr = baseLr
        self.endLr = endLr
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_epoch, cur_iter):
        return self.endLr + (self.baseLr - self.endLr) * (1 - float(cur_iter) / self.total_iters)


class WarmupLinearIncreaseLR(BaseLrSecheduler):
    def __init__(self, baseLr, endLr, total_iters, warm_epoch=0, warmup_iters=2000):
        super().__init__()
        self.baseLr = baseLr
        self.endLr = endLr
        self.warm_epoch = warm_epoch
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_epoch, cur_iter):
        if (cur_epoch == self.warm_epoch) and (cur_iter <= self.warmup_iters):
            lr = self.baseLr * (cur_iter / self.warmup_iters) ** 4
            return lr
        else:
            return self.endLr + (self.baseLr - self.endLr) * (1 - float(cur_iter) / self.total_iters)


class MultiStageLR(BaseLrSecheduler):
    def __init__(self, baseLr, lr_stages):
        super().__init__()
        assert type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2, \
            'lr_stages must be list or tuple, with [iters, lr] format'
        self.baseLr = baseLr
        self._lr_stagess = lr_stages

    def get_lr(self, cur_epoch, cur_iter):
        for it_lr in self._lr_stagess:
            if cur_epoch < it_lr[0]:
                return self.baseLr * it_lr[1]


class WarmupMultiStepLR(BaseLrSecheduler):
    def __init__(self, baseLr, lr_stages, warm_epoch=0, warmup_iters=2000):
        super().__init__()
        self.baseLr = baseLr
        self.warm_epoch = warm_epoch
        self.warmup_iters = warmup_iters
        self._lr_stagess = lr_stages

    def get_lr(self, cur_epoch, cur_iter):
        if (cur_epoch == self.warm_epoch) and (cur_iter <= self.warmup_iters):
            lr = self.baseLr * (cur_iter / self.warmup_iters) ** 4
            return lr
        else:
            for it_lr in self._lr_stagess:
                if cur_epoch < it_lr[0]:
                    return self.baseLr * it_lr[1]


class PolyLR(BaseLrSecheduler):
    def __init__(self, baseLr, total_iters, lr_power=0.9):
        super().__init__()
        self.baseLr = baseLr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_epoch, cur_iter):
        return self.baseLr * ((1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class WarmupPolyLR(BaseLrSecheduler):
    def __init__(self, baseLr, total_iters, lr_power=0.9, warm_epoch=0, warmup_iters=2000):
        super().__init__()
        self.baseLr = baseLr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

        self.warm_epoch = warm_epoch
        self.warmup_iters = warmup_iters

    def get_lr(self, cur_epoch, cur_iter):
        if (cur_epoch == self.warm_epoch) and (cur_iter <= self.warmup_iters):
            lr = self.baseLr * (cur_iter / self.warmup_iters) ** 4
            return lr
        else:
            return self.baseLr * ((1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class CosineLR(BaseLrSecheduler):
    def __init__(self, baseLr, total_iters):
        super().__init__()
        self.baseLr = baseLr
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_epoch, cur_iter):
        return self.baseLr * (1 + math.cos(math.pi * float(cur_iter) / self.total_iters)) / 2


class WarmupCosineLR(BaseLrSecheduler):
    def __init__(self, baseLr, total_iters, warm_epoch=0, warmup_iters=5):
        super().__init__()
        self.baseLr = baseLr
        self.total_iters = total_iters + 0.0

        self.warm_epoch = warm_epoch
        self.warmup_iters = warmup_iters

    def get_lr(self, cur_epoch, cur_iter):
        if (cur_epoch == self.warm_epoch) and (cur_iter <= self.warmup_iters):
            lr = self.baseLr * (cur_iter / self.warmup_iters) ** 4
            return lr
        else:
            return self.baseLr * (1 + math.cos(math.pi * float(cur_iter) / self.total_iters)) / 2
