from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler


class LRStepScheduler(_LRScheduler):
    def __init__(self, optimizer, steps, last_epoch=-1):
        self.lr_steps = steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        pos = max(bisect_right([x for x, y in self.lr_steps], self.last_epoch) - 1, 0)
        return [self.lr_steps[pos][1] if self.lr_steps[pos][0] <= self.last_epoch else base_lr for base_lr in self.base_lrs]


class PolyLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to poly learning rate policy
    """
    def __init__(self, optimizer, max_iter=90000, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.last_epoch = (self.last_epoch + 1) % self.max_iter
        return [base_lr * ((1 - float(self.last_epoch) / self.max_iter) ** (self.power)) for base_lr in self.base_lrs]

class ExponentialLRScheduler(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(ExponentialLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= 0:
            return self.base_lrs
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]

