from easyai.loss.utility.base_loss import *
import numpy as np


class SmoothCrossEntropy(BaseLoss):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    def __init__(self, weight=None, reduction='sum'):
        super().__init__(LossType.SmoothCrossEntropy)
        self.weight = weight
        self.reduction = reduction

    def one_hot(self, y, num_classes):
        y = y.view(-1, 1)
        y_onehot = torch.cuda.FloatTensor(y.shape[0], num_classes)

        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        return y_onehot

    def smooth_label(self, y, num_classes):
        onehot = self.one_hot(y, num_classes)
        onehot = onehot.cpu().numpy()
        uniform_distribution = np.full(num_classes, 1.0 / num_classes)
        deta = 0.01

        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
        smooth_onehot = torch.from_numpy(smooth_onehot)
        smooth_onehot = smooth_onehot.type(torch.cuda.FloatTensor)
        return smooth_onehot

    def forward(self, input_, target):
        logsoftmax = nn.LogSoftmax(dim=1)
        res = -target * logsoftmax(input_)

        if self.weight is not None:
            res = self.weight * res

        if self.reduction == 'elementwise_mean':
            return torch.mean(torch.sum(res, dim=1))
        elif self.reduction == 'sum':
            return torch.sum(torch.sum(res, dim=1))
        else:
            return res