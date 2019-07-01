import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

# num_class = 5
# x = torch.randn(3, num_class, requires_grad=True)
# weight = torch.randn(1, num_class)
# y = torch.empty(3, dtype=torch.long).random_(num_class)

def smoothLabel(y, num_classes):
    onehot = one_hot(y, num_classes)
    onehot = onehot.cpu().numpy()
    uniform_distribution = np.full(num_classes, 1.0 / num_classes)
    deta = 0.01
    
    smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
    smooth_onehot = torch.from_numpy(smooth_onehot)
    smooth_onehot = smooth_onehot.type(torch.cuda.FloatTensor)
    return smooth_onehot

def one_hot(y, num_classes):

    y = y.view(-1, 1)
    y_onehot = torch.cuda.FloatTensor(y.shape[0], num_classes)
    
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot


# criterion = nn.CrossEntropyLoss(weight=weight, size_average=False)
#
# def cross_entropy_one_hot(input, target, weight=weight):
#     _, labels = target.max(dim=1)
#     return F.cross_entropy(input, labels, weight=weight, size_average=False)

class cross_entropy(nn.Module):
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
        super(cross_entropy, self).__init__()
        self.weight = weight
        self.reduction = reduction
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

# loss = criterion(x, y)
# print("loss",loss.item())
# loss_1 = cross_entropy_one_hot(x, one_hot(y), weight=weight)
# print("loss_one_hot", loss_1.item())
# ce = cross_entropy(weight=weight, reduction='sum')
# ce.cuda()
# loss_2 = ce(x, one_hot(y))
# print("loss_custom", loss_2.item())

