import torch
import torch.nn as nn

def cross_entropy2d(input, target, weight=None, ignore_index = -1, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index,
                      weight=weight, size_average=size_average)
    loss = loss_fn(input, target)
    #if size_average:
    #    loss = loss / mask.sum().item()

    return loss

class cross_entropy2dDet(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=250):
        super(cross_entropy2dDet, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h > ht and w > wt:  # upsample labels
            target = target.unsequeeze(1)
            target = F.upsample(target, size=(h, w), mode='nearest')
            target = target.sequeeze(1)
        elif h < ht and w < wt:  # upsample images
            input = F.upsample(input, size=(ht, wt), mode='bilinear')
        elif h != ht and w != wt:
            raise Exception("Only support upsampling")

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index,
                                      weight=self.weight, size_average=self.size_average)
        loss = loss_fn(input, target)
        # if size_average:
        #    loss = loss / mask.sum().item()

        return loss

def enet_weighing(labels, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    for label in labels:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    class_count = class_count[:num_classes]
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


# def focalLoss(input, target, class_num, ignoreIndex=None, alpha=None, gamma=2, size_average=True):
#     if alpha is None:
#         alpha = Variable(torch.ones(class_num, 1))
#     else:
#         if isinstance(alpha, Variable):
#             alpha = alpha
#         else:
#             alpha = Variable(alpha)
#
#     P = F.softmax(input, dim=1)
#     P = P.transpose(1, 2).transpose(2, 3).contiguous().view(-1, class_num)
#
#     ids = target.view(-1, 1)
#     if ignoreIndex != None:
#         P = P[(ids != ignoreIndex).expand_as(P)].view(-1, class_num)
#         ids = ids[ids != ignoreIndex].view(-1, 1)
#
#     class_mask = Variable(torch.zeros(P.shape))
#     class_mask.scatter_(1, ids.cpu(), 1.)
#
#     if input.is_cuda and not alpha.is_cuda:
#         alpha = alpha.cuda()
#         class_mask = class_mask.cuda()
#     alpha = alpha[ids.data.view(-1)]
#
#     probs = (P * class_mask).sum(1).view(-1, 1)
#     log_p = probs.log()
#     batch_loss = -log_p
#
#     if size_average:
#         loss = batch_loss.mean()
#     else:
#         loss = batch_loss.sum()
#
#     return loss