from easyai.loss.base_loss import *
import numpy as np

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


class CrossEntropy2d(BaseLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=250):
        super().__init__(LossType.CrossEntropy2d)
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.ignore_index,
                                                 weight=self.weight,
                                                 size_average=self.size_average)

    def segment_resize(self, input, target):
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

        return input, target[mask]

    def forward(self, input, target=None):
        if target is not None:
            loss = self.loss_function(input, target)
        else:
            loss = F.softmax(input, dim=1)
        return loss

