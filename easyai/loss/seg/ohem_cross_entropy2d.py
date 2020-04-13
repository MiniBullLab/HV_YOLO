from easyai.loss.utility.base_loss import *
import numpy as np


class OhemCrossEntropy2d(BaseLoss):

    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=int(32 // 1 * 640 * 352 // 16)):
        super().__init__(LossType.OhemCrossEntropy2d)
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def compute_ohem_loss(self, input_data, target):
        n, c, h, w = input_data.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(input_data, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                # index = mask_prob.argsort()
                index = np.argsort(mask_prob.cpu().detach().numpy())
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(n, h, w)
        return input_data, target

    def forward(self, input_data, target):
        if target is not None:
            input_data, target = self.compute_ohem_loss(input_data, target)
            loss = self.loss_function(input_data, target)
        else:
            loss = F.softmax(input_data)
        return loss


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):

    def __init__(self, aux=False, aux_weight=0.4, weight=None, min_kept=100000, ignore_index=-1, **kwargs):
        super().__init__(min_kept=min_kept, ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))