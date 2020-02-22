from easyai.loss.utility.base_loss import *


class FocalLoss(BaseLoss):
    def __init__(self, gamma=0, alpha=None, class_num=0, ignoreIndex=None, size_average=True):
        super().__init__(LossType.FocalLoss)
        if alpha is None:
            alpha = torch.ones(class_num, 1)

        self.gamma = gamma
        self.alpha = alpha
        self.class_num = class_num
        self.ignoreIndex = ignoreIndex
        self.size_average = size_average

    def forward(self, input, target):
        P = F.softmax(input, dim=1)
        if input.dim() > 2:
            P = P.transpose(1, 2).transpose(2, 3).contiguous().view(-1, self.class_num)

        ids = target.view(-1, 1)
        if self.ignoreIndex != None:
            P = P[(ids != self.ignoreIndex).expand_as(P)].view(-1, self.class_num)
            ids = ids[ids != self.ignoreIndex].view(-1, 1)

        class_mask = torch.zeros(P.shape)
        class_mask.scatter_(1, ids.cpu(), 1.)

        if input.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
            class_mask = class_mask.cuda()
            self.alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -self.alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class FocalBinaryLoss(BaseLoss):
    def __init__(self, gamma=0, alpha=None, reduce=False):
        super().__init__(LossType.FocalBinaryLoss)

        self.gamma = gamma
        self.alpha = alpha
        self.reduce = reduce

        self.bce = nn.BCELoss(reduce=self.reduce)

    def forward(self, input, target):

        if self.alpha is None:
            self.alpha = torch.ones(input.shape).type(torch.cuda.FloatTensor)

        loss = self.alpha * (torch.pow(torch.abs(target - input), self.gamma)) * self.bce(input, target)

        return loss