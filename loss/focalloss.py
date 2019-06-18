import torch
import torch.nn as nn
import torch.nn.functional as F

class focalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, class_num=0, ignoreIndex=None, size_average=True):
        super(focalLoss, self).__init__()
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

class focalBinaryLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduce=False):
        super(focalBinaryLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduce = reduce

        self.bce = nn.BCELoss(reduce=self.reduce)

    def forward(self, input, target):

        if self.alpha is None:
            self.alpha = torch.ones(input.shape).type(torch.cuda.FloatTensor)

        loss = self.alpha * (torch.pow(torch.abs(target - input), self.gamma)) * self.bce(input, target)

        return loss

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = logpt.data.exp()
#
#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             # at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * self.alpha
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # N = 4
    # C = 5
    # CE = nn.CrossEntropyLoss()
    FL = focalLoss(gamma=2, alpha=None, class_num=1, ignoreIndex=None, size_average=True)
    # FLS = FocalLoss(gamma=2, alpha=torch.ones(N, 1), size_average=True)
    # inputs = torch.rand(N, C)
    # targets = torch.LongTensor(N).random_(C)
    #
    # print('----inputs----')
    # print(inputs)
    # print('---target-----')
    # print(targets)
    #
    # fl_loss = FL(inputs, targets)
    # ce_loss = CE(inputs, targets)
    # fls_loss = FLS(inputs, targets)
    # print('ce = {}, fl ={}, fls = {}'.format(ce_loss.data[0], fl_loss.data[0], fls_loss.data[0]))


    input = torch.rand(1, 1)
    inputSigmoid = input.sigmoid()
    target = torch.FloatTensor(1).random_(2)
    bce = nn.BCELoss(reduce=False)
    fl_bce = focalBinaryLoss(gamma=2, reduce=False)

    bce_loss = bce(inputSigmoid, target)
    fl_bce = fl_bce(inputSigmoid, target)
    # fl_loss = FL(input, target.type(torch.LongTensor))
    print(inputSigmoid)
    print(target)

    print('bce = {}\nfl_bce = {}\n'.format(bce_loss, fl_bce))