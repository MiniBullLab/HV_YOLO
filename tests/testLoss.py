import torch
import torch.nn as nn
from easy_ai.loss import FocalLoss, FocalBinaryLoss

if __name__ == "__main__":
    # N = 4
    # C = 5
    # CE = nn.CrossEntropyLoss()
    FL = FocalLoss(gamma=2, alpha=None, class_num=1, ignoreIndex=None, size_average=True)
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
    fl_bce = FocalBinaryLoss(gamma=2, reduce=False)

    bce_loss = bce(inputSigmoid, target)
    fl_bce = fl_bce(inputSigmoid, target)
    # fl_loss = FL(input, target.type(torch.LongTensor))
    print(inputSigmoid)
    print(target)

    print('bce = {}\nfl_bce = {}\n'.format(bce_loss, fl_bce))