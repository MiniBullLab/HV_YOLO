import os
import random
import torch
import time
import numpy as np
import torch.nn as nn

import config as config
from segmentTest import *
from data_loader import *
from model.mobileV2FCN import MobileFCN
from loss.enetLoss import cross_entropy2dDet
from loss.loss import OhemCrossEntropy2d
from loss.focalloss import focalLoss
from utility.lr_policy import PolyLR
from utility.logger import AverageMeter, Logger

cuda = torch.cuda.is_available()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

def main():
    latest_weights_file = "./weights/mobileFCN.pt"
    logger = Logger(os.path.join("./weights", "logs"))
    # Get dataloader
    dataloader = ImageSegmentTrainDataLoader("/home/wfw/data/VOCdevkit/CULane/ImageSets/train.txt", batch_size=config.train_batch_size,
                                            img_size=[640, 352], augment=True)

    # set lr_policy
    total_iteration = config.maxEpochs * len(dataloader)
    lr_policy = PolyLR(config.base_lr, config.lr_power, total_iteration)

    print("Train data num: {}".format(dataloader.nF))

    # init model
    model = MobileFCN("./cfg/mobileFCN.cfg", img_size=[640, 352])

    # freeze bn
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()

    # set optimize
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.base_lr,
                                    momentum=config.momentum, weight_decay=config.weight_decay)

    # loss_fn = cross_entropy2dDet(ignore_index=250, size_average=True)
    # min_kept = int(config.train_batch_size // 1 * 640 * 352 // 16)
    # loss_fn = OhemCrossEntropy2d(ignore_index=250, thresh=0.7, min_kept=min_kept).cuda()
    loss_fn = focalLoss(gamma=0, alpha=None, class_num=2, ignoreIndex=250).cuda()
    # focalLoss = focalLoss
    # boundedInverseClassLoss = boundedInverseClassLoss

    if len(config.CUDA_DEVICE) > 1:
        print('Using {} GPUS'.format(len(config.CUDA_DEVICE)))
        model = nn.DataParallel(model, device_ids=config.CUDA_DEVICE)
    model.cuda()

    if config.pretainModel is not None:
        if not os.path.exists(config.pretainModel):
            raise Exception("Invaid path", config.pretainModel)
        print("Loading pretainModel from {}".format(config.pretainModel))
        model.load_state_dict(torch.load(config.pretainModel), strict=True)

    if config.resume is not None:
        if not os.path.exists(config.resume):
            raise Exception("Invaid path", config.resume)
        print("Loading resume from {}".format(config.resume))
        if len(config.CUDA_DEVICE) > 1:
            checkpoint = torch.load(latest_weights_file, map_location='cpu')
            state = convert_state_dict(checkpoint['model'])
            model.load_state_dict(state)
        else:
            checkpoint = torch.load(latest_weights_file, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

        #if checkpoint['optimizer'] is not None:# and \
                #len(optimizer.param_groups) == len(checkpoint['optimizer']):
            #logging.info("Loading optimizer param")
            #optimizer.load_state_dict(checkpoint['optimizer'])

        bestmIoU = checkpoint['best_mIoU']

        start_epoch = checkpoint['epoch'] + 1
        print("Epoch: {}, mIoU: {}".format(checkpoint['epoch'], bestmIoU))
    else:
        bestmIoU = -100
        start_epoch = 0

    # summary the model
    # model_info(model)

    lossTrain = AverageMeter()

    # if config.evaluate == True:
    #     logging.info("Evalution---------------------------")
    #     [mAP, aps, score, class_iou] = evalModel(model.state_dict())
    #     logging.info("++ mAP: {}, mIou: {}".format(mAP, score['Mean IoU : \t']))

    t0 = time.time()
    for epoch in range(start_epoch, config.maxEpochs):

        model.train()

        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
        # if config.optimType == 'multiStep':
        #     if epoch > 70:
        #         lr = config.base_lr / 100
        #     elif epoch > 50: # 50 for no_balance_sample
        #         lr = config.base_lr / 10
        #     else:
        #         lr = config.base_lr

        optimizer.zero_grad()
        for idx, (imgs, segments) in enumerate(dataloader):

            current_idx = epoch * len(dataloader) + idx
            lr = lr_policy.get_lr(current_idx)

            # SGD burn-in
            if False:
                if (epoch == 0) & (idx <= 1000):
                    lr = config.base_lr * (idx / 1000) ** 4
                    for g in optimizer.param_groups:
                        g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            lossSeg = 0
            output = model(imgs.cuda())

            loss = loss_fn(output, segments.cuda())
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((idx + 1) % config.accumulated_batches == 0) or (idx == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            print('Epoch: {}/{}[{}/{}]\t Loss: {}\t LossSeg: {}\t Rate: {} \t Time: {}\t'.format(epoch, config.maxEpochs - 1,\
                   idx, len(dataloader), '%.3f' % loss, '%.3f' % lossSeg, '%.7f' % lr, time.time() - t0))

            lossTrain.update(loss.data)
            if (epoch * (len(dataloader) - 1) + idx) % config.display == 0:
                print(lossTrain.avg)
                logger.scalar_summary("losstrain", lossTrain.avg, (epoch * (len(dataloader) - 1) + idx))
                lossTrain.reset()
            t0 = time.time()

        # test
        score, class_iou = evalModel(model.state_dict())
        print("++ Evalution Epoch: {}, mIou: {}, Lane IoU: {}".format(epoch, score['Mean IoU : \t'], class_iou[1]))

        # checkpoint = {'epoch': epoch,
        #               'best_mIoU': score['Mean IoU : \t'],
        #               'model': model.state_dict(),
        #               'optimizer': optimizer.state_dict()}
        # torch.save(checkpoint, latest_weights_file)

        # Save best checkpoint
        if score['Mean IoU : \t'] > bestmIoU:
            bestmIoU = score['Mean IoU : \t']
            print("++ Saving model: Epoch: {}, mIou: {}".format(epoch, bestmIoU))
            checkpoint = {'epoch': epoch,
                          'best_mIoU': bestmIoU,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, latest_weights_file)

        # # Save backup weights every 5 epochs
        # if (epoch > 0) & (epoch % 5 == 0):
        #     backup_file_name = 'backup{}.pt'.format(epoch)
        #     backup_file_path = os.path.join(config.snapshotPath, backup_file_name)
        #     os.system('cp {} {}'.format(
        #         latest_weights_file,
        #         backup_file_path,
        #     ))

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write("Epoch: {} | mIoU: {:.3f}".format(epoch, score['Mean IoU : \t']))
            file.write("\n")

if __name__ == "__main__":
    main()
