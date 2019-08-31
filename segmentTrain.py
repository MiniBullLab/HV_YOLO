import os
import random
import torch
import time
import numpy as np
import torch.nn as nn
from data_loader import *
from model.modelParse import ModelParse
from loss.enetLoss import cross_entropy2dDet
from loss.loss import OhemCrossEntropy2d
from loss.focalloss import focalLoss
from utility.lr_policy import PolyLR
from utility.torchDeviceProcess import TorchDeviceProcess
from utility.torchModelProcess import TorchModelProcess
from utility.logger import AverageMeter, Logger
from config import configSegment
import segmentTest

def testModel(valPath, img_size, cfg, epoch):
    # test
    weights_path = 'snapshot/latest.pt'
    score, class_iou = segmentTest.main(cfg, weights_path, img_size, valPath)
    print("++ Evalution Epoch: {}, mIou: {}, Lane IoU: {}".format(epoch, score['Mean IoU : \t'], class_iou[1]))

    # Write epoch results
    with open('results.txt', 'a') as file:
        # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
        file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU : \t']))
        for i, iou in enumerate(class_iou):
            file.write(configSegment.className[i] + ": {:.3f} ".format(iou))
        file.write("\n")

    return score, class_iou

def initOptimizer(model, checkpoint):
    start_epoch = 0
    bestmIoU = -1
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=configSegment.base_lr,
                                momentum=configSegment.momentum, weight_decay=configSegment.weight_decay)
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        if checkpoint.get('optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('best_value'):
            bestmIoU = checkpoint['best_value']
    return start_epoch, bestmIoU, optimizer

def saveBestModel(score, bestmIoU, latest_weights_file, best_weights_file):
    if score >= bestmIoU:
        bestmIoU = score
        os.system('cp {} {}'.format(
            latest_weights_file,
            best_weights_file,
        ))
    return bestmIoU

def train():

    os.makedirs(configSegment.snapshotPath, exist_ok=True)
    latest_weights_file = os.path.join(configSegment.snapshotPath, 'latest.pt')
    best_weights_file = os.path.join(configSegment.snapshotPath, 'best.pt')
    logger = Logger(os.path.join("./weights", "logs"))

    # Configure run
    train_path = configSegment.trainPath
    valPath = configSegment.valPath

    torchModelProcess = TorchModelProcess()
    # Get dataloader
    dataloader = ImageSegmentTrainDataLoader(train_path, batch_size=configSegment.train_batch_size,
                                            img_size=(640, 352), augment=True)

    # set lr_policy
    total_iteration = configSegment.maxEpochs * len(dataloader)
    lr_policy = PolyLR(configSegment.base_lr, configSegment.lr_power, total_iteration)
    print("Train data num: {}".format(dataloader.nF))

    # model init
    model = torchModelProcess.initModel(configSegment.net_config_path, 0)

    loss_fn = cross_entropy2dDet(ignore_index=250, size_average=True)
    # min_kept = int(config.train_batch_size // 1 * 640 * 352 // 16)
    # loss_fn = OhemCrossEntropy2d(ignore_index=250, thresh=0.7, min_kept=min_kept).cuda()
    # loss_fn = focalLoss(gamma=0, alpha=None, class_num=2, ignoreIndex=250).cuda()
    # focalLoss = focalLoss
    # boundedInverseClassLoss = boundedInverseClassLoss

    # if torch.cuda.device_count() > 1:
    #     print('Using ', torch.cuda.device_count(), ' GPUs')
    #     model = nn.DataParallel(model)
    # model.to(device).train()

    lossTrain = AverageMeter()

    checkpoint = None
    if configSegment.resume is not None:
        checkpoint = torchModelProcess.loadLatestModelWeight(latest_weights_file, model)
        torchModelProcess.modelTrainInit(model)
    else:
        torchModelProcess.modelTrainInit(model)
    start_epoch, bestmIoU, optimizer = initOptimizer(model, checkpoint)

    # summary the model
    # model_info(model)

    t0 = time.time()
    for epoch in range(start_epoch, configSegment.maxEpochs):
        optimizer.zero_grad()
        for idx, (imgs, segments) in enumerate(dataloader):

            current_idx = epoch * len(dataloader) + idx
            lr = lr_policy.get_lr(current_idx)

            for g in optimizer.param_groups:
                g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss = 0
            output = model(imgs.cuda())[0]

            loss = loss_fn(output, segments.cuda())
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((idx + 1) % configSegment.accumulated_batches == 0) or (idx == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            print('Epoch: {}/{}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch, configSegment.maxEpochs - 1,\
                   idx, len(dataloader), '%.3f' % loss, '%.7f' % optimizer.param_groups[0]['lr'], time.time() - t0))

            lossTrain.update(loss.data)
            if (epoch * (len(dataloader) - 1) + idx) % configSegment.display == 0:
                # print(lossTrain.avg)
                logger.scalar_summary("losstrain", lossTrain.avg, (epoch * (len(dataloader) - 1) + idx))
                lossTrain.reset()
            t0 = time.time()

        torchModelProcess.saveLatestModel(latest_weights_file, model, optimizer, epoch, bestmIoU)
        score, class_iou = testModel(valPath, configSegment.imgSize, configSegment.net_config_path, epoch)
        bestmIoU = saveBestModel(score['Mean IoU : \t'], bestmIoU, latest_weights_file, best_weights_file)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
