import os
import random
import torch
import time
import numpy as np
import torch.nn as nn

import config as config
from segmentTest import *
from data_loader import *
from model.modelParse import ModelParse
from loss.enetLoss import cross_entropy2dDet
from loss.loss import OhemCrossEntropy2d
from loss.focalloss import focalLoss
from utility.lr_policy import PolyLR
from utility.torchModelProcess import TorchModelProcess
from utility.logger import AverageMeter, Logger

import segmentTest

cuda = torch.cuda.is_available()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

def testModel(valPath, img_size, cfg, epoch):
    # test
    weights_path = 'weights/latest.pt'
    score, class_iou = segmentTest.main(cfg, weights_path, img_size, valPath)
    print("++ Evalution Epoch: {}, mIou: {}, Lane IoU: {}".format(epoch, score['Mean IoU : \t'], class_iou[1]))


    # Write epoch results
    with open('results.txt', 'a') as file:
        # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
        file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU : \t']))
        for i, iou in enumerate(class_iou):
            file.write(config.className[i] + ": {:.3f} ".format(iou))
        file.write("\n")

    return score, class_iou

def train():
    device = "cuda:0"
    print("Using device: \"{}\"".format(device))

    os.makedirs(config.snapshotPath, exist_ok=True)
    latest_weights_file = os.path.join(config.snapshotPath, 'latest.pt')
    best_weights_file = os.path.join(config.snapshotPath, 'best.pt')
    logger = Logger(os.path.join("./weights", "logs"))

    # Configure run
    train_path = config.trainPath
    valPath = config.valPath

    modelProcess = TorchModelProcess()
    # Get dataloader
    dataloader = ImageSegmentTrainDataLoader(train_path, batch_size=config.train_batch_size,
                                            img_size=(640, 352), augment=True)

    # set lr_policy
    total_iteration = config.maxEpochs * len(dataloader)
    lr_policy = PolyLR(config.base_lr, config.lr_power, total_iteration)
    print("Train data num: {}".format(dataloader.nF))

    # init model
    modelParse = ModelParse()
    model = modelParse.parse("./cfg/mobileFCN.cfg")

    loss_fn = cross_entropy2dDet(ignore_index=250, size_average=True)
    # min_kept = int(config.train_batch_size // 1 * 640 * 352 // 16)
    # loss_fn = OhemCrossEntropy2d(ignore_index=250, thresh=0.7, min_kept=min_kept).cuda()
    # loss_fn = focalLoss(gamma=0, alpha=None, class_num=2, ignoreIndex=250).cuda()
    # focalLoss = focalLoss
    # boundedInverseClassLoss = boundedInverseClassLoss

    if torch.cuda.device_count() > 1:
        print('Using ', torch.cuda.device_count(), ' GPUs')
        model = nn.DataParallel(model)
    model.to(device).train()

    lossTrain = AverageMeter()
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
            state = modelProcess.convert_state_dict(checkpoint['model'])
            model.load_state_dict(state)
        else:
            checkpoint = torch.load(latest_weights_file, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.base_lr,
                                    momentum=config.momentum, weight_decay=config.weight_decay)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            bestmIoU = checkpoint['best_mIoU']

        del checkpoint  # current, saved
        print("Epoch: {}, mIoU: {}".format(checkpoint['epoch'], bestmIoU))

    else:
        start_epoch = 0
        bestmIoU = -1

        # set optimize
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.base_lr,
                                    momentum=config.momentum, weight_decay=config.weight_decay)

    # summary the model
    # model_info(model)

    t0 = time.time()
    for epoch in range(start_epoch, config.maxEpochs):

        model.train()

        optimizer.zero_grad()
        for idx, (imgs, segments) in enumerate(dataloader):

            current_idx = epoch * len(dataloader) + idx
            lr = lr_policy.get_lr(current_idx)

            for g in optimizer.param_groups:
                g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            lossSeg = 0
            output = model(imgs.cuda())[0]

            loss = loss_fn(output, segments.cuda())
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((idx + 1) % config.accumulated_batches == 0) or (idx == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            print('Epoch: {}/{}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch, config.maxEpochs - 1,\
                   idx, len(dataloader), '%.3f' % loss, '%.7f' % optimizer.param_groups[0].lr, time.time() - t0))

            lossTrain.update(loss.data)
            if (epoch * (len(dataloader) - 1) + idx) % config.display == 0:
                # print(lossTrain.avg)
                logger.scalar_summary("losstrain", lossTrain.avg, (epoch * (len(dataloader) - 1) + idx))
                lossTrain.reset()
            t0 = time.time()

        score, class_iou = testModel(valPath, config.imgSize, config.net_config_path, epoch)

        checkpoint = {'epoch': epoch,
                      'best_mIoU': score['Mean IoU : \t'],
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest_weights_file)

        # Save best checkpoint
        if score['Mean IoU : \t'] > bestmIoU:
            bestmIoU = score['Mean IoU : \t']
            print("++ Saving model: Epoch: {}, mIou: {}".format(epoch, bestmIoU))
            checkpoint = {'epoch': epoch,
                          'best_mIoU': bestmIoU,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, latest_weights_file)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
