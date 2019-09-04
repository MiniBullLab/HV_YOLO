import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from optparse import OptionParser
from data_loader import *
from utility.torchModelProcess import TorchModelProcess
from config import classifyConfig
from utility.lr_policy import PolyLR

def parse_arguments():

    parser = OptionParser()
    parser.description = "This program train model"

    parser.add_option("-i", "--trainPath", dest="trainPath",
                      metavar="PATH", type="string", default="./train.txt",
                      help="path to data config file")

    parser.add_option("-v", "--valPath", dest="valPath",
                      metavar="PATH", type="string", default="./val.txt",
                      help="path to data config file")

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/yolov3.cfg",
                      help="cfg file path")

    parser.add_option("-p", "--pretrainModel", dest="pretrainModel",
                      metavar="PATH", type="string", default="weights/pretrain.pt",
                      help="path to store weights")

    (options, args) = parser.parse_args()

    if options.trainPath:
        if not os.path.exists(options.trainPath):
            parser.error("Could not find the input train file")
        else:
            options.input_path = os.path.normpath(options.trainPath)
    else:
        parser.error("'trainPath' option is required to run this program")

    return options

def initOptimizer(model, checkpoint):
    start_epoch = 0
    best_mAP = -1
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=classifyConfig.base_lr,\
                                momentum=classifyConfig.momentum, weight_decay=classifyConfig.weight_decay)
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        if checkpoint.get('optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('best_value'):
            best_mAP = checkpoint['best_value']
    return start_epoch, best_mAP, optimizer

def main(trainPath, valPath, cfgPath):

    torchModelProcess = TorchModelProcess()

    model = torchModelProcess.initModel(cfgPath, 0)

    dataloader = ImageClassifyTrainDataLoader(trainPath, batch_size=classifyConfig.train_batch_size,
                                            img_size=(32, 32))

    # set lr_policy
    total_iteration = classifyConfig.maxEpochs * len(dataloader)
    lr_policy = PolyLR(classifyConfig.base_lr, classifyConfig.lr_power, total_iteration)

    checkpoint = None
    if classifyConfig.resume:
        checkpoint = torchModelProcess.loadLatestModelWeight(latest_weights_file, model)
        torchModelProcess.modelTrainInit(model)
    else:
        torchModelProcess.modelTrainInit(model)

    start_epoch, best_value, optimizer = initOptimizer(model, checkpoint)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    for epoch in range(start_epoch, classifyConfig.maxEpochs):

        for i, (imgs, target) in enumerate(dataloader):
            current_idx = epoch * len(dataloader) + i
            lr = lr_policy.get_lr(current_idx)

            for g in optimizer.param_groups:
                g['lr'] = lr

            # compute output
            loss = 0
            
            output = model(imgs.cuda())
            # print(output.shape, target.shape)
            loss = criterion(output, target.cuda())
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % classifyConfig.accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # tensorboard logger
            infotrain = {
                'losstrain': loss.data
            }

            if i % classifyConfig.display == 0:
                print('Epoch: {}/{}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch, classifyConfig.maxEpochs - 1,\
                    idx, len(dataloader), '%.3f' % loss, '%.7f' % optimizer[0].lr, time.time() - t0))
                
                for tag, value in infotrain.items():
                    self.logger.scalar_summary(tag, value, (epoch * (len(dataloader) - 1) + i))

        # remember best prec@1 and save checkpoint
        # if val_prec1 > best_prec1:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': self.model.state_dict(),
        #         'best_prec1': best_prec1,
        #     }, is_best, path=self.savePath)
        #     logging.info('\n Epoch: {0}\t'
        #                  'Training Loss {train_loss:.4f} \t'
        #                  'Training Prec@1 {train_prec1:.3f} \t'
        #                  'Training Prec@5 {train_prec5:.3f} \t'
        #                  'Validation Loss {val_loss:.4f} \t'
        #                  'Validation Prec@1 {val_prec1:.3f} \t'
        #                  'Validation Prec@5 {val_prec5:.3f} \n'
        #                  .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
        #                          train_prec1=train_prec1, val_prec1=val_prec1,
        #                          train_prec5=train_prec5, val_prec5=val_prec5))

if __name__ == "__main__":
    main()
