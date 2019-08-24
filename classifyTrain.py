import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_loader import *
from torch.utils.data import Dataset, DataLoader
from model.modelParse import ModelParse
import config as config
from utility.lr_policy import PolyLR

cuda = torch.cuda.is_available()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

def main():
    dataloader = ImageClassifyTrainDataLoader(config.trainList, batch_size=config.train_batch_size,
                                            img_size=[32, 32])

    # set lr_policy
    total_iteration = config.maxEpochs * len(dataloader)
    lr_policy = PolyLR(config.base_lr, config.lr_power, total_iteration)

    # Initialize model
    modelParse = ModelParse()
    model = modelParse.parse("./cfg/cifar100.cfg")

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

        bestPrec1 = checkpoint['best_prec1']

        start_epoch = checkpoint['epoch'] + 1
        print("Epoch: {}, bestPrec1: {}".format(checkpoint['epoch'], bestPrec1))
    else:
        bestPrec1 = -100
        start_epoch = 0

    # set optimize
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.base_lr,
                                    momentum=config.momentum, weight_decay=config.weight_decay)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    model.cuda()
    model.train()

    optimizer.zero_grad()
    for epoch in range(start_epoch, config.maxEpochs):

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
            if ((i + 1) % config.accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # tensorboard logger
            infotrain = {
                'losstrain': loss.data
            }

            if i % config.display == 0:
                print('Epoch: {}/{}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch, config.maxEpochs - 1,\
                    idx, len(dataloader), '%.3f' % loss, '%.7f' % optimizer[0].lr, time.time() - t0))
                
                for tag, value in infotrain.items():
                    self.logger.scalar_summary(tag, value, (epoch * (len(data_loader) - 1) + i))

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
