import os
import torch
import time
from optparse import OptionParser
from data_loader import *
from loss.enetLoss import cross_entropy2dDet
from loss.loss import OhemCrossEntropy2d
from loss.focalloss import focalLoss
from utility.lr_policy import PolyLR
from utility.torchModelProcess import TorchModelProcess
from utility.logger import AverageMeter, Logger
from config import configSegment
import segmentTest

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
                      metavar="PATH", type="string", default="cfg/mobileFCN.cfg",
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

def testModel(valPath, cfgPath, weights_path, epoch):
    # test
    score, class_iou = segmentTest.segmentTest(valPath, cfgPath, weights_path)
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

def segmentTrain(trainPath, valPath, cfgPath):
    if not os.path.exists(configSegment.snapshotPath):
        os.makedirs(configSegment.snapshotPath, exist_ok=True)
    latest_weights_file = os.path.join(configSegment.snapshotPath, 'latest.pt')
    best_weights_file = os.path.join(configSegment.snapshotPath, 'best.pt')

    torchModelProcess = TorchModelProcess()

    #logger
    logger = Logger(os.path.join("./weights", "logs"))

    # Get dataloader
    dataloader = ImageSegmentTrainDataLoader(trainPath, batch_size=configSegment.train_batch_size,
                                            img_size=configSegment.imgSize, augment=True)

    # set learning policy
    total_iteration = configSegment.maxEpochs * len(dataloader)
    polyLR = PolyLR(configSegment.base_lr, configSegment.lr_power, total_iteration)

    # model init
    model = torchModelProcess.initModel(cfgPath, 0)

    loss_fn = cross_entropy2dDet(ignore_index=250, size_average=True)
    # min_kept = int(config.train_batch_size // 1 * 640 * 352 // 16)
    # loss_fn = OhemCrossEntropy2d(ignore_index=250, thresh=0.7, min_kept=min_kept).cuda()
    # loss_fn = focalLoss(gamma=0, alpha=None, class_num=2, ignoreIndex=250).cuda()
    # loss_fn = boundedInverseClassLoss

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
            lr = polyLR.get_lr(current_idx)

            for g in optimizer.param_groups:
                g['lr'] = lr

            # Compute loss, compute gradient, update parameters
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
                logger.scalar_summary("losstrain", lossTrain.avg, (epoch * (len(dataloader) - 1) + idx))
                lossTrain.reset()
            t0 = time.time()

        torchModelProcess.saveLatestModel(latest_weights_file, model, optimizer, epoch, bestmIoU)
        score, class_iou = testModel(valPath, cfgPath, latest_weights_file, epoch)
        bestmIoU = saveBestModel(score['Mean IoU : \t'], bestmIoU, latest_weights_file, best_weights_file)

def main():
    print("process start...")
    torch.cuda.empty_cache()
    options = parse_arguments()
    segmentTrain(options.trainPath, options.valPath, options.cfg)
    print("process end!")

if __name__ == "__main__":
    main()