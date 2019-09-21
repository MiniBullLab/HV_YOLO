import os
import time
from optparse import OptionParser
import torch
from data_loader import *
from torch_utility.torchModelProcess import TorchModelProcess
from torch_utility.torchOptimizer import TorchOptimizer
from torch_utility.model_summary import summary
from torch_utility.lr_policy import MultiStageLR
from utility.logger import AverageMeter, Logger
from config import detectConfig
import detectTest

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

def testModel(valPath, cfgPath, weights_path, epoch):
    # Calculate mAP
    mAP, aps = detectTest.detectTest(valPath, cfgPath, weights_path)

    # Write epoch results
    with open('results.txt', 'a') as file:
        # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
        file.write("Epoch: {} | mAP: {:.3f} | ".format(epoch, mAP))
        for i, ap in enumerate(aps):
            file.write(detectConfig.className[i] + ": {:.3f} ".format(ap))
        file.write("\n")

    return mAP, aps

def detectTrain(trainPath, valPath, cfgPath):

    if not os.path.exists(detectConfig.snapshotPath):
        os.makedirs(detectConfig.snapshotPath, exist_ok=True)

    torchModelProcess = TorchModelProcess()
    torchOptimizer = TorchOptimizer(detectConfig.optimizerConfig)

    #logger
    logger = Logger(os.path.join("./log", "logs"))
    #set learning policy
    multiLR = MultiStageLR([[49, detectConfig.base_lr * 0.1], [70, detectConfig.base_lr * 0.01]])
    #model init
    model = torchModelProcess.initModel(cfgPath, 0)
    #loss init
    lossTrain = AverageMeter()
    #get dataloader
    dataloader = ImageDetectTrainDataLoader(trainPath, detectConfig.className,
                                            detectConfig.train_batch_size, detectConfig.imgSize,
                                            multi_scale=False, augment=True, balancedSample=True)

    avg_loss = -1
    checkpoint = None
    if detectConfig.resume:
        checkpoint = torchModelProcess.loadLatestModelWeight(detectConfig.latest_weights_file, model)
        torchModelProcess.modelTrainInit(model)
    else:
        torchModelProcess.modelTrainInit(model)
    start_epoch, best_mAP = torchModelProcess.getLatestModelValue(checkpoint)
    optimizer = torchOptimizer.getLatestModelOptimizer(model, checkpoint)

    # summary(model, [1, 3, 640, 352])
    t0 = time.time()
    for epoch in range(start_epoch, detectConfig.maxEpochs):

        # get learning rate
        lr = multiLR.get_lr(epoch)
        optimizer = torchOptimizer.adjust_optimizer(epoch, lr)
        optimizer.zero_grad()
        for i, (imgs, targets) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) and (i <= 1000):
                lr = detectConfig.base_lr * (i / 1000) ** 4
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss = 0
            output = model(imgs.to(torchModelProcess.getDevice()))
            for k in range(0, 3):
                loss += model.lossList[k](output[k], targets)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % detectConfig.accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            if avg_loss < 0:
                avg_loss = (loss.cpu().detach().numpy() / detectConfig.train_batch_size)
            avg_loss = 0.9 * (loss.cpu().detach().numpy() / detectConfig.train_batch_size) + 0.1 * avg_loss
            print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch, i, len(dataloader), '%.3f' % avg_loss, '%.7f' % optimizer.param_groups[0]['lr'], time.time() - t0))

            lossTrain.update(loss.data)
            if (epoch * (len(dataloader) - 1) + i) % detectConfig.display == 0:
                # print(lossTrain.avg)
                logger.scalar_summary("losstrain", lossTrain.avg, (epoch * (len(dataloader) - 1) + i))
                lossTrain.reset()

            t0 = time.time()

        torchModelProcess.saveLatestModel(detectConfig.latest_weights_file, model,
                                          optimizer, epoch, best_mAP)
        mAP, aps = testModel(valPath, cfgPath, detectConfig.latest_weights_file, epoch)
        best_mAP = torchModelProcess.saveBestModel(mAP, detectConfig.latest_weights_file,
                                        detectConfig.best_weights_file)

def main():
    print("process start...")
    options = parse_arguments()
    detectTrain(options.trainPath, options.valPath, options.cfg)
    print("process end!")

if __name__ == '__main__':
    main()
