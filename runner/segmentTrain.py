import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import time
from data_loader.imageSegmentTrainDataLoader import ImageSegmentTrainDataLoader
from solver.lr_scheduler import PolyLR
from torch_utility.torchModelProcess import TorchModelProcess
from torch_utility.torchOptimizer import TorchOptimizer
from utility.logger import AverageMeter, Logger
from helper.arguments_parse import ArgumentsParse
from config import segmentConfig
from runner.segmentTest import SegmentTest

class SegmentTrain():

    def __init__(self, cfg_path, gpu_id):
        if not os.path.exists(segmentConfig.snapshotPath):
            os.makedirs(segmentConfig.snapshotPath, exist_ok=True)

        self.logger = Logger(os.path.join("./log", "logs"))
        self.lossTrain = AverageMeter()

        self.torchModelProcess = TorchModelProcess()
        self.torchOptimizer = TorchOptimizer(segmentConfig.optimizerConfig)
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.start_epoch = 0
        self.bestmIoU = 0

        self.segment_test = SegmentTest(cfg_path, gpu_id)

    def load_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path and os.path.exists(latest_weights_path):
            checkpoint = self.torchModelProcess.loadLatestModelWeight(latest_weights_path, self.model)
            self.torchModelProcess.modelTrainInit(self.model)
        else:
            self.torchModelProcess.modelTrainInit(self.model)
        self.start_epoch, self.bestmIoU = self.torchModelProcess.getLatestModelValue(checkpoint)

        self.torchOptimizer.createOptimizer(self.start_epoch, self.model, segmentConfig.base_lr)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def update_logger(self, step, value):
        self.lossTrain.update(value)
        if (step % segmentConfig.display) == 0:
            # print(lossTrain.avg)
            self.logger.scalar_summary("losstrain", self.lossTrain.avg, step)
            self.lossTrain.reset()

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        targets = targets.to(self.device)
        for k in range(0, loss_count):
            output, target = self.model.lossList[k].segment_resize(output_list[k], targets)
            loss += self.model.lossList[k](output, target)
        return loss


    def test(self, val_path, epoch):
        save_model_path = os.path.join(segmentConfig.snapshotPath, "model_epoch_%d.pt" % epoch)
        self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                               self.optimizer, epoch, self.bestmIoU)
        self.segment_test.load_weights(save_model_path)
        score, class_iou = self.segment_test.test(val_path)
        self.segment_test.save_test_result(epoch, score, class_iou)
        self.bestmIoU = self.torchModelProcess.saveBestModel(score['Mean IoU : \t'],
                                                             save_model_path,
                                                             segmentConfig.best_weights_file)

    def train(self, train_path, val_path):
        dataloader = ImageSegmentTrainDataLoader(train_path, batch_size=segmentConfig.train_batch_size,
                                                 img_size=segmentConfig.imgSize, augment=True)
        self.load_param(segmentConfig.latest_weights_file)

        # set learning policy
        total_iteration = segmentConfig.maxEpochs * len(dataloader)
        polyLR = PolyLR(segmentConfig.base_lr, total_iteration, segmentConfig.lr_power)

        t0 = time.time()
        for epoch in range(self.start_epoch, segmentConfig.maxEpochs):
            # self.optimizer = torchOptimizer.adjust_optimizer(epoch, lr)
            self.optimizer.zero_grad()
            for idx, (imgs, segments) in enumerate(dataloader):

                current_idx = epoch * len(dataloader) + idx
                lr = polyLR.get_lr(epoch, current_idx)
                polyLR.adjust_learning_rate(self.optimizer, lr)

                # Compute loss, compute gradient, update parameters
                output_list = self.model(imgs.to(self.device))
                loss = self.compute_loss(output_list, segments)
                loss.backward()

                # accumulate gradient for x batches before optimizing
                if ((idx + 1) % segmentConfig.accumulated_batches == 0) or (idx == len(dataloader) - 1):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                print(
                    'Epoch: {}/{}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch,
                                                                                     segmentConfig.maxEpochs - 1, \
                                                                                     idx,
                                                                                     len(dataloader),
                                                                                     '%.3f' % loss,
                                                                                     '%.7f' %
                                                                                     self.optimizer.param_groups[0][
                                                                                         'lr'],
                                                                                     time.time() - t0))

                self.update_logger(current_idx, loss.data)

                t0 = time.time()

            self.test(val_path, epoch)


def main():
    print("process start...")
    options = ArgumentsParse.train_input_parse()
    segment_train = SegmentTrain(options.cfg, 0)
    segment_train.train(options.trainPath, options.valPath)
    print("process end!")


if __name__ == "__main__":
    main()
