import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import os
import time
import torch.nn as nn
from helper.arguments_parse import ArgumentsParse
from data_loader.imageClassifyTrainDataLoard import ImageClassifyTrainDataLoader
from torch_utility.torch_model_process import TorchModelProcess
from solver.torch_optimizer import TorchOptimizer
from solver.lr_scheduler import MultiStageLR
from utility.logger import AverageMeter, Logger
from config import classifyConfig
from runner.classifyTest import ClassifyTest


class ClassifyTrain():

    def __init__(self, cfg_path, gpu_id):
        self.logger = Logger(os.path.join("./log", "logs"))
        self.lossTrain = AverageMeter()

        self.torchModelProcess = TorchModelProcess()
        self.torchOptimizer = TorchOptimizer(classifyConfig.optimizerConfig)
        self.multiLR = MultiStageLR(classifyConfig.base_lr, [[50, 1], [70, 0.1], [100, 0.01]])
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.start_epoch = 0
        self.best_prec = 0

        self.classify_test = ClassifyTest(cfg_path, gpu_id)

    def load_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path and os.path.exists(latest_weights_path):
            checkpoint = self.torchModelProcess.loadLatestModelWeight(latest_weights_path, self.model)
            self.torchModelProcess.modelTrainInit(self.model)
        else:
            self.torchModelProcess.modelTrainInit(self.model)
        self.start_epoch, self.best_prec = self.torchModelProcess.getLatestModelValue(checkpoint)

        self.torchOptimizer.createOptimizer(self.start_epoch, self.model, classifyConfig.base_lr)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def update_logger(self, step, value):
        self.lossTrain.update(value)
        if step % classifyConfig.display == 0:
            # print(lossTrain.avg)
            self.logger.scalar_summary("losstrain", self.lossTrain.avg, step)
            self.lossTrain.reset()

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        targets = targets.to(self.device)
        for k in range(0, loss_count):
            loss += self.model.lossList[k](output_list[k], targets)
        return loss

    def test(self, val_path, epoch):
        save_model_path = os.path.join(classifyConfig.snapshotPath, "model_epoch_%d.pt" % epoch)
        self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                               self.optimizer, epoch, self.best_prec)
        self.classify_test.load_weights(save_model_path)
        prec1, prec5 = self.classify_test.test(val_path)
        self.classify_test.save_test_result(epoch, prec1, prec5)
        self.best_prec = self.torchModelProcess.saveBestModel(prec1, save_model_path,
                                                              classifyConfig.best_weights_file)

    def train(self, train_path, val_path):
        dataloader = ImageClassifyTrainDataLoader(train_path,
                                                  batch_size=classifyConfig.train_batch_size,
                                                  img_size=classifyConfig.imgSize)
        total_images = len(dataloader)
        self.load_param(classifyConfig.latest_weights_file)
        t0 = time.time()
        for epoch in range(self.start_epoch, classifyConfig.maxEpochs):
            # self.optimizer = torchOptimizer.adjust_optimizer(epoch, lr)
            self.optimizer.zero_grad()
            for idx, (imgs, targets) in enumerate(dataloader):
                current_iter = epoch * total_images + idx
                lr = self.multiLR.get_lr(epoch, current_iter)
                self.multiLR.adjust_learning_rate(self.optimizer, lr)
                # Compute loss, compute gradient, update parameters
                output_list = self.model(imgs.to(self.device))
                loss = self.compute_loss(output_list, targets)
                loss.backward()

                # accumulate gradient for x batches before optimizing
                if ((idx + 1) % classifyConfig.accumulated_batches == 0) or (idx == len(dataloader) - 1):
                    self.optimizer.step()
                    self.optimizer.zero_grad()


                print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch,
                                                                                    idx,
                                                                                    len(dataloader),
                                                                                    '%.3f' % loss,
                                                                                    '%.7f' %
                                                                                    self.optimizer.param_groups[0]['lr'],
                                                                                    time.time() - t0))
                self.update_logger(current_iter, loss.data)

                t0 = time.time()

            self.test(val_path, epoch)


def main():
    print("process start...")
    options = ArgumentsParse.train_input_parse()
    classify_train = ClassifyTrain(options.cfg, 0)
    classify_train.train(options.trainPath, options.valPath)
    print("process end!")


if __name__ == "__main__":
    main()
