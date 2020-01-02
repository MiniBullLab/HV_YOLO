import os
import sys
import time
from easyAI.data_loader.imageClassifyTrainDataLoard import ImageClassifyTrainDataLoader
from easyAI.helper.arguments_parse import ArgumentsParse
from easyAI.torch_utility.torchModelProcess import TorchModelProcess
from easyAI.config import classifyConfig
from easyAI.utility.logger import AverageMeter


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ClassifyTest():

    def __init__(self, cfg_path, gpu_id):
        self.torchModelProcess = TorchModelProcess()
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def test(self, val_path):
        dataloader = ImageClassifyTrainDataLoader(val_path,
                                                  batch_size=classifyConfig.test_batch_size,
                                                  img_size=(32, 32))

        prev_time = time.time()
        for idx, (imgs, labels) in enumerate(dataloader):
            output_list = self.model(imgs.to(self.device))
            if len(output_list) == 1:
                output = self.model.lossList[0](output_list[0])
                prec1, prec5 = accuracy(output.data, labels.to(self.device), topk=(1, 5))
            self.top1.update(prec1, imgs.size(0))
            self.top5.update(prec5, imgs.size(0))

        print('prec1: {} \t prec5: {}\t'.format(self.top1.avg, self.top5.avg))
        return self.top1.avg, self.top5.avg

    def save_test_result(self, epoch, prec1, prec5):
        # Write epoch results
        with open('results.txt', 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | prec1: {:.3f} | prec5: {:.3f}\n".format(epoch, prec1, prec5))


def main():
    print("process start...")
    options = ArgumentsParse.test_input_parse()
    test = ClassifyTest(options.cfg, 0)
    test.load_weights(options.weights)
    test.test(options.valPath)
    print("process end!")


if __name__ == '__main__':
    main()


