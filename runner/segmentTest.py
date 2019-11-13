import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import time
from data_loader.imageSegmentValDataLoader import ImageSegmentValDataLoader
from utility.utils import *
from torch_utility.torch_model_process import TorchModelProcess
from evaluation.metrics import runningScore
from config import segmentConfig
from helper.arguments_parse import ArgumentsParse


class SegmentTest():

    def __init__(self, cfg_path, gpu_id):
        self.running_metrics = runningScore(2)

        self.torchModelProcess = TorchModelProcess()
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def test(self, val_path):
        dataloader = ImageSegmentValDataLoader(val_path,
                                               batch_size=segmentConfig.test_batch_size,
                                               img_size=segmentConfig.imgSize)
        print("Eval data num: {}".format(len(dataloader)))
        prev_time = time.time()
        for i, (img, segMap_val) in enumerate(dataloader):
            # Get detections
            with torch.no_grad():
                output_list = self.model(img.to(self.device))
                if len(output_list) == 1:
                    output = self.model.lossList[0](output_list[0])
                    # ------------seg---------------------
                    pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
                    gt = segMap_val[0].data.cpu().numpy()
                    self.running_metrics.update(gt, pred)

            print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
            prev_time = time.time()

        score, class_iou = self.running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        self.running_metrics.reset()

        return score, class_iou

    def save_test_result(self, epoch, score, class_iou):
        # Write epoch results
        with open('results.txt', 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU : \t']))
            for i, iou in enumerate(class_iou):
                file.write(segmentConfig.className[i] + ": {:.3f} ".format(iou))
            file.write("\n")


def main():
    print("process start...")
    options = ArgumentsParse.test_input_parse()
    test = SegmentTest(options.cfg, 0)
    test.load_weights(options.weights)
    test.test(options.valPath)
    print("process end!")


if __name__ == '__main__':
    main()
