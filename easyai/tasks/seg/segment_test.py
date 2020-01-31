#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import time
from easyai.data_loader.seg.segment_dataloader import get_segment_val_dataloader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.evaluation.metrics import runningScore
from easyai.config import segment_config


class SegmentionTest():

    def __init__(self, cfg_path, gpu_id):
        self.torchModelProcess = TorchModelProcess()
        self.result_process = SegmentResultProcess()

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.running_metrics = runningScore(2)
        self.running_metrics.reset()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def test(self, val_path):
        dataloader = get_segment_val_dataloader(val_path, segment_config.imgSize, 1)
        print("Eval data num: {}".format(len(dataloader)))
        prev_time = time.time()
        for i, (images, segment_targets) in enumerate(dataloader):
            # Get detections
            with torch.no_grad():
                output_list = self.model(images.to(self.device))
                output = self.compute_output(output_list)
                prediction = self.result_process.get_detection_result(output)
                gt = segment_targets[0].data.cpu().numpy()

            print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
            prev_time = time.time()

            self.running_metrics.update(gt, prediction)

        score, class_iou = self.running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        self.running_metrics.reset()

        return score, class_iou

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = np.squeeze(prediction.data.cpu().numpy())
        return prediction

    def save_test_result(self, epoch, score, class_iou):
        # Write epoch results
        save_result_path = os.path.join(segment_config.root_save_dir, 'results.txt')
        with open(save_result_path, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU : \t']))
            for i, iou in enumerate(class_iou):
                file.write(segment_config.className[i] + ": {:.3f} ".format(iou))
            file.write("\n")
