import os
import sys
import time
from easyAI.data_loader.imagesLoader import ImagesLoader
from easyAI.data_loader.videoLoader import VideoLoader
from easyAI.data_loader.trainDataProcess import TrainDataProcess
from easyAI.utility.utils import *
from easyAI.model.modelResultProcess import ModelResultProcess
from easyAI.torch_utility.torchModelProcess import TorchModelProcess
from easyAI.drawing.imageDraw import ImageDraw
from easyAI.base_algorithm.nms import non_max_suppression
from easyAI.config import detectConfig
from easyAI.helper.arguments_parse import ArgumentsParse


class Detection():

    def __init__(self, cfg_path, gpu_id):
        self.imageDraw = ImageDraw()
        self.trainDataProcess = TrainDataProcess()
        self.torchModelProcess = TorchModelProcess()
        self.modelResultProcess = ModelResultProcess()

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def detect(self, input_path):
        if os.path.isdir(input_path):
            dataloader = ImagesLoader(input_path, batch_size=1, img_size=detectConfig.imgSize)
        else:
            dataloader = VideoLoader(input_path, batch_size=1, img_size=detectConfig.imgSize)

        for i, (oriImg, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            prev_time = time.time()
            # Get detections
            with torch.no_grad():
                output = self.model(img.to(self.device))
                pred = self.modelResultProcess.detectResult(self.model, output, detectConfig.confThresh)
                # print(pred)
                detections = non_max_suppression(pred, detectConfig.confThresh, detectConfig.nmsThresh)

            print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))

            detectObjects = self.trainDataProcess.resizeDetectObjects(oriImg, detectConfig.imgSize,
                                                                      detections, detectConfig.className)
            self.imageDraw.drawDetectObjects(oriImg, detectObjects)

            cv2.namedWindow("image", 0)
            cv2.resizeWindow("image", int(oriImg.shape[1] * 0.8), int(oriImg.shape[0] * 0.8))
            cv2.imshow("image", oriImg)
            if cv2.waitKey() & 0xff == ord('q'):  # 按q退出
                break


def main():
    print("process start...")
    options = ArgumentsParse.parse_arguments()
    detect = Detection(options.cfg, 0)
    detect.load_weights(options.weights)
    detect.detect(options.inputPath)
    print("process end!")


if __name__ == '__main__':
    main()
