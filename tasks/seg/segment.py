import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import time
from utility.utils import *
import scipy.misc as misc
from data_loader.utility.images_loader import ImagesLoader
from data_loader.utility.video_loader import VideoLoader
from torch_utility.torch_model_process import TorchModelProcess
from drawing.imageDraw import ImageDraw
from helper.arguments_parse import ArgumentsParse
from config import segmentConfig


class Segmentation():

    def __init__(self, cfg_path, gpu_id):
        self.torchModelProcess = TorchModelProcess()
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def segment(self, input_path):
        if os.path.isdir(input_path):
            dataloader = ImagesLoader(input_path, segmentConfig.imgSize)
        else:
            dataloader = VideoLoader(input_path, segmentConfig.imgSize)

        prev_time = time.time()
        for i, (oriImg, img) in enumerate(dataloader):
            # Get detections
            with torch.no_grad():
                output_list = self.model(img.to(self.device))
                if len(output_list) == 1:
                    output = self.model.lossList[0](output_list[0])

            print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
            prev_time = time.time()

            # oriImage shape, input shape
            ori_w, ori_h = oriImg.shape[1], oriImg.shape[0]
            pre_h, pre_w = segmentConfig.imgSize[1], segmentConfig.imgSize[0]

            # segmentation
            segmentations = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
            ratio = min(float(pre_w) / ori_w, float(pre_h) / ori_h)  # ratio  = old / new
            new_shape = [round(ori_h * ratio), round(ori_w * ratio)]
            dw = pre_w - new_shape[1]  # width padding
            dh = pre_h - new_shape[0]  # height padding
            padH = [dh // 2, dh - (dh // 2)]
            padW = [dw // 2, dw - (dw // 2)]
            segmentations = segmentations[padH[0]:pre_h - padH[1], padW[0]:pre_w - padW[1]]
            segmentations = segmentations.astype(np.float32)
            segmentations = misc.imresize(segmentations, [ori_h, ori_w], 'nearest',
                                          mode='F')  # float32 with F mode, resize back to orig_size
            drawSeg = ImageDraw()
            decoded = drawSeg.drawSegmentResult(oriImg, segmentations, segmentConfig.className)

            cv2.namedWindow("image", 0)
            cv2.resizeWindow("image", int(decoded.shape[1] * 0.5), int(decoded.shape[0] * 0.5))
            cv2.imshow('image', decoded)

            if cv2.waitKey() & 0xFF == 27:
                break


def main():
    print("process start...")
    options = ArgumentsParse.parse_arguments()
    segment = Segmentation(options.cfg, 0)
    segment.load_weights(options.weights)
    segment.segment(options.inputPath)
    print("process end!")


if __name__ == '__main__':
    main()
