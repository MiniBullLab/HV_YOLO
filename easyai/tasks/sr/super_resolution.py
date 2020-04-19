#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import time
import torch
import cv2
import numpy as np
from easyai.data_loader.utility.images_loader import ImagesLoader
from easyai.data_loader.utility.video_loader import VideoLoader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.helper.arguments_parse import ArgumentsParse
from easyai.model.sr.MSRResNet import MSRResNet
from easyai.config.task import super_resolution_config
from PIL import Image
from torchvision.transforms import ToTensor


class SuperResolution():

    def __init__(self):
        self.torchModelProcess = TorchModelProcess()
        self.device = self.torchModelProcess.getDevice()
        self.model = MSRResNet(super_resolution_config.in_nc, upscale_factor=super_resolution_config.upscale_factor).to(self.device)

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def super_resolution(self, input_path):
        if os.path.isdir(input_path):
            dataloader = ImagesLoader(input_path)
        else:
            dataloader = VideoLoader(input_path)

        prev_time = time.time()
        for i, (oriImg, imgs) in enumerate(dataloader):
            img_pil = Image.fromarray(cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB))
            img = img_pil.convert('YCbCr')
            y, cb, cr = img.split()
            img_to_tensor = ToTensor()
            input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
            # Get detections
            with torch.no_grad():
                output = self.model(input.to(self.device))[0]

            print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
            prev_time = time.time()

            out_img_y = output.cpu().detach().numpy()
            out_img_y *= 255.0
            out_img_y = out_img_y.clip(0, 255)
            out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

            out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
            out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
            out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
            show_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)

            cv2.namedWindow("image", 0)
            cv2.resizeWindow("image", int(show_img.shape[1] * 0.5), int(show_img.shape[0] * 0.5))
            cv2.imshow('image', show_img)

            if cv2.waitKey() & 0xFF == 27:
                break


def main():
    print("process start...")
    options = ArgumentsParse.parse_arguments()
    segment = SuperResolution()
    segment.load_weights(options.weights)
    segment.super_resolution(options.inputPath)
    print("process end!")


if __name__ == '__main__':
    main()
