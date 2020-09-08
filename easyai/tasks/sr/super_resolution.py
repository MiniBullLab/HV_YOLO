#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import numpy as np
from easyai.tasks.utility.base_inference import BaseInference
from easyai.base_name.task_name import TaskName


class SuperResolution(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(cfg_path, config_path, TaskName.Segment_Task)

        self.model_args['upscale_factor'] = self.task_config.upscale_factor
        self.model = self.torchModelProcess.initModel(self.model_args, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def process(self, input_path, is_show=False):
        pass
        # for i, (oriImg, imgs) in enumerate(dataloader):
        #     img_pil = Image.fromarray(cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB))
        #     img = img_pil.convert('YCbCr')
        #     y, cb, cr = img.split()
        #     img_to_tensor = ToTensor()
        #     input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
        #     # Get detections
        #     with torch.no_grad():
        #         output = self.model(input.to(self.device))[0]
        #
        #     print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        #     prev_time = time.time()
        #
        #     out_img_y = output.cpu().detach().numpy()
        #     out_img_y *= 255.0
        #     out_img_y = out_img_y.clip(0, 255)
        #     out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        #
        #     out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        #     out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        #     out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        #     show_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)
        #
        #     cv2.namedWindow("image", 0)
        #     cv2.resizeWindow("image", int(show_img.shape[1] * 0.5), int(show_img.shape[0] * 0.5))
        #     cv2.imshow('image', show_img)
        #
        #     if cv2.waitKey() & 0xFF == 27:
        #         break

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            prediction = self.compute_output(output_list[:])
        return prediction, output_list

    def postprocess(self, result):
        pass

    def compute_output(self, output_list):
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        prediction = None
        if loss_count == 1 and output_count == 1:
            temp_output = self.model.lossList[0](output_list[0])
            prediction = temp_output.cpu().detach().numpy()
        elif loss_count > 1 and loss_count == output_count:
            preds = []
            for i in range(0, loss_count):
                temp = self.model.lossList[i](output_list[i])
                preds.append(temp)
            prediction = torch.cat(preds, 1)
            prediction = np.squeeze(prediction.cpu().detach().numpy())
        else:
            print("sr compute output error!")
        return prediction


if __name__ == '__main__':
    pass
