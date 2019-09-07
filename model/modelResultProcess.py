import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from utility.nonMaximumSuppression import *

class ModelResultProcess():

    def __init__(self):
        pass

    def detectResult(self, model, output, confThresh):
        preds = []
        useNewConf = False
        min_wh = 2  # (pixels) minimum box width and height
        count = len(output)
        for i in range(0, count):
            predEach = model.lossList[i](output[i])
            preds.append(predEach)
        pred = torch.cat(preds, 1)
        pred = pred.squeeze(0)

        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        if useNewConf:
            pred[:, 4] *= class_conf # improves mAP from 0.549 to 0.551

        # Select only suitable predictions
        indexList = (pred[:, 4] > confThresh) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
        pred = pred[indexList]
        if len(pred) > 0:
            # Select predicted classes
            class_conf = class_conf[indexList]
            class_pred = class_pred[indexList].unsqueeze(1).float()
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            pred[:, :4] = self.xywh2xyxy(pred[:, :4])

            # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
            pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)
        return pred

    def xywh2xyxy(self, x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
        y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
        y[:, 0] = (x[:, 0] - x[:, 2] / 2)
        y[:, 1] = (x[:, 1] - x[:, 3] / 2)
        y[:, 2] = (x[:, 0] + x[:, 2] / 2)
        y[:, 3] = (x[:, 1] + x[:, 3] / 2)
        return y