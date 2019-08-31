import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from utility.nonMaximumSuppression import *

class ModelResultProcess():

    def __init__(self):
        pass

    def detectResult(self, model, output, confThresh, nmsThresh):
        detections = []
        preds = []
        for i in range(0, 3):
            predEach = model.lossList[i](output[i])
            preds.append(predEach)
        pred = torch.cat(preds, 1)
        pred = pred[pred[:, :, 4] > confThresh]
        if len(pred) > 0:
            detections = non_max_suppression(pred.unsqueeze(0), confThresh,
                                             nmsThresh)  # select nms method (or, and, soft-nms)
        return detections