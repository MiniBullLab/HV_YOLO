import cv2
import numpy as np
from .colorDefine import ColorDefine

class ImageDraw():

    def __init__(self):
        pass

    def drawSegmentResult(self, srcImage, segmentResult, className):
        count = len(className)
        label_colours = dict(zip(range(19), ColorDefine.colors))
        img = cv2.cvtColor(np.asarray(srcImage), cv2.COLOR_RGB2BGR)  # convert PIL.image to cv2.mat

        r = segmentResult.copy()
        g = segmentResult.copy()
        b = segmentResult.copy()
        for l in range(0, count):
            r[segmentResult == l] = label_colours[l][0]
            g[segmentResult == l] = label_colours[l][1]
            b[segmentResult == l] = label_colours[l][2]

        rgb = np.zeros((segmentResult.shape[0], segmentResult.shape[1], 3))

        rgb[:, :, 0] = (r * 0.4 + img[:, :, 2] * 0.6) / 255.0
        rgb[:, :, 1] = (g * 0.4 + img[:, :, 1] * 0.6) / 255.0
        rgb[:, :, 2] = (b * 0.4 + img[:, :, 0] * 0.6) / 255.0

        return rgb