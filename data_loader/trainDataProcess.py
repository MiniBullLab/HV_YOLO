import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import cv2
import numpy as np

from helper.dataType import DetectionObject

class TrainDataProcess():

    def __init__(self):
        pass

    def resize_square(self, img, imageSize, color=(0, 0, 0)):  # resize a rectangular image to a padded square
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(imageSize[0]) / shape[1], float(imageSize[1]) / shape[0])  # ratio  = old / new
        new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
        dw = imageSize[0] - new_shape[1]  # width padding
        dh = imageSize[1] - new_shape[0]  # height padding
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)  # resized, no border
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw, dh

    def resizeDetectObjects(self, srcImage, imageSize, detections, className):
        result = []
        # The amount of padding that was added
        pad_x = 0 if (imageSize[0] / srcImage.shape[1]) < (imageSize[1] / srcImage.shape[0]) else imageSize[0] - imageSize[1] / srcImage.shape[0] * srcImage.shape[1]
        pad_y = 0 if (imageSize[0] / srcImage.shape[1]) > (imageSize[1] / srcImage.shape[0]) else \
            imageSize[1] - imageSize[0] / srcImage.shape[1] * srcImage.shape[0]

        # Image height and width after padding is removed
        unpad_h = imageSize[1] - pad_y
        unpad_w = imageSize[0] - pad_x
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * srcImage.shape[0]
            box_w = ((x2 - x1) / unpad_w) * srcImage.shape[1]
            y1 = (((y1 - pad_y // 2) / unpad_h) * srcImage.shape[0]).round().item()
            x1 = (((x1 - pad_x // 2) / unpad_w) * srcImage.shape[1]).round().item()
            x2 = (x1 + box_w).round().item()
            y2 = (y1 + box_h).round().item()
            x1, y1, x2, y2 = max(x1, 1.0), max(y1, 1.0), min(x2, srcImage.shape[1]-1.0), min(y2, srcImage.shape[0]-1.0)
            tempObject = DetectionObject()
            tempObject.min_corner.x = x1
            tempObject.min_corner.y = y1
            tempObject.max_corner.x = x2
            tempObject.max_corner.y = y2
            tempObject.classIndex = int(cls_pred)
            tempObject.objectConfidence = conf
            tempObject.classConfidence = cls_conf
            tempObject.name = className[tempObject.classIndex]
            result.append(tempObject)
        return result

    def encode_segmap(self, mask, volid_label, valid_label):
        classes = -np.ones([100, 100])
        valid = [x for j in valid_label for x in j]
        for i in range(0, len(valid_label)):
            classes[i, :len(valid_label[i])] = valid_label[i]
        for label in volid_label:
            mask[mask == label] = 250
        for validc in valid:
            mask[mask == validc] = np.uint8(np.where(classes == validc)[0])

        return mask