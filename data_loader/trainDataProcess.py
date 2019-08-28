import cv2
import numpy as np

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