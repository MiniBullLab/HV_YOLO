import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import glob
import random
import math
from PIL import Image
import numpy as np
import cv2
import torch
from helper import DirProcess

class DataLoader():

    def __init__(self, path):
        self.dataPath = path
        self.dirProcess = DirProcess()

    def setDataPath(self, path):
        self.dataPath = path

    def getDataPath(self):
        return self.dataPath

    def getImageAndAnnotationList(self, trainPath):
        result = []
        imagesDir = os.path.join(self.dataPath, "../JPEGImages")
        annotationDir = os.path.join(self.dataPath, "../Annotations")
        for fileNameAndPost in self.dirProcess.getFileData(trainPath):
            fileName, post = os.path.splitext(fileNameAndPost)
            annotationFileName = fileName + ".xml"
            fileName, post = os.path.splitext(fileNameAndPost)
            annotationPath = os.path.join(annotationDir, annotationFileName)
            imagePath = os.path.join(imagesDir, fileNameAndPost)
            #print(imagePath)
            if os.path.exists(annotationPath) and \
                    os.path.exists(imagePath):
                result.append((imagePath, annotationPath))
        return result

    def resize_square(self, img, width=416, height=416, color=(0, 0, 0)):  # resize a rectangular image to a padded square
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(width) / shape[1], float(height) / shape[0])  # ratio  = old / new
        new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
        dw = width - new_shape[1]  # width padding
        dh = height - new_shape[0]  # height padding
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)  # resized, no border
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw, dh

    def random_affine(self, img, targets=None, Segment=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1),
                      shear=(-3, 3),
                      borderValue=(127.5, 127.5, 127.5)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        border = 0  # width of added border (optional)
        height = img.shape[0] + border * 2
        width = img.shape[1] + border * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=borderValue)  # BGR order borderValue
        if Segment is not None:
            segw = cv2.warpPerspective(Segment, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                       borderValue=250)
            return imw, segw, M

        # Return warped points also
        if targets is not None:
            if len(targets) > 0:
                n = targets.shape[0]
                points = targets[:, 1:5].copy()
                area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

                # warp points
                xy = np.ones((n * 4, 3))
                xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = (xy @ M.T)[:, :2].reshape(n, 8)

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # apply angle-based reduction
                radians = a * math.pi / 180
                reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
                x = (xy[:, 2] + xy[:, 0]) / 2
                y = (xy[:, 3] + xy[:, 1]) / 2
                w = (xy[:, 2] - xy[:, 0]) * reduction
                h = (xy[:, 3] - xy[:, 1]) * reduction
                xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

                # reject warped points outside of image
                np.clip(xy, 0, width, out=xy)
                w = xy[:, 2] - xy[:, 0]
                h = xy[:, 3] - xy[:, 1]
                area = w * h
                ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

                targets = targets[i]
                targets[:, 1:5] = xy[i]

            return imw, targets, M
        else:
            return imw

    def decode_labels(self, img, labels):

        h, w, _ = img.shape

        x1 = w * (labels[1] - labels[3] / 2)
        y1 = h * (labels[2] - labels[4] / 2)
        x2 = w * (labels[1] + labels[3] / 2)
        y2 = h * (labels[2] + labels[4] / 2)

        return x1, y1, x2, y2

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

    def decode_segmap(self, temp):

        colors = [  # [  0,   0,   0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]]

        label_colours = dict(zip(range(19), colors))

        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, 19):
            r[temp == l] = label_colours[l][0]
            g[temp == l] = label_colours[l][1]
            b[temp == l] = label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))

        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return rgb