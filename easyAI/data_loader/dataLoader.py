import os
import random
import math
import numpy as np
import cv2
import torch
from easyAI.helper import DirProcess


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

    def convertTorchTensor(self, rgbImages):
        result = None
        if rgbImages is None:
            result = None
        elif rgbImages.ndim == 3:
            image = rgbImages.transpose(2, 0, 1)
            image = np.ascontiguousarray(image, dtype=np.float32)
            image /= 255.0
            result = torch.from_numpy(image).unsqueeze(0)
        elif rgbImages.ndim == 4:
            img_all = rgbImages.transpose(0, 3, 1, 2)
            img_all = np.ascontiguousarray(img_all, dtype=np.float32)
            img_all /= 255.0
            result = torch.from_numpy(img_all)
        return result

    def random_affine(self, img, targets=None, Segment=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1),
                      shear=(-3, 3),
                      borderValue=(127.5, 127.5, 127.5)):

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
