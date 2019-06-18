import glob
import math
import os
import random
from sys import platform

import cv2
from PIL import Image
import numpy as np
import torch

# from torch.utils.data import Dataset
from utils.utils import xyxy2xywh
# from utils import xyxy2xywh
import config.config as config

class load_video():
    def __init__(self, path, batch_size=1, img_size=[416, 416]):
        if not os.path.exists(path):
            raise Exception("Not exists path!", path)
        if not path.endswith(".MPG") and not path.endswith(".avi") and not path.endswith(".mp4"):
            raise Exception("Invalid path!", path)

        self.videoCapture = cv2.VideoCapture(path)
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        img_all = []
        frame_all = []

        for i in range(0, self.batch_size):
            success, frame = self.videoCapture.read()

            if success == False:
                break

            # Padded resize
            img, _, _, _ = resize_square(frame, width=self.width, height=self.height, color=(127.5, 127.5, 127.5))

            img_all.append(img)
            frame_all.append(frame)

        if len(img_all) > 0:
            img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
            img_all = np.ascontiguousarray(img_all, dtype=np.float32)
            # img_all -= self.rgb_mean
            # img_all /= self.rgb_std
            img_all /= 255.0

            return frame_all, torch.from_numpy(img_all)
        else:
            raise StopIteration

class load_images():  # for inference
    def __init__(self, path, batch_size=1, img_size=[416, 416]):
        if os.path.isdir(path):
            self.files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]

        assert self.nF > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img = cv2.imread(img_path)  # BGR

        # Padded resize
        img, _, _, _ = resize_square(img, width=self.width, height=self.height, color=(127.5, 127.5, 127.5))

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        # img -= self.rgb_mean
        # img /= self.rgb_std
        img /= 255.0

        return [img_path], img

    def __len__(self):
        return self.nB  # number of batches

class load_images_and_labels():  # for training
    def __init__(self, path, batch_size=1, img_size=[768, 320], multi_scale=False, augment=False, balancedSample=False):
        self.path = path

        with open(path, 'r') as file:
            self.img_files = file.readlines()

        self.img_files = [path.replace('\n', '') for path in self.img_files]

        if balancedSample:# MacOS (local)
            self.balancedFiles = {}
            self.nFs = {}
            for i in range(0, len(config.className)):
                with open("./data/classes/" + config.className[i] + ".txt", "r") as file:
                    self.balancedFiles[config.className[i]] = file.readlines()
                    self.balancedFiles[config.className[i]] = [path.replace('\n', '') for path in self.balancedFiles[config.className[i]]]
                    self.nFs[config.className[i]] = len(self.balancedFiles[config.className[i]])

        self.label_files = [path.replace('JPEGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in
                            self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]
        self.volid_label_det = [6, 7, 8]
        self.valid_label_det = [[0], [1], [2], [3], [4], [5], [9]]

        self.multi_scale = multi_scale
        self.augment = augment
        self.balancedSample = balancedSample

        assert self.nB > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((1, 3, 1, 1))

    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)#

        self.balancedCount = np.zeros(len(config.className))
        self.shuffled_vectors = {}
        if self.balancedSample:
            for i in range(0, len(config.className)):
                    self.shuffled_vectors[config.className[i]] = np.random.permutation(self.nFs[config.className[i]]) \
                        if self.augment else np.arange(self.nFs[config.className[i]])#
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        if self.balancedSample:
            randomLabel = np.random.randint(0, len(self.valid_label_det))
            print("loading labels {}".format(config.className[randomLabel]))
            balanced_file = self.balancedFiles[config.className[randomLabel]]
            ia = int(self.balancedCount[randomLabel])
            self.balancedCount[randomLabel] += self.batch_size
            ib = int(min(self.balancedCount[randomLabel], self.nFs[config.className[randomLabel]]))
            if self.balancedCount[randomLabel] > self.nFs[config.className[randomLabel]]:
                self.balancedCount[randomLabel] = 0
        else:
            ia = self.count * self.batch_size
            ib = min((self.count + 1) * self.batch_size, self.nF)

        # ia = self.count * self.batch_size
        # ib = min((self.count + 1) * self.batch_size, self.nF)

        if self.multi_scale:
            # Multi-Scale YOLO Training
            width = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
            scale = float(self.width) / float(self.height)
            height = int(round(float(width / scale) / 32.0) * 32)
        else:
            # Fixed-Scale YOLO Training
            height = self.height
            width = self.width

        img_all = []
        labels_all = []
        for index, files_index in enumerate(range(ia, ib)):
            if self.balancedSample:
                img_path = balanced_file[self.shuffled_vectors[config.className[randomLabel]][files_index]]
                label_path = img_path.replace("JPEGImages", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            else:
                img_path = self.img_files[self.shuffled_vector[files_index]]
                label_path = self.label_files[self.shuffled_vector[files_index]]

            img = cv2.imread(img_path)  # BGR
            if img is None:
                continue

            augment_hsv = True
            if self.augment and augment_hsv:
                # SV augmentation by 50%
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            h, w, _ = img.shape
            img, ratio, padw, padh = resize_square(img, width=width, height=height, color=(127.5, 127.5, 127.5))

            # Load labels
            if os.path.isfile(label_path):
                labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)

                # Normalized xywh to pixel xyxy format
                labels0 = encode_label(labels0, self.volid_label_det, self.valid_label_det)
                labels = labels0.copy()
                labels[:, 1] = ratio * w * (labels0[:, 1] - labels0[:, 3] / 2) + padw // 2
                labels[:, 2] = ratio * h * (labels0[:, 2] - labels0[:, 4] / 2) + padh // 2
                labels[:, 3] = ratio * w * (labels0[:, 1] + labels0[:, 3] / 2) + padw // 2
                labels[:, 4] = ratio * h * (labels0[:, 2] + labels0[:, 4] / 2) + padh // 2
            else:
                labels = np.array([])

            # Augment image and labels
            if self.augment:
                img, labels, M = random_affine(img, labels, degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.1))

            plotFlag = False
            if plotFlag:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10)) if index == 0 else None
                plt.subplot(4, 4, index + 1).imshow(img[:, :, ::-1])
                plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
                plt.axis('off')

            nL = len(labels)
            if nL > 0:
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy())
                labels[:, (1,3)] = labels[:, (1,3)]/width
                labels[:, (2,4)] = labels[:, (2,4)]/height

            if self.augment:
                # random left-right flip
                lr_flip = True
                if lr_flip & (random.random() > 0.5):
                    img = np.fliplr(img)
                    if nL > 0:
                        labels[:, 1] = 1 - labels[:, 1]

                # random up-down flip
                ud_flip = False
                if ud_flip & (random.random() > 0.5):
                    img = np.flipud(img)
                    if nL > 0:
                        labels[:, 2] = 1 - labels[:, 2]

            delete_index = []
            for i, label in enumerate(labels):
                if label[2] + label[4]/2 > 1:
                    yold = label[2] - label[4] / 2
                    label[2] = (yold + float(1)) / float(2)
                    label[4] = float(1) - yold
                if label[3] < 0.005 or label[4] < 0.005:
                    delete_index.append(i)

            labels = np.delete(labels, delete_index, axis=0)

            img_all.append(img)
            labels_all.append(torch.from_numpy(labels))

        # Normalize
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        # img_all -= self.rgb_mean
        # img_all /= self.rgb_std
        img_all /= 255.0

        return torch.from_numpy(img_all), labels_all

    def __len__(self):
        return self.nB  # number of batches

class load_images_and_segment():  # for training
    def __init__(self, path, batch_size=1, img_size=[768, 320], augment=False):
        self.path = path
        # self.img_files = sorted(glob.glob('%s/*.*' % path))
        with open(path, 'r') as file:
            self.img_files = file.readlines()

        if platform == 'darwin':  # MacOS (local)
            self.img_files = [path.replace('\n', '').replace('/images', './data/coco/images')
                              for path in self.img_files]
        else:  # linux (gcp cloud)
            self.img_files = [path.replace('\n', '').replace('/images', './data/coco/images') for path in self.img_files]

        self.seg_files = [path.replace('JPEGImages', 'SegmentLabel').replace('.png', '.png').replace('.jpg', '.png') for path in
                            self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]
        # self.volid_label_seg = [0, 1, 4, 5, 8, 9, 10, 11, 12, 14, 16, 18, 22, 25, 26, 28, 31, 32, 33, 34, 35, 36,
        #                         37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 53, 58, 59, 60, 62, 63, 64, 65]
        # self.valid_label_seg = [[54, 55, 56, 61], [19, 20, 21], [52, 57], [7, 13], [2, 3, 15, 29], [6, 17], [30],
        #                         [27], [23, 24], [48], [49, 50]]
        self.volid_label_seg = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_label_seg = [[7], [8], [11], [12], [13], [17], [19], [20], [21], [22],
                                [23], [24], [25], [26], [27], [28], [31], [32], [33]]

        self.augment = augment

        assert self.nB > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((1, 3, 1, 1))

    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

        multi_scale = False
        if multi_scale and self.augment:
            # Multi-Scale YOLO Training
            height = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
            width = int(float(self.width) / float(self.height) * height)
        else:
            # Fixed-Scale YOLO Training
            height = self.height
            width = self.width

        img_all = []
        seg_all = []
        for index, files_index in enumerate(range(ia, ib)):
            img_path = self.img_files[self.shuffled_vector[files_index]]
            seg_path = self.seg_files[self.shuffled_vector[files_index]]

            img = cv2.imread(img_path)  # BGR
            seg = Image.open(seg_path)
            if img is None:
                continue

            augment_hsv = True
            if self.augment and augment_hsv:
                # SV augmentation by 50%
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            h, w, _ = img.shape
            img, ratio, padw, padh = resize_square(img, width=width, height=height, color=(127.5, 127.5, 127.5))
            #----------seg------------------------------------------------------------------
            ratio = min(float(width) / w, float(height) / h)  # ratio  = old / new
            new_shape = [round(h * ratio), round(w * ratio)]
            seg = encode_segmap(np.array(seg, dtype=np.uint8), self.volid_label_seg, self.valid_label_seg)
            #seg = np.array(seg, dtype=np.uint8)
            #print(set(list(seg.flatten())))
            seg = cv2.resize(seg, (new_shape[1], new_shape[0]))
            seg = np.pad(seg, ((padh // 2, padh - (padh // 2)), (padw // 2, padw - (padw // 2))), 'constant', constant_values=250)
            #seg[(seg != 0) & (seg != 250)] = 1
            #seg = decode_segmap(seg)
            ################################################################################

            # Augment image and labels
            if self.augment:
                img, seg, M = random_affine(img, Segment=seg, degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.1))

            if self.augment:
                # random left-right flip
                lr_flip = True
                if lr_flip & (random.random() > 0.5):
                    img = np.fliplr(img)
                    seg = np.fliplr(seg)

            # seg
            valid_masks = np.zeros(seg.shape)
            for l in range(0, len(self.valid_label_seg)):
                valid_mask = seg == l
                valid_masks += valid_mask
            valid_masks[valid_masks==0]=-1
            seg = np.float32(seg) * valid_masks
            seg[seg < 0] = 250
            seg = np.uint8(seg)
            # print(set(list(seg.flatten())))
            # segdraw = decode_segmap(seg)
            # cv2.imshow("Seg", segdraw)
            # cv2.waitKey()

            img_all.append(img)
            seg_all.append(seg)

        # Normalize
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        # img_all -= self.rgb_mean
        # img_all /= self.rgb_std
        img_all /= 255.0

        #-------------seg
        seg_all = torch.from_numpy(np.array(seg_all)).long()

        return torch.from_numpy(img_all), seg_all

    def __len__(self):
        return self.nB  # number of batches

class load_images_and_labels_and_segment():  # for training
    def __init__(self, path, batch_size=1, img_size=[768, 320], augment=False):
        self.path = path
        # self.img_files = sorted(glob.glob('%s/*.*' % path))
        with open(path, 'r') as file:
            self.img_files = file.readlines()

        if platform == 'darwin':  # MacOS (local)
            self.img_files = [path.replace('\n', '').replace('/images', './data/coco/images')
                              for path in self.img_files]
        else:  # linux (gcp cloud)
            self.img_files = [path.replace('\n', '').replace('/images', './data/coco/images') for path in self.img_files]

        self.seg_files = [path.replace('JPEGImages', 'SegmentLabel').replace('.png', '_bin.png').replace('.jpg', '_bin.png') for path in
                            self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment

        assert self.nB > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((1, 3, 1, 1))

    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

        multi_scale = False
        if multi_scale and self.augment:
            # Multi-Scale YOLO Training
            height = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
            width = int(float(self.width) / float(self.height) * height)
        else:
            # Fixed-Scale YOLO Training
            height = self.height
            width = self.width

        img_all = []
        seg_all = []
        for index, files_index in enumerate(range(ia, ib)):
            img_path = self.img_files[self.shuffled_vector[files_index]]
            seg_path = self.seg_files[self.shuffled_vector[files_index]]

            img = cv2.imread(img_path)  # BGR
            seg = Image.open(seg_path)
            if img is None:
                continue

            augment_hsv = True
            if self.augment and augment_hsv:
                # SV augmentation by 50%
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            h, w, _ = img.shape
            img, ratio, padw, padh = resize_square(img, width=width, height=height, color=(127.5, 127.5, 127.5))
            #----------seg------------------------------------------------------------------
            ratio = min(float(width) / w, float(height) / h)  # ratio  = old / new
            new_shape = [round(h * ratio), round(w * ratio)]
            seg = encode_segmap(np.array(seg, dtype=np.uint8))
            #seg = np.array(seg, dtype=np.uint8)
            #print(set(list(seg.flatten())))
            seg = cv2.resize(seg, (new_shape[1], new_shape[0]))
            seg = np.pad(seg, ((padh // 2, padh - (padh // 2)), (padw // 2, padw - (padw // 2))), 'constant', constant_values=250)
            #seg[(seg != 0) & (seg != 250)] = 1
            #seg = decode_segmap(seg)
            ################################################################################

            # Augment image and labels
            if self.augment:
                img, seg, M = random_affine(img, Segment=seg, degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.1))

            if self.augment:
                # random left-right flip
                lr_flip = True
                if lr_flip & (random.random() > 0.5):
                    img = np.fliplr(img)
                    seg = np.fliplr(seg)

            seg[(seg != 0) & (seg != 1) & (seg != 2) & (seg != 3) & (seg != 4) & (seg != 5) & (seg != 6) & (seg != 7)] = 250
            #segdraw = decode_segmap(seg)
            #cv2.imshow("Seg", segdraw)
            #cv2.waitKey()

            img_all.append(img)
            seg_all.append(seg)

        # Normalize
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        # img_all -= self.rgb_mean
        # img_all /= self.rgb_std
        img_all /= 255.0

        #-------------seg
        seg_all = torch.from_numpy(np.array(seg_all)).long()

        return torch.from_numpy(img_all), seg_all

    def __len__(self):
        return self.nB  # number of batches

def resize_square(img, width=416, height=416, color=(0, 0, 0)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(width) / shape[1], float(height) / shape[0])  # ratio  = old / new
    new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
    dw = width - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)  # resized, no border
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw, dh

def random_affine(img, targets=None, Segment=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-3, 3),
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

def convert_tif2bmp(p='/Users/glennjocher/Downloads/DATA/xview/val_images_bmp'):
    import glob
    import cv2
    files = sorted(glob.glob('%s/*.tif' % p))
    for i, f in enumerate(files):
        print('%g/%g' % (i + 1, len(files)))

        img = cv2.imread(f)

        cv2.imwrite(f.replace('.tif', '.bmp'), img)
        os.system('rm -rf ' + f)

def encode_segmap(mask, volid_label, valid_label):
    classes = -np.ones([100, 100])
    valid =  [x for j in valid_label for x in j]
    for i in range(0, len(valid_label)):
        classes[i,:len(valid_label[i])] = valid_label[i]
    for label in volid_label:
        mask[mask == label] = 250
    for validc in valid:
        mask[mask == validc] = np.uint8(np.where(classes == validc)[0])

    return mask

def encode_label(labels, volid_label, valid_label):
    classes = -np.ones([100, 100])
    for i in range(0, len(valid_label)):
        classes[i,:len(valid_label[i])] = valid_label[i]
    for label in volid_label:
        labels = labels[labels[:,0] != label]
    for row in range(0, labels.shape[0]):
        labels[row, 0] = np.float32(np.where(classes == int(labels[row, 0]))[0])

    return labels

def decode_labels(img, labels):

    h, w, _ = img.shape

    x1 = w * (labels[1] - labels[3]/2)
    y1 = h * (labels[2] - labels[4]/2)
    x2 = w * (labels[1] + labels[3]/2)
    y2 = h * (labels[2] + labels[4]/2)

    return x1, y1, x2, y2

def decode_segmap(temp):

    colors = [#[  0,   0,   0],
              [128,  64, 128],
              [244,  35, 232],
              [ 70,  70,  70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170,  30],
              [220, 220,   0],
              [107, 142,  35],
              [152, 251, 152],
              [  0, 130, 180],
              [220,  20,  60],
              [255,   0,   0],
              [  0,   0, 142],
              [  0,   0,  70],
              [  0,  60, 100],
              [  0,  80, 100],
              [  0,   0, 230],
              [119,  11,  32]]

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

def test():
    # Get dataloader
    color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]
    dataloader = load_images_and_labels("/home/sugon/yolov3-det/data/imglist/train_20000.txt", batch_size=8, img_size=[640, 352], augment=True, balancedSample=False)
    for i, (imgs, labels) in enumerate(dataloader):
        for img, label in zip(imgs, labels):
            print("Image: {}".format(i))
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0)).copy()
            label = label.numpy()
            for l in label:
                xmin, ymin, xmax, ymax = decode_labels(img, l)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color[int(l[0])], 2)
            cv2.imshow('img', img)
            key = cv2.waitKey(100)
        if key == 27:
            break

if __name__ == "__main__":
    test()

'''
            targets = targets[0].numpy()
            flag = 0
            for target in targets:
                xmin, ymin, xmax, ymax = decode_labels(img, target)
                #if (xmax - xmin) < 0.005 or (ymax- ymin) < 0.005:
                #print("error##############################\n"+
                #      "###################################")
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                #    flag = 1
            #if flag==1:
            cv2.imshow('img', img)
            key = cv2.waitKey()

   # Get dataloader
    dataloader = load_images_and_segment("/home/wfw/data/VOCdevkit/LaneMask/trainLane.txt", batch_size=1, img_size=[640, 352], augment=False)
    for i, (imgs, segs) in enumerate(dataloader):
        for img, seg in zip(imgs, segs):
            print("Image: {}".format(i))
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0)).copy()
            seg = decode_segmap(seg.numpy())
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            cv2.imshow('seg', seg)
            key = cv2.waitKey()
            if key == 27:
                break
'''
