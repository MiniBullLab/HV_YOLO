import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from helper import XMLProcess
from .dataLoader import *
import config.config as config
from utils.utils import xyxy2xywh

class ImageDetectTrainDataLoader(DataLoader):
    def __init__(self, trainPath, batchSize=1, imageSize=[768, 320], \
                 multi_scale=False, augment=False, balancedSample=False):
        super().__init__(trainPath)
        path, _ = os.path.split(trainPath)
        self.dataPath = path
        self.xmlProcess = XMLProcess()
        self.multi_scale = multi_scale
        self.augment = augment
        self.balancedSample = balancedSample
        self.balancedFiles = {}
        self.balanceFileCount = {}
        self.imageAndLabelList = []

        self.imageAndLabelList = self.getImageAndAnnotationList(trainPath)
        if self.balancedSample:
            self.balancedFiles, self.balanceFileCount = self.getBlanceFileList()

        self.nF = len(self.imageAndLabelList)  # number of image files
        self.nB = math.ceil(self.nF / batchSize)  # number of batches
        self.batch_size = batchSize
        self.width = imageSize[0]
        self.height = imageSize[1]

        assert self.nB > 0, 'No images found in path %s' % trainPath

    def __iter__(self):
        self.count = -1
        self.shuffledImages()
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        if self.balancedSample:
            randomLabel = np.random.randint(0, len(config.className))
            print("loading labels {}".format(config.className[randomLabel]))
            balanced_file = self.balancedFiles[config.className[randomLabel]]
            ia = int(self.balancedCount[randomLabel])
            self.balancedCount[randomLabel] = (self.balancedCount[randomLabel] + self.batch_size) % \
                                              self.balanceFileCount[config.className[randomLabel]]
        else:
            ia = self.count * self.batch_size

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
        for index, files_index in enumerate(range(ia, ia + self.batch_size)):
            if self.balancedSample:
                trainFileIndex = files_index % self.balanceFileCount[config.className[randomLabel]]
                img_path, label_path = balanced_file[self.shuffled_vectors[config.className[randomLabel]][trainFileIndex]]
            else:
                trainFileIndex = files_index % self.nF
                img_path, label_path = self.imageAndLabelList[self.shuffled_vector[trainFileIndex]]

            img = cv2.imread(img_path)  #BGR
            if img is None:
                continue

            rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.augmentImageHSV(rgbImage)

            h, w, _ = rgbImage.shape
            rgbImage, ratio, padw, padh = self.resize_square(rgbImage, width=width, height=height,\
                                                             color=(127.5, 127.5, 127.5))
            _, _, boxes = self.xmlProcess.parseRectData(label_path)
            labels = self.getResizeLabels(boxes, ratio, (padw, padh))

            # Augment image and labels
            if self.augment:
                rgbImage, labels, M = self.random_affine(rgbImage, labels, degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.1))

            nL = len(labels)
            if nL > 0:
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy())
                labels[:, (1,3)] = labels[:, (1,3)] / width
                labels[:, (2,4)] = labels[:, (2,4)] / height

            self.augmentImageFlip(rgbImage, labels)

            delete_index = []
            for i, label in enumerate(labels):
                if label[2] + label[4]/2 > 1:
                    yold = label[2] - label[4] / 2
                    label[2] = (yold + float(1)) / float(2)
                    label[4] = float(1) - yold
                if label[3] < 0.005 or label[4] < 0.005:
                    delete_index.append(i)

            labels = np.delete(labels, delete_index, axis=0)

            img_all.append(rgbImage)
            labels_all.append(torch.from_numpy(labels))

        # Normalize
        img_all = np.stack(img_all).transpose(0, 3, 1, 2)
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        # img_all -= self.rgb_mean
        # img_all /= self.rgb_std
        img_all /= 255.0

        return torch.from_numpy(img_all), labels_all

    def getBlanceFileList(self):
        blancedFileList = {}
        blancedFileCount = {}
        for i in range(0, len(config.className)):
            classFile = config.className[i] + ".txt"
            calssFilePath = os.path.join(self.dataPath, classFile)
            blancedFileList[config.className[i]] = self.getImageAndAnnotationList(calssFilePath)
            blancedFileCount[config.className[i]] = len(blancedFileList[config.className[i]])
        return blancedFileList, blancedFileCount

    def getResizeLabels(self, boxes, ratio, pad):
        result = np.array([])
        allNames = [box.name for box in boxes if box.name in config.className]
        if len(allNames) > 0:
            result = np.zeros((len(allNames), 5), dtype=np.float32)
            index = 0
            for box in boxes:
                if box.name in config.className:
                    classId = config.className.index(box.name)
                    minX = ratio * box.min_corner.x + pad[0] // 2
                    minY = ratio * box.min_corner.y + pad[1] // 2
                    maxX = ratio * box.max_corner.x + pad[0] // 2
                    maxY = ratio * box.max_corner.y + pad[1] // 2
                    result[index, :] = np.array([classId, minX, minY, maxX, maxY])
        return result

    def shuffledImages(self):
        if self.balancedSample:
            self.balancedCount = np.zeros(len(config.className))
            self.shuffled_vectors = {}
            for i in range(0, len(config.className)):
                self.shuffled_vectors[config.className[i]] = \
                    np.random.permutation(self.balanceFileCount[config.className[i]])
        else:
            self.shuffled_vector = np.random.permutation(self.nF)

    def augmentImageHSV(self, inputRGBImage):
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(inputRGBImage, cv2.COLOR_RGB2HSV)
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
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=inputRGBImage)

    def augmentImageFlip(self, inputRGBImage, labels):
        nL = len(labels)
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                inputRGBImage = np.fliplr(inputRGBImage)
                if nL > 0:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip & (random.random() > 0.5):
                inputRGBImage = np.flipud(inputRGBImage)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

    def __len__(self):
        return self.nB  # number of batches