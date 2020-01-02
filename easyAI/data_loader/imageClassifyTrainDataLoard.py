from easyAI.data_loader.dataLoader import DataLoader
import math
import cv2
import torch
import numpy as np

class ImageClassifyTrainDataLoader(DataLoader):
    def __init__(self, path, batch_size=1, img_size=(416, 416)):
        super().__init__(path)
        with open(path, 'r') as file:
            self.img_files = file.readlines()

        self.img_files = [path.replace('\n', '') for path in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.imageSize = img_size
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

        ia = self.count * self.batch_size

        img_all = []
        labels_all = []

        for i in range(ia, ia + self.batch_size):
            # Read image
            trainFileIndex = i % self.nF
            img_path = self.img_files[trainFileIndex]
            imagePath = img_path.split(" ")[0]
            label = int(img_path.split(" ")[1])
            img = cv2.imread(imagePath)  # BGR
            img = cv2.resize(img, self.imageSize)

            # Padded resize
            rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_all.append(rgbImage)
            labels_all.append(label)
        trochLabels = torch.from_numpy(np.asarray(labels_all))
        numpyImages = np.stack(img_all)
        torchImages = self.convertTorchTensor(numpyImages)

        return torchImages, trochLabels

    def __len__(self):
        return self.nB