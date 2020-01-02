import os
from easyAI.data_loader.dataLoader import *
from easyAI.data_loader.trainDataProcess import TrainDataProcess
from easyAI.helper import DirProcess
from easyAI.helper import ImageProcess


class ImagesLoader(DataLoader):
    '''
    load images
    '''
    def __init__(self, path, batch_size=1, img_size=(416, 416)):
        super().__init__(path)

        self.imageProcess = ImageProcess()
        self.trainDataProcess = TrainDataProcess()
        dirProcess = DirProcess()
        tempFiles = dirProcess.getDirFiles(path, "*.*")
        if tempFiles:
            self.files = list(tempFiles)
            self.nF = len(self.files)
        else:
            self.files = []
            self.nF = []
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.imageSize = img_size
        self.color = (127.5, 127.5, 127.5)
        assert self.nF > 0, 'No images found in path %s' % path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        srcImage, rgbImage = self.imageProcess.readRgbImage(img_path)

        # Padded resize
        rgbImage, _, _, _ = self.trainDataProcess.resize_square(rgbImage, self.imageSize, self.color)
        torchImage = self.convertTorchTensor(rgbImage)

        return srcImage, torchImage

    def __len__(self):
        return self.nB  # number of batches