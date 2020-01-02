import os.path
from easyAI.helper import DirProcess, XMLProcess
from easyAI.data_loader.dataLoader import *
from easyAI.data_loader.trainDataProcess import TrainDataProcess

class ImageDetectValDataLoader(DataLoader):

    def __init__(self, valPath, batch_size=1, img_size=(416, 416)):
        super().__init__(valPath)
        path, _ = os.path.split(valPath)
        self.dataPath = path
        self.xmlProcess = XMLProcess()
        self.trainDataProcess = TrainDataProcess()
        self.imageAndLabelList = self.getImageAndAnnotationList(valPath)
        self.nF = len(self.imageAndLabelList)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.imageSize = img_size
        self.color = (127.5, 127.5, 127.5)
        assert self.nF > 0, 'No images found in path %s' % valPath

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration
        img_path, _ = self.imageAndLabelList[self.count]

        # Read image
        img = cv2.imread(img_path)  # BGR
        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Padded resize
        rgbImage, _, _, _ = self.trainDataProcess.resize_square(rgbImage, self.imageSize, self.color)
        torchImage = self.convertTorchTensor(rgbImage)

        return img_path, torchImage

    def __len__(self):
        return self.nB  # number of batches