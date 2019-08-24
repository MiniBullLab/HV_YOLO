import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from helper import DirProcess, XMLProcess
from .dataLoader import *

class ImageDetectValDataLoader(DataLoader):

    def __init__(self, valPath, batch_size=1, img_size=[416, 416]):
        super().__init__(valPath)
        path, _ = os.path.split(valPath)
        self.dataPath = path
        self.xmlProcess = XMLProcess()
        self.imageAndLabelList = self.getImageAndAnnotationList(valPath)
        self.nF = len(self.imageAndLabelList)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]
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
        rgbImage, _, _, _ = self.resize_square(rgbImage, width=self.width, height=self.height,
                                          color=(127.5, 127.5, 127.5))
        rgbImage = rgbImage.transpose(2, 0, 1)
        rgbImage = np.ascontiguousarray(rgbImage, dtype=np.float32)
        # img -= self.rgb_mean
        # img /= self.rgb_std
        rgbImage /= 255.0

        return img_path, torch.from_numpy(rgbImage).unsqueeze(0)

    def __len__(self):
        return self.nB  # number of batches