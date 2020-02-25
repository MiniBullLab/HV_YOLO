import os
import numpy as np
from PIL import Image
import cv2

class ImageProcess():

    def __init__(self):
        pass

    def isImageFile(self, imagePath):
        if os.path.exists(imagePath):
            return any(imagePath.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
        else:
            return False

    def loadYUVImage(self, filepath):
        img = Image.open(filepath).convert('YCbCr')
        y, _, _ = img.split()
        return y

    def readRgbImage(self, imagePath):
        rgbImage = None
        srcImage = cv2.imread(imagePath)  # BGR
        if srcImage is not None:
            rgbImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
        return srcImage, rgbImage

    def read_gray_image(self, image_path):
        gray_image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        # cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return gray_image

    def opencvImageRead(self, imagePath):
        img = cv2.imread(imagePath)  # BGR
        return img