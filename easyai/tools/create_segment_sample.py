import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import random
import cv2
import numpy as np
from easyai.helper import DirProcess
from easyai.helper import ImageProcess


class CreateSegmentionSample():

    def __init__(self):
        self.dirProcess = DirProcess()
        self.image_process = ImageProcess()
        self.annotation_post = ".png"

    def create_train_and_test(self, inputDir, outputPath, probability):
        annotationsDir = os.path.join(inputDir, "../SegmentLabel")
        saveTrainFilePath = os.path.join(outputPath, "train.txt")
        saveTestFilePath = os.path.join(outputPath, "val.txt")
        saveTrainFilePath = open(saveTrainFilePath, "w")
        saveTestFilePath = open(saveTestFilePath, "w")

        imageList = list(self.dirProcess.getDirFiles(inputDir, "*.*"))
        random.shuffle(imageList)
        for imageIndex, imagePath in enumerate(imageList):
            print(imagePath)
            image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            path, file_name_and_post = os.path.split(imagePath)
            imageName, post = os.path.splitext(file_name_and_post)
            seg_label_name = imageName + self.annotation_post
            label_path = os.path.join(annotationsDir, seg_label_name)
            if (image is not None) and os.path.exists(label_path):
                if (imageIndex + 1) % probability == 0:
                    saveTestFilePath.write("%s\n" % file_name_and_post)
                else:
                    saveTrainFilePath.write("%s\n" % file_name_and_post)
        saveTrainFilePath.close()
        saveTestFilePath.close()


def test():
    print("start...")
    test = CreateSegmentionSample()
    print("End of game, have a nice day!")


if __name__ == "__main__":
   test()

