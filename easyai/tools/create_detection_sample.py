import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import random
import cv2
import numpy as np
from easyai.helper import DirProcess, XMLProcess
from easyai.config.task import detect2d_config


class CreateDetectionSample():

    def __init__(self):
        self.dirProcess = DirProcess()
        self.xmlProcess = XMLProcess()
        self.annotation_post = ".xml"

    def createBalanceSample(self, inputTrainPath, outputPath):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        path, _ = os.path.split(inputTrainPath)
        annotationDir = os.path.join(path, "../Annotations")
        imagesDir = os.path.join(path, "../JPEGImages")
        writeFile = self.createWriteFile(outputPath)
        for fileNameAndPost in self.dirProcess.getFileData(inputTrainPath):
            fileName, post = os.path.splitext(fileNameAndPost)
            annotationFileName = fileName + self.annotation_post
            annotationPath = os.path.join(annotationDir, annotationFileName)
            imagePath = os.path.join(imagesDir, fileNameAndPost)
            print(imagePath, annotationPath)
            if os.path.exists(annotationPath) and \
               os.path.exists(imagePath):
                _, _, boxes = self.xmlProcess.parseRectData(annotationPath)
                allNames = [box.name for box in boxes if box.name in detect2d_config.className]
                names = set(allNames)
                print(names)
                for className in names:
                    writeFile[className].write(fileNameAndPost + "\n")

    def createTrainAndTest(self, inputDir, outputPath, probability):

        annotationsDir = os.path.join(inputDir, "../Annotations")
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
            xmlPath = os.path.join(annotationsDir, "%s%s" % (imageName, self.annotation_post))
            if (image is not None) and os.path.exists(xmlPath):
                if (imageIndex + 1) % probability == 0:
                    saveTestFilePath.write("%s\n" % file_name_and_post)
                else:
                    saveTrainFilePath.write("%s\n" % file_name_and_post)
        saveTrainFilePath.close()
        saveTestFilePath.close()

    def createWriteFile(self, outputPath):
        result = {}
        for className in detect2d_config.className:
            classImagePath = os.path.join(outputPath, className + ".txt")
            result[className] = open(classImagePath, "w")
        return result


def test():
    print("start...")
    test = CreateDetectionSample()
    test.createBalanceSample("/home/lpj/github/data/Berkeley/ImageSets/train.txt",
                "/home/lpj/github/data/Berkeley/ImageSets")
    print("End of game, have a nice day!")


if __name__ == "__main__":
   test()
