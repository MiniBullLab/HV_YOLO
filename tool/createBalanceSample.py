import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from helper import DirProcess, XMLProcess
import config

class CreateBalanceSample():

    def __init__(self):
        self.dirProcess = DirProcess()
        self.xmlProcess = XMLProcess()

    def create(self, inputTrainPath, outputPath):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        path, _ = os.path.split(inputTrainPath)
        annotationDir = os.path.join(path, "../Annotations")
        imagesDir = os.path.join(path, "../JPEGImages")
        writeFile = self.createWriteFile(outputPath)
        for fileNameAndPost in self.dirProcess.getFileData(inputTrainPath):
            fileName, post = os.path.splitext(fileNameAndPost)
            annotationFileName = fileName + ".xml"
            annotationPath = os.path.join(annotationDir, annotationFileName)
            imagePath = os.path.join(imagesDir, fileNameAndPost)
            print(imagePath, annotationPath)
            if os.path.exists(annotationPath) and \
               os.path.exists(imagePath):
                _, _, boxes = self.xmlProcess.parseRectData(annotationPath)
                allNames = [box.name for box in boxes if box.name in config.className]
                names = set(allNames)
                print(names)
                for className in names:
                    writeFile[className].write(fileNameAndPost + "\n")

    def createWriteFile(self, outputPath):
        result = {}
        for className in config.className:
            classImagePath = os.path.join(outputPath, className + ".txt")
            result[className] = open(classImagePath, "w")
        return result

def test():
    print("start...")
    test = CreateBalanceSample()
    test.create("/home/lpj/github/data/VOCdevkit/Berkeley/ImageSets/train.txt",
                "/home/lpj/github/data/VOCdevkit/Berkeley/ImageSets")
    print("End of game, have a nice day!")

if __name__ == "__main__":
   test()