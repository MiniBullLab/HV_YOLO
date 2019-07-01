# -*- coding: utf-8 -*-
from optparse import OptionParser
import os
import os.path
import glob
import shutil
import xml.etree.ElementTree as ElementTree
import pickle
import random
import cv2
import numpy as np

classNames = ["car"]

MIN_WIDTH = 0
MIN_HEIGHT = 0

class Point2d(object):
    """
    Helper class to define Box
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return

class Box(object):
    """
    class to define Box
    """

    def __init__(self):
        self.name = ""
        self.min_corner = Point2d(0, 0)
        self.max_corner = Point2d(0, 0)
        return

    def copy(self):
        b = Box()
        b.min_corner = self.min_corner
        b.max_corner = self.max_corner
        return b

    def width(self):
        return self.max_corner.x - self.min_corner.x

    def height(self):
        return self.max_corner.y - self.min_corner.y

    def __str__(self):
        return '%s:%d %d %d %d' % (self.name, self.min_corner.x, self.min_corner.y, self.max_corner.x, self.max_corner.y)

def getDirFiles(dataDir, filePost):
    imagePathPattern = os.path.join(dataDir, filePost)
    for filePath in glob.iglob(imagePathPattern):
        yield filePath
    return

def parseXML(xmlPath):
    xmlTree = ElementTree.parse(xmlPath)
    root = xmlTree.getroot()
    folder_node = root.find("folder")
    folder = folder_node.text
    image_node = root.find("filename")
    image_name = image_node.text
    #print(image_name)
    imageSizeNode = root.find("size")
    widthNode = imageSizeNode.find("width")
    heightNode = imageSizeNode.find("height")
    imageSize = Point2d(int(widthNode.text), int(heightNode.text))

    boxes = []
    for object_node in root.findall('rectObject'):
        box_node = object_node.find("bndbox")
        name_node = object_node.find("name")
        xMinNode = box_node.find("xmin")
        yMinNode = box_node.find("ymin")
        xMaxNode = box_node.find("xmax")
        yMaxNode = box_node.find("ymax")
        xMin = int(xMinNode.text)
        yMin = int(yMinNode.text)
        xMax = int(xMaxNode.text)
        yMax = int(yMaxNode.text)
        box = Box()
        box.name = name_node.text
        box.min_corner.x = xMin
        box.min_corner.y = yMin
        box.max_corner.x = xMax
        box.max_corner.y = yMax
        if box.width() >= MIN_WIDTH and box.height() >= MIN_HEIGHT:
            boxes.append(box)
    return image_name, imageSize, boxes

def convertTOLOData(imageSize, box):
    dw = 1. / imageSize.x
    dh = 1. / imageSize.y
    x = (box.min_corner.x + box.max_corner.x) / 2.0 - 1
    y = (box.min_corner.y + box.max_corner.y) / 2.0 - 1
    w = box.width()
    h = box.height()
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def writeYOLOClass(classData, savePath):
    outFile = open(savePath, 'w')
    for className in classData:
        outFile.write("%s\n" % className)
    outFile.close()

def convertYOLOAnnotation(xmlPath, txtPath):
    outFile = open(txtPath, 'w')
    imageName, imageSize, boxes = parseXML(xmlPath)
    for box in boxes:
        if box.name in classNames:
            classID = classNames.index(box.name)
            x, y, w, h = convertTOLOData(imageSize, box)
            outFile.write("%d %f %f %f %f\n" % (classID, x, y, w, h))
    outFile.close()

def processYOLOTrainData(inputPath, outputPath, flag):
    annotationsDir = os.path.join(inputPath, "../Annotations")
    if not os.path.exists(annotationsDir):
        print("% is not exits" % annotationsDir)
        return

    labelsDir = os.path.join(inputPath, "../labels")
    if not os.path.exists(labelsDir):
        os.makedirs(labelsDir)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    saveClassPath = os.path.join(outputPath, "%s.names" % flag)
    saveFilePath = os.path.join(outputPath, "%s.txt" % flag)
    saveFile = open(saveFilePath, "w")

    imageList = list(getDirFiles(inputPath, "*.*"))
    random.shuffle(imageList)
    for imagePath in imageList:
        print(imagePath)
        image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        path, file_name_and_post = os.path.split(imagePath)
        imageName, post = os.path.splitext(file_name_and_post)
        xmlPath = os.path.join(annotationsDir, "%s.xml" % imageName)
        if (image is not None) and os.path.exists(xmlPath):
            txtPath = os.path.join(labelsDir, "%s.txt" % imageName)
            convertYOLOAnnotation(xmlPath, txtPath)
            saveFile.write("%s\n" % imagePath)
    saveFile.close()

    writeYOLOClass(classNames, saveClassPath)

def processYOLOTrainAndTestData(inputPath, outputPath, dataName, probability):
    annotationsDir = os.path.join(inputPath, "../Annotations")
    if not os.path.exists(annotationsDir):
        print("% is not exits" % annotationsDir)
        return

    labelsDir = os.path.join(inputPath, "../labels")
    if not os.path.exists(labelsDir):
        os.makedirs(labelsDir)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    saveClassPath = os.path.join(outputPath, "%s.names" % dataName)
    saveTrainFilePath = os.path.join(outputPath, "%s_train.txt" % dataName)
    saveTestFilePath = os.path.join(outputPath, "%s_test.txt" % dataName)
    saveTrainFilePath = open(saveTrainFilePath, "w")
    saveTestFilePath = open(saveTestFilePath, "w")

    imageList = list(getDirFiles(inputPath, "*.*"))
    random.shuffle(imageList)
    for imageIndex, imagePath in enumerate(imageList):
        print(imagePath)
        print(imagePath)
        image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        path, file_name_and_post = os.path.split(imagePath)
        imageName, post = os.path.splitext(file_name_and_post)
        xmlPath = os.path.join(annotationsDir, "%s.xml" % imageName)
        if (image is not None) and os.path.exists(xmlPath):
            txtPath = os.path.join(labelsDir, "%s.txt" % imageName)
            convertYOLOAnnotation(xmlPath, txtPath)
            if (imageIndex + 1) % probability == 0:
                saveTestFilePath.write("%s\n" % imagePath)
            else:
                saveTrainFilePath.write("%s\n" % imagePath)
    saveTrainFilePath.close()
    saveTestFilePath.close()

    writeYOLOClass(classNames, saveClassPath)

def processYOLOData(inputPath, outputPath, flag, probability):
    if  "train_test"in flag.strip():
        processYOLOTrainAndTestData(inputPath, outputPath, "YOLO", probability)
    elif  "train"in flag.strip():
        processYOLOTrainData(inputPath, outputPath, "train")
    elif  "test"in flag.strip():
        processYOLOTrainData(inputPath, outputPath, "test")

def main():
    print("start...")
    processYOLOData("C:/Users/lpj/Desktop/CarData/JPEGImages", "C:/Users/lpj/Desktop/CarData", "train_test", 10)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()
