# -*- coding: utf-8 -*-
import cv2
from optparse import OptionParser
import os
import os.path
import glob
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import random
import codecs
#python2
#reload(sys)
#sys.setdefaultencoding('utf-8')

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def getDirFiles(dataDir, filePost):
    imagePathPattern = os.path.join(dataDir, filePost)
    for filePath in glob.iglob(imagePathPattern):
        yield filePath
    return

def getDataClass(dataPath):
    result = []
    dirNames = os.listdir(dataPath)
    for name in dirNames:
        if not name.startswith("."):
            filePath = os.path.join(dataPath, name)
            if os.path.isdir(filePath):
                result.append(name)
    return result

def plotPoint(xDatas, yDatas, maxX, maxY):
    plt.scatter(xDatas, yDatas, cmap=plt.cm.Blues, edgecolors='none', s=2)

    # 设置图表标题并给坐标轴加上标签
    plt.title('Image Size', fontsize=24)
    plt.xlabel('Width', fontsize=14)
    plt.ylabel('Height', fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', which='major', labelsize=14)
    # 设置每个坐标轴的取值范围
    plt.axis([0, maxX, 0, maxY])
    plt.savefig("point.png")
    plt.show()

def plotLine(xDatas, yDatas, xLabels):
    plt.figure(figsize=(20, 5))
    plt.plot(xDatas, yDatas, linewidth=2, color='r', marker='o', label="class")
    #plt.plot(x2, y2, '', label="2016年")
    plt.title('图')
    plt.legend(loc='upper right')
    plt.xticks(xDatas, xLabels)
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.grid(xDatas)
    plt.savefig("line.png")
    plt.show()

def showAllImageClass(inputPath, imagePost):
    classLabels = []
    datasX = []
    datasY = []
    dataClass = getDataClass(inputPath)
    if dataClass:
        for index, className in enumerate(dataClass):
            dataClassPath = os.path.join(inputPath, className)
            imageCount = len(list(getDirFiles(dataClassPath, imagePost)))
            classLabels.append(className)
            datasX.append(index)
            datasY.append(imageCount)
    plotLine(datasX, datasY, classLabels)

def showAllImageSize(inputPath, imagePost):
    datasWidth = []
    datasHeight = []
    dataClass = getDataClass(inputPath)
    if dataClass:
        for _, className in enumerate(dataClass):
            dataClassPath = os.path.join(inputPath, className)
            for imagePath in getDirFiles(dataClassPath, imagePost):
                print(imagePath)
                image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    imageHeight, imageWidth = image.shape
                    datasWidth.append(imageWidth)
                    datasHeight.append(imageHeight)
    else:
        for imagePath in getDirFiles(inputPath, "*.*"):
            print(imagePath)
            image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                imageHeight, imageWidth = image.shape
                datasWidth.append(imageWidth)
                datasHeight.append(imageHeight)
    plotPoint(datasWidth, datasHeight, 1000, 1000)

def readCaffeTestFileData(testFilePath):
    result = []
    with open(testFilePath, "r") as testFile:
        for line in testFile:
            data = line.strip()
            if data:
                splitData = [x for x in data.split() if x.strip()]
                if len(splitData) == 2:
                    result.append((splitData[0], splitData[1]))
    return result

def writeCaffeImageData(imagePath, saveImageDir, classIndex, saveFile, flag):
    path, fileNameAndPost = os.path.split(imagePath)
    fileName, post = os.path.splitext(fileNameAndPost)
    imageName = "%d_%s" % (classIndex, fileNameAndPost)
    image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), 1)
    if image is not None:
        saveImagePath = os.path.join(saveImageDir, imageName)
        writeContent = ""
        if "train" in flag:
            writeContent = "%d/%s %d\n" % (classIndex, imageName, classIndex)
        elif "val" in flag:
            writeContent = "%s %d\n" % (imageName, classIndex)
        cv2.imencode('.jpg', image)[1].tofile(saveImagePath)
        saveFile.write(writeContent)

def writeCaffeDataClass(dataClass, outputPath):
    classDefine = {}
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for index, className in enumerate(dataClass):
        classDefine[index] = className
    saveClassDefinePath = os.path.join(outputPath, "class.json")
    with codecs.open(saveClassDefinePath, 'w', encoding='utf-8') as f:
        json.dump(classDefine, f, sort_keys=True, indent=4, ensure_ascii=False)

def processCaffeTrainData(inputPath, outputPath, flag):

    dataClass = getDataClass(inputPath)

    saveImageDir = os.path.join(outputPath, flag)
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    else:
        print("%s is exits" % saveImageDir)
        return
    saveFilePath = os.path.join(outputPath, "%s.txt" % flag)
    saveFile = open(saveFilePath, "w")

    for index, className in enumerate(dataClass):
        dataClassDir = os.path.join(inputPath, className)
        saveClassDir = os.path.join(saveImageDir, "%d" % index)
        if "train" in flag:
            if not os.path.exists(saveClassDir):
                os.makedirs(saveClassDir)
        imageList = list(getDirFiles(dataClassDir, "*.*"))
        random.shuffle(imageList)
        for imagePath in imageList:
            print(imagePath)
            if "train"in flag:
                writeCaffeImageData(imagePath, saveClassDir, index, saveFile, flag)
            elif "val" in flag:
                writeCaffeImageData(imagePath, saveImageDir, index, saveFile, flag)

    saveFile.close()
    writeCaffeDataClass(dataClass, outputPath)

def processCaffeTrainAndValData(inputPath, outputPath, probability):
    dataClass = getDataClass(inputPath)

    saveTrainImageDir = os.path.join(outputPath, "train")
    saveValImageDir = os.path.join(outputPath, "val")
    if (not os.path.exists(saveTrainImageDir)) and \
        (not os.path.exists(saveValImageDir)):
        os.makedirs(saveTrainImageDir)
        os.makedirs(saveValImageDir)
    else:
        print("%s or %s is exits" % (saveTrainImageDir, saveValImageDir))
        return
    saveTrainFilePath = os.path.join(outputPath, "train.txt")
    saveValFliePath = os.path.join(outputPath, "val.txt")
    saveTrainFile = open(saveTrainFilePath, "w")
    saveValFile = open(saveValFliePath, "w")

    for classIndex, className in enumerate(dataClass):
        dataClassDir = os.path.join(inputPath, className)
        saveTrainClassDir = os.path.join(saveTrainImageDir, "%d" % classIndex)
        if not os.path.exists(saveTrainClassDir):
            os.makedirs(saveTrainClassDir)
        imageList = list(getDirFiles(dataClassDir, "*.*"))
        random.shuffle(imageList)
        for imageIndex, imagePath in enumerate(imageList):
            print(imagePath)
            if (imageIndex + 1) % probability == 0:
                writeCaffeImageData(imagePath, saveValImageDir, classIndex, saveValFile, "val")
            else:
                writeCaffeImageData(imagePath, saveTrainClassDir, classIndex, saveTrainFile, "train")

    saveTrainFile.close()
    saveValFile.close()
    writeCaffeDataClass(dataClass, outputPath)

def processCaffeData(inputPath, outputPath, flag, probability):
    if  "train_val"in flag.strip():
        processCaffeTrainAndValData(inputPath, outputPath, probability)
    elif  "train"in flag.strip():
        processCaffeTrainData(inputPath, outputPath, "train")
    elif  "val"in flag.strip():
        processCaffeTrainData(inputPath, outputPath, "val")

def processCaffeTestData(inputPath, outputPath):
    dataClass = getDataClass(inputPath)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    saveFilePath = os.path.join(outputPath, "val.txt")
    saveFile = open(saveFilePath, "w")
    for className in dataClass:
        dataClassDir = os.path.join(inputPath, className)
        imageList = list(getDirFiles(dataClassDir, "*.*"))
        for imageIndex, imagePath in enumerate(imageList):
            path, fileNameAndPost = os.path.split(imagePath)
            saveImageDir = os.path.join(outputPath, "val")
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)
            saveImagePath = os.path.join(saveImageDir, fileNameAndPost)
            saveFile.write("%s %s\n" % (fileNameAndPost, className))
            os.rename(imagePath, saveImagePath)
    saveFile.close()

def filterSameCaffeTestData(inputPath, testfFilePath, outputPath):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    testData = readCaffeTestFileData(testfFilePath)
    for imagePath in getDirFiles(inputPath, "*.*"):
        path, fileNameAndPost = os.path.split(imagePath)
        for imageName, className in testData:
            if fileNameAndPost in imageName:
                break
        else:
            saveImagePath = os.path.join(outputPath, fileNameAndPost)
            os.rename(imagePath, saveImagePath)

def main():
    print("start...")
    #showAllImageClass("C:/Users/lpj/Desktop/第一届TMD大赛数据/train", "*.*")
    #showAllImageSize("/home/wfw/lipj/faceData", "*.*")
    #processCaffeTestData("/home/wfw/lipj/TMD/test", "/home/wfw/lipj/TMD")
    #filterSameCaffeTestData("/home/wfw/lipj/test1", "/home/wfw/lipj/TMD/val.txt", "/home/wfw/lipj/output")
    processCaffeData("/home/wfw/lipj/li", "/home/wfw/lipj/TMD", "train", 5)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()
