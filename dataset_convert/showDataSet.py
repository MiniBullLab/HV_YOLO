# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .processDataSet import getDataClass, getDirFiles

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

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

if __name__ == "__main__":
    print("start...")
    # showAllImageClass("C:/Users/lpj/Desktop/第一届TMD大赛数据/train", "*.*")
    # showAllImageSize("/home/wfw/lipj/faceData", "*.*")
    print("End of game, have a nice day!")