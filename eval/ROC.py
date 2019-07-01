import os
import cv2
import numpy as np
from detection import *
from xml_read import xmlInfo
from detect_tool import Iou_cal
#import matplotlib.pyplot as plt

def drawROC(output,labelNames):
	for labelName in labelNames:
		f=open(output+labelName+'.txt', 'r')
		results=f.readlines()

		acc = []
		rec = []
		for result in results:
			obj = result[:-1].split(">>")
			acc.append(obj[1].split(":")[1])
			rec.append(obj[2].split(":")[1])

		color = [0, 255, 0]
		linewidth = 0.3
		plt.plot(acc, rec, 'r*-')#, color = color, linewidth = linewidth)
		plt.plot([0.0,1.1], [0.0,1.1], 'b-')
		plt.xlim(0.0,1.1)
		plt.ylim(0.0,1.1)
		plt.title('ROC for '+ labelName +' In YOLOV3')
		plt.xlabel('ACC')
		plt.ylabel('REC')
		plt.show()

def calAccAndRecall(objects,groudTruth,thresh_iou):
	Num_True_detection = 0

	True_detection = []
	Overlap_detection = []
	False_detection = [ele for ele in objects]   
	Loss_detection = [ele for ele in groudTruth]

	for i in range(len(groudTruth)):
		Iou_sameLabel = []
		for j in range(len(objects)):
			if False_detection[j] != 0 and groudTruth[i]['label_name'] == objects[j]['label_name'] and Iou_cal(objects[j]['box'], groudTruth[i]['box']) > thresh_iou:
				Iou_sameLabel.append(Iou_cal(objects[j]['box'], groudTruth[i]['box']))
			else:
				Iou_sameLabel.append(0)
		if sum(Iou_sameLabel) > 0:
			Loss_detection[i] = 0
			Num_True_detection+=1
			for Iou in Iou_sameLabel:
				if Iou == max(Iou_sameLabel):
					True_detection.append(objects[Iou_sameLabel.index(Iou)])
					False_detection[Iou_sameLabel.index(Iou)] = 0
				elif Iou > 0:
					#print('max(Iou_sameLabel): {}'.format(max(Iou_sameLabel)))
					Overlap_detection.append(objects[Iou_sameLabel.index(Iou)])
					False_detection[Iou_sameLabel.index(Iou)] = 0

	return Num_True_detection

def bouncing_box(model,test_path,img_size,thresh_conf,thresh_iou,output,labelList):
	Num_img = 0
	TPList = np.zeros(len(labelList))
	groundTruthLabelNum = np.zeros(len(labelList))
	objectsLabelNum = np.zeros(len(labelList))

	print('--------loading network done!----------------')
	
	for img_name in os.listdir(test_path):
		name = img_name.split('.')
		xml_path_list = test_path.split("/")[:-2]
		xml_path_list.append('Annotations')
		xml_path_list.append(name[0])
		xml_path = '/'.join(xml_path_list)+'.xml'
		if os.path.exists(xml_path):
			Num_img+=1
			print('Image {}, {} is processing!'.format(Num_img, img_name))

			#-------------Read groudtruth--------
			groudTruth = xmlInfo(xml_path, labelList)
			#-------------Achieve prediction--------
			objects = detect(model, test_path+img_name, img_size, labelList, thresh_conf)

			for labelName in labelList:
				groundTruthLabel = [ele for ele in groudTruth if ele['label_name']==labelName]
				objectsLabel = [ele for ele in objects if ele['label_name']==labelName]
				groundTruthLabelNum[labelList.index(labelName)] = groundTruthLabelNum[labelList.index(labelName)]+len(groundTruthLabel)
				objectsLabelNum[labelList.index(labelName)] = objectsLabelNum[labelList.index(labelName)]+len(objectsLabel)

				TP = calAccAndRecall(objectsLabel,groundTruthLabel,thresh_iou)
				TPList[labelList.index(labelName)] = TPList[labelList.index(labelName)]+TP
				#print("groundTruthLabelNum: {}".format(groundTruthLabelNum))
				#print("objectsLabelNum: {}".format(objectsLabelNum))
				#print("TPList :{}".format(TPList))

	for labelName in labelList:
		if groundTruthLabelNum[labelList.index(labelName)]==0 or objectsLabelNum[labelList.index(labelName)]==0:
			True_detection_rate = 0
		else:
			True_detection_rate = float(TPList[labelList.index(labelName)])/float(objectsLabelNum[labelList.index(labelName)])
			True_recall_rate = float(TPList[labelList.index(labelName)])/float(groundTruthLabelNum[labelList.index(labelName)])
		f = open(output+labelName+'.txt','a')
		f.write('Thresh: {} >> Accuracy: {} >> Recall: {} \n'.format(round(thresh_conf, 2),True_detection_rate,True_recall_rate))
		f.close()

def main():
	print("start...")
	img_size = [640, 352]
	cfg = "./cfg/yolov3-spp.cfg"
	model_name = "./weights/best.pt"
	test_path="/home/sugon/data/VOCdevkit/BerkeleyDet/val/"
	labelList = ["person"]
	output="./results/"
	step=0.02

	model = networkInit(cfg, img_size, model_name)
	for labelName in labelList:
		if os.path.exists(output + labelName + '.txt'):
			os.remove(output + labelName + '.txt')
	for thresh_conf in np.arange(0.01,1,step):
		bouncing_box(model,test_path,img_size,float(thresh_conf),float(0.5),output,labelList)
		print("process end!")
	#drawROC(output,labelList)

if __name__ == "__main__":
	main()
