import os
import cv2
import numpy as np
from detection import *
from xml_read import xmlInfo
from detect_tool import Iou_cal

def bouncing_box(test_path, cfg, model_name, img_size, labels, thresh_iou=0.5, Virsual="True", output="./eval/result.txt"):
	model = networkInit(cfg, img_size, model_name)

	Num_img = 0
	Num_True_detection = 0
	Num_False_detection = 0
	Num_Loss_detection = 0
	Total_groundTruth = 0
	Total_objects = 0
	
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
		groudTruth = xmlInfo(xml_path, labels)
		Total_groundTruth = Total_groundTruth + len(groudTruth)
		#-------------Achieve prediction--------
		objects = detect(model, test_path+img_name, img_size, labels, 0.24)
		Total_objects = Total_objects + len(objects)
		#--------------Find matches------------

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

		#--------------False_detection--------------
		while 0 in False_detection:
			False_detection.remove(0)
		Num_False_detection = Num_False_detection + len(False_detection)
		#-------------loss_detection----------------
		while 0 in Loss_detection:
			Loss_detection.remove(0)
		Num_Loss_detection = Num_Loss_detection + len(Loss_detection)

		#-------------virsual-----------------------
		if Virsual=='True':
			img = cv2.imread(test_path+img_name)
			font = cv2.FONT_HERSHEY_SIMPLEX
			#--------------True_detection--------------
			for o in range(len(True_detection)):
				label_name = True_detection[o]['label_name']
				score = True_detection[o]['score']
				xmin = True_detection[o]['box'][0]
				ymin = True_detection[o]['box'][1]
				xmax = True_detection[o]['box'][2]
				ymax = True_detection[o]['box'][3]
				cv2.putText(img,str('%s %d%%' % (label_name, int(score*100))),(xmin,ymin-10),font,0.5,(0,255,0),2)
				cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),1)
			#--------------Overlap_detection--------------
			for o in range(len(Overlap_detection)):
				label_name = Overlap_detection[o]['label_name']
				score = Overlap_detection[o]['score']
				xmin = Overlap_detection[o]['box'][0]
				ymin = Overlap_detection[o]['box'][1]
				xmax = Overlap_detection[o]['box'][2]
				ymax = Overlap_detection[o]['box'][3]
				cv2.putText(img,str('%s %d%%' % (label_name, int(score*100))),(xmin,ymin-10),font,0.5,(0,255,255),2)
				cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,255),1)
			#--------------False_detection--------------
			for p in range(len(False_detection)):
				label_name = False_detection[p]['label_name']
				xmin = int(False_detection[p]['box'][0])
				ymin = int(False_detection[p]['box'][1])
				xmax = int(False_detection[p]['box'][2])
				ymax = int(False_detection[p]['box'][3])
				cv2.putText(img,str('%s' % label_name),(xmin,ymin-10),font,0.5,(255,0,0),2)
				cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),1)
			#-------------loss_detection----------------
			for q in range(len(Loss_detection)):
				label_name = Loss_detection[q]['label_name']
				xmin = int(Loss_detection[q]['box'][0])
				ymin = int(Loss_detection[q]['box'][1])
				xmax = int(Loss_detection[q]['box'][2])
				ymax = int(Loss_detection[q]['box'][3])
				cv2.putText(img,str('%s' %label_name),(xmin,ymin-10),font,0.5,(0,0,255),2)
				cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
			#cv2.imwrite('/home/wissen/Test_results/result/'+name[0]+'.png', img)
			cv2.namedWindow("image",0)
			cv2.resizeWindow("image", 1280, 720)
			cv2.imshow('image', img)
			key = cv2.waitKey()
			if (key == 27 or key == 1048603):
				break

	if Total_objects==0 or Total_groundTruth==0:
		True_detection_rate = 0
		True_recall_rate = 0
		False_detection_rate = 0
		Loss_detection_rate = 0
	else:
		True_detection_rate = float(Num_True_detection)/float(Total_objects)*100
		True_recall_rate = float(Num_True_detection)/float(Total_groundTruth)*100
		False_detection_rate = float(Num_False_detection)/float(Total_objects)*100
		Loss_detection_rate = float(Num_Loss_detection)/float(Total_groundTruth)*100
	
	#print(float(Num_True_detection)/float(Total_objects)*100)
	print('-----------result-------------\n>> Recall: {}% \n>> Precious: {}% \n>> False_detection_rate: {}%\n>> Loss_detection_rate: {}%\n>> Loss_detection_per_img: {}'.format(float(True_recall_rate), True_detection_rate, False_detection_rate, Loss_detection_rate, float(Num_Loss_detection)/float(Num_img)))

	f = open(output, 'a')
	f.write('------------------------------\nmodel_name: {}, img_num: {}\n'.format(model_name, Num_img))
	f.write('-----------result-------------\n>> Recall: {}% \n>> Precious: {}% \n>> False_detection_rate: {}%\n>> Loss_detection_rate: {}%\n>> Loss_detection_per_img: {}\n'.format(True_recall_rate, True_detection_rate, False_detection_rate, Loss_detection_rate, float(Num_Loss_detection)/float(Num_img)))

def main():
	test_path = "/home/sugon/data/VOCdevkit/BerkeleyDet/val/"
	labels = ['person']
	img_size = [640, 352]
	cfg = "./cfg/yolov3-spp.cfg"
	model_name = "./weights/best.pt"

	bouncing_box(test_path, cfg, model_name, img_size, labels, Virsual="False")

if __name__ == '__main__':
	main()





