import numpy as np
import caffe
import cv2
import os
from easy_convert.inference.det2d import detection_utility
from easy_convert.helper.dirProcess import DirProcess


class CaffeYoloV3Inference():

	def __init__(self, model_def, model_weights):
		self.dir_process = DirProcess()
		self.image_size = (352, 640)  # h, w
		self.feat1Name = '628'
		self.feat2Name = '654'
		self.feat3Name = '680'
		self.class_list = ('bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck')
		self.classes = 1
		self.nms_threshold = 0.45
		self.thresh_conf = 0.24
		self.box_of_each_grid = 3
		self.biases = np.array([8.95, 8.57, 12.43, 26.71, 19.71, 14.43,
								26.36, 58.52, 36.09, 25.55, 64.42, 42.90, 96.44,
								79.10, 158.37, 115.59, 218.65, 192.90])
		self.net = caffe.Net(model_def, model_weights, caffe.TEST)

	def yolov3_detect(self, image_dir):
		for img_path in self.dir_process.getDirFiles(image_dir):
			im, img, img_shape = detection_utility.preProcess(img_path, self.image_size[1],
															self.image_size[0])

			#transformed_image = np.load('result.npy')

			self.net.blobs['data'].data[...] = img
			output = self.net.forward()
			feat1 = self.net.blobs[self.feat1Name].data[0]
			feat2 = self.net.blobs[self.feat2Name].data[0]
			feat3 = self.net.blobs[self.feat3Name].data[0]

			totalBoxes = []
			totalCount = 0
			mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
			for k in range(self.box_of_each_grid):
				feat = eval('feat'+str(k+1))

				boxes, count = detection_utility.get_yolo_detections(feat, feat.shape[2],
																	feat.shape[1], self.biases,
																	self.box_of_each_grid, self.classes,
																	img_shape[1], img_shape[0],
																	self.image_size[1], self.image_size[0],
																	self.thresh_conf, mask[k], 0)  # n, c, h, w
				totalBoxes += boxes
				totalCount += count
			results = detection_utility.detectYolo(totalBoxes, totalCount, self.classes, self.nms_threshold)
			detection_utility.draw_image(img_path, results, self.class_list, scale=0.8)
			key = cv2.waitKey()
			if key == 1048603 or key == 27:
				break


def main():
	caffe.set_device(0)
	caffe.set_mode_gpu()

	model_def = './data/caffe/yolov3.prototxt'
	model_weights = './data/caffe/yolov3.caffemodel'
	test = CaffeYoloV3Inference(model_def, model_weights)
	test.yolov3_detect("./data/image")


if __name__ == "__main__":
	main()
